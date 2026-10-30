"""PyTorch hooks for reading and editing native MLP neuron activations.

A "neuron" here is one coordinate of the MLP intermediate activation: the input to
the MLP's down/output projection. For gated MLPs (Llama, Qwen, Mistral) that is
``act(gate(x)) * up(x)``; for GPT-2 style MLPs it is ``act(fc(x))``. Hooking the
down projection's *input* gives the same definition on every architecture.

Each neuron has two poles, isolated with ``torch.clamp``: the positive pole is
``clamp(act, min=0)`` and the negative pole is ``clamp(act, max=0)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from functools import reduce
from typing import Any

import torch
from torch import nn

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.schemas import (
    ActivatingExample,
    DictionarySite,
    Histogram,
    Pole,
    RunEnvironment,
)

# Probed in order; the first path that resolves on the model wins.
LAYER_PATHS = (
    "model.layers",  # Llama, Qwen2, Mistral, Gemma, Phi-3
    "transformer.h",  # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",  # Pythia / GPT-NeoX
    "model.decoder.layers",  # OPT
    "transformer.layers",
    "layers",
)
DOWN_PROJ_PATHS = (
    "mlp.down_proj",  # Llama, Qwen2, Mistral, Gemma
    "mlp.c_proj",  # GPT-2
    "mlp.dense_4h_to_h",  # GPT-NeoX, Falcon
    "mlp.fc_out",  # GPT-J
    "mlp.fc2",  # Phi
    "fc2",  # OPT
    "feed_forward.w2",
)

# Attention output projection; its *input* is the concatenated head outputs
# [batch, seq, n_heads * head_dim], so head h is one contiguous slice of it.
ATTN_OUT_PATHS = (
    "self_attn.o_proj",  # Llama, Qwen2, Mistral, Gemma, Phi-3
    "attn.c_proj",  # GPT-2
    "attention.dense",  # GPT-NeoX
    "attn.out_proj",  # GPT-J
    "self_attn.out_proj",  # OPT
    "self_attention.dense",  # Falcon
)

# Normalisation feeding the MLP; dictionaries trained on TransformerLens activations read
# its input standardised without the gain and bias (``ln2.hook_normalized``).
PRE_MLP_NORM_PATHS = (
    "pre_feedforward_layernorm",  # Gemma 2/3 (whose post_attention_layernorm is not it)
    "post_attention_layernorm",  # Llama, Qwen2, Mistral, GPT-NeoX
    "ln_2",  # GPT-2
    "final_layer_norm",  # OPT
)

# Normalisation of the MLP's output before it joins the residual stream, where there is
# one; TransformerLens's ``hook_mlp_out``, which dictionaries reconstruct, is after it.
POST_MLP_NORM_PATHS = ("post_feedforward_layernorm",)  # Gemma 2/3

# Final normalisation before the unembedding; its *input* is the final residual stream.
FINAL_NORM_PATHS = (
    "model.norm",  # Llama, Qwen2, Mistral, Gemma
    "transformer.ln_f",  # GPT-2, GPT-J, Falcon
    "gpt_neox.final_layer_norm",  # Pythia / GPT-NeoX
    "model.decoder.final_layer_norm",  # OPT
    "model.final_layernorm",  # Phi
)

Inputs = str | Sequence[str] | torch.Tensor | Mapping[str, torch.Tensor]


class _StopForward(Exception):
    """Raised from a hook to end a forward pass once what it needs has been captured."""


Intervention = Callable[[torch.Tensor], torch.Tensor | None]
# Receives a dictionary's input x at its site and the MLP output; may return a new output.
DictionaryIntervention = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]


def _get_path(root: nn.Module, path: str) -> Any:
    return reduce(getattr, path.split("."), root)


def _try_path(root: nn.Module, path: str) -> Any | None:
    try:
        return _get_path(root, path)
    except AttributeError:
        return None


def ablate_head(head: int, head_dim: int, value: torch.Tensor | None = None) -> Intervention:
    """A ``head_interventions`` entry replacing one head's output with ``value`` (default 0)."""

    def intervene(z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        span = slice(head * head_dim, (head + 1) * head_dim)
        z[..., span] = 0 if value is None else value.to(z.dtype)
        return z

    return intervene


def ablate_pole(neuron_idx: int, pole: Pole) -> Intervention:
    """An ``interventions`` entry zeroing one neuron's ``pole``, leaving the other intact."""

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        act[..., neuron_idx] = act[..., neuron_idx] - isolate_pole(act[..., neuron_idx], pole)
        return act

    return intervene


def isolate_pole(act: torch.Tensor, pole: Pole) -> torch.Tensor:
    """Return only the component of ``act`` belonging to ``pole`` (signed)."""
    return torch.clamp(act, min=0) if pole == "positive" else torch.clamp(act, max=0)


def pole_strength(act: torch.Tensor, pole: Pole) -> torch.Tensor:
    """Non-negative magnitude of a pole: larger always means "more active"."""
    return torch.clamp(act, min=0) if pole == "positive" else -torch.clamp(act, max=0)


def replace_pole(target: torch.Tensor, source: torch.Tensor, pole: Pole) -> torch.Tensor:
    """Swap ``target``'s ``pole`` component for ``source``'s, leaving the other pole intact."""
    return target - isolate_pole(target, pole) + isolate_pole(source, pole)


def logits_of(output: Any) -> torch.Tensor:
    return output.logits if hasattr(output, "logits") else output


class NeuronHookRunner:
    """Attach temporary hooks to MLP intermediate activations of a causal LM.

    Works with any Hugging Face ``PreTrainedModel`` whose layout matches
    :data:`LAYER_PATHS` / :data:`DOWN_PROJ_PATHS`, or pass explicit paths.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        *,
        layer_path: str | None = None,
        down_proj_path: str | None = None,
        attn_out_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.layers, self.layer_path = self._resolve_layers(layer_path)
        self.down_proj_path = self._resolve_down_proj(down_proj_path)
        self._attn_out_path = attn_out_path  # resolved on first use: MLP audits don't need it
        self.model_revision: str | None = None  # Hugging Face commit, set by the loader

    def _resolve_layers(self, explicit: str | None) -> tuple[nn.ModuleList, str]:
        for path in (explicit,) if explicit else LAYER_PATHS:
            layers = _try_path(self.model, path)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers, path
        raise ArchitectureNotSupportedError(
            f"Could not find transformer layers on {type(self.model).__name__}; "
            "pass layer_path= explicitly."
        )

    def _resolve_down_proj(self, explicit: str | None) -> str:
        for path in (explicit,) if explicit else DOWN_PROJ_PATHS:
            if isinstance(_try_path(self.layers[0], path), nn.Module):
                return path
        raise ArchitectureNotSupportedError(
            f"Could not find the MLP down projection under {self.layer_path}[0]; "
            "pass down_proj_path= explicitly."
        )

    # -- model introspection -------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def down_proj(self, layer: int) -> nn.Module:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers}-layer model")
        return _get_path(self.layers[layer], self.down_proj_path)

    def attn_out(self, layer: int) -> nn.Module:
        """The attention output projection of ``layer`` (input: concatenated head outputs)."""
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers}-layer model")
        if self._attn_out_path is None:
            for path in ATTN_OUT_PATHS:
                if isinstance(_try_path(self.layers[0], path), nn.Module):
                    self._attn_out_path = path
                    break
            else:
                raise ArchitectureNotSupportedError(
                    f"Could not find the attention output projection under {self.layer_path}[0]; "
                    "pass attn_out_path= explicitly."
                )
        return _get_path(self.layers[layer], self._attn_out_path)

    @property
    def n_heads(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("num_attention_heads", "n_head", "num_heads"):
            if isinstance(getattr(config, name, None), int):
                return getattr(config, name)
        raise ArchitectureNotSupportedError("model config does not give the number of heads")

    def head_dim(self, layer: int) -> int:
        proj = self.attn_out(layer)
        width = int(proj.in_features) if hasattr(proj, "in_features") else proj.weight.shape[0]
        if width % self.n_heads:
            raise ArchitectureNotSupportedError(
                f"attention output width {width} is not a multiple of {self.n_heads} heads"
            )
        return width // self.n_heads

    def head_output_weights(self, layer: int, head: int) -> torch.Tensor:
        """W_O for one head: ``[d_model, head_dim]``, so its write is ``W_O @ z``."""
        if not 0 <= head < self.n_heads:
            raise IndexError(f"head {head} out of range for {self.n_heads} heads")
        proj, d = self.attn_out(layer), self.head_dim(layer)
        weight = proj.weight if hasattr(proj, "in_features") else proj.weight.T  # Conv1D
        return weight[:, head * d : (head + 1) * d].detach()

    def mlp_width(self, layer: int) -> int:
        proj = self.down_proj(layer)
        if hasattr(proj, "in_features"):
            return int(proj.in_features)
        # transformers' Conv1D stores weight as (in_features, out_features).
        return int(proj.weight.shape[0])

    def mlp_norm(self, layer: int) -> nn.Module:
        """The normalisation whose output is ``layer``'s MLP input."""
        for path in PRE_MLP_NORM_PATHS:
            module = _try_path(self.layers[layer], path)
            if isinstance(module, nn.Module):
                return module
        raise ArchitectureNotSupportedError(
            f"Could not find the norm before the MLP under {self.layer_path}[{layer}]"
        )

    def mlp_write(self, layer: int) -> nn.Module:
        """The module whose output is what ``layer``'s MLP adds to the residual stream: the
        norm after the MLP if it has one, else the down projection."""
        for path in POST_MLP_NORM_PATHS:
            module = _try_path(self.layers[layer], path)
            if isinstance(module, nn.Module):
                return module
        return self.down_proj(layer)

    def mlp(self, layer: int) -> nn.Module:
        """``layer``'s MLP block: the module that owns the down projection."""
        parent, _, _ = self.down_proj_path.rpartition(".")
        if not parent:
            raise ArchitectureNotSupportedError(
                f"the down projection {self.down_proj_path} is not inside an MLP module"
            )
        return _get_path(self.layers[layer], parent)

    def final_norm(self) -> nn.Module | None:
        for path in FINAL_NORM_PATHS:
            module = _try_path(self.model, path)
            if isinstance(module, nn.Module):
                return module
        return None

    def neuron_output_direction(self, layer: int, neuron_idx: int) -> torch.Tensor:
        """The residual-stream vector written by one unit of activation of this neuron."""
        proj = self.down_proj(layer)
        if hasattr(proj, "in_features"):
            return proj.weight[:, neuron_idx].detach()
        return proj.weight[neuron_idx, :].detach()

    # -- running -------------------------------------------------------------

    def encode(self, inputs: Inputs) -> dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            ids = inputs if inputs.dim() == 2 else inputs.unsqueeze(0)
            return {"input_ids": ids.to(self.device)}
        if isinstance(inputs, Mapping):
            return {k: v.to(self.device) for k, v in inputs.items()}
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to run on text inputs")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    @contextmanager
    def hooks(
        self,
        interventions: Mapping[int, Intervention] | None = None,
        output_interventions: Mapping[int, Intervention] | None = None,
        head_interventions: Mapping[int, Intervention] | None = None,
        dictionary_interventions: Mapping[int, tuple[DictionarySite, DictionaryIntervention]]
        | None = None,
    ) -> Iterator[None]:
        """Install hooks for the duration of the block.

        ``interventions[layer]`` receives the intermediate activation
        ``[batch, seq, d_mlp]`` and may return a replacement (or ``None`` to only
        observe). ``output_interventions[layer]`` does the same for the down
        projection's output ``[batch, seq, d_model]``, and ``head_interventions[layer]``
        for the attention heads' concatenated outputs ``[batch, seq, n_heads * head_dim]``.
        ``dictionary_interventions[layer] = (site, fn)`` calls ``fn(x, out)`` with a
        dictionary's input ``x`` read at ``site`` (see ``DictionaryProfile``) and the MLP's
        write to the residual stream (:meth:`mlp_write`), which ``fn`` may replace.
        """
        handles = []

        def pre_hook(fn: Intervention):
            def hook(_module, args):
                replaced = fn(args[0])
                return None if replaced is None else (replaced, *args[1:])

            return hook

        def post_hook(fn: Intervention):
            def hook(_module, _args, output):
                return fn(output)

            return hook

        try:
            for layer, fn in (interventions or {}).items():
                handles.append(self.down_proj(layer).register_forward_pre_hook(pre_hook(fn)))
            for layer, fn in (output_interventions or {}).items():
                handles.append(self.down_proj(layer).register_forward_hook(post_hook(fn)))
            for layer, fn in (head_interventions or {}).items():
                handles.append(self.attn_out(layer).register_forward_pre_hook(pre_hook(fn)))
            for layer, (site, fn) in (dictionary_interventions or {}).items():
                from openinterp.harness.sites import site_hooks

                handles += site_hooks(self, layer, site, fn)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return logits_of(self.model(**encoded))

    @torch.no_grad()
    def run_with_cache(
        self, inputs: Inputs, layers: Sequence[int]
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run the model, returning logits and intermediate activations for ``layers``."""
        cache: dict[int, torch.Tensor] = {}

        def recorder(layer: int) -> Intervention:
            def record(act: torch.Tensor) -> None:
                cache[layer] = act.detach().clone()

            return record

        encoded = self.encode(inputs)
        with self.hooks({layer: recorder(layer) for layer in layers}):
            logits = self.forward(encoded)
        return logits, cache

    @torch.no_grad()
    def run_with_interventions(
        self,
        inputs: Inputs,
        interventions: Mapping[int, Intervention] | None = None,
        output_interventions: Mapping[int, Intervention] | None = None,
        head_interventions: Mapping[int, Intervention] | None = None,
        dictionary_interventions: Mapping[int, tuple[DictionarySite, DictionaryIntervention]]
        | None = None,
    ) -> torch.Tensor:
        encoded = self.encode(inputs)
        with self.hooks(
            interventions, output_interventions, head_interventions, dictionary_interventions
        ):
            return self.forward(encoded)

    # -- attention heads -----------------------------------------------------

    @torch.no_grad()
    def head_outputs(self, inputs: Inputs, layer: int) -> torch.Tensor:
        """Every head's output ``z`` in ``layer``: ``[batch, seq, n_heads, head_dim]``.

        The forward pass stops once ``layer``'s heads have run: later layers can't change them.
        """
        captured: list[torch.Tensor] = []

        def capture(z: torch.Tensor) -> None:
            captured.append(z.detach().clone())
            raise _StopForward

        with suppress(_StopForward):
            self.run_with_interventions(inputs, head_interventions={layer: capture})
        z = captured[-1]
        return z.reshape(*z.shape[:-1], self.n_heads, self.head_dim(layer))

    @torch.no_grad()
    def head_write_norm(self, inputs: Inputs, layer: int, head: int) -> torch.Tensor:
        """``|W_O z|`` per token ``[batch, seq]``: how much the head writes to the residual."""
        z = self.head_outputs(inputs, layer)[..., head, :]
        w_o = self.head_output_weights(layer, head).to(z.dtype)
        return (z @ w_o.T).float().norm(dim=-1)

    @torch.no_grad()
    def attention_patterns(self, inputs: Inputs, layer: int) -> torch.Tensor:
        """Attention weights of ``layer``: ``[batch, n_heads, query, key]``.

        Needs eager attention (load with ``attn_implementation="eager"``): SDPA and
        flash attention don't return the weights.
        """
        output = self.model(**self.encode(inputs), output_attentions=True)
        attentions = getattr(output, "attentions", None)
        if not attentions or attentions[layer] is None:
            raise ArchitectureNotSupportedError(
                'the model returned no attention weights; load it with attn_implementation="eager"'
            )
        return attentions[layer]

    @torch.no_grad()
    def neuron_activations(
        self, inputs: Inputs, layer: int, neuron_idx: int, pole: Pole | None = None
    ) -> torch.Tensor:
        """Per-token activation ``[batch, seq]`` of one neuron, optionally as a pole strength."""
        _, cache = self.run_with_cache(inputs, [layer])
        act = cache[layer][..., neuron_idx]
        return act if pole is None else pole_strength(act, pole)

    def max_pole_strength(self, text: str, layer: int, neuron_idx: int, pole: Pole) -> float:
        return float(self.neuron_activations(text, layer, neuron_idx, pole).max())


def run_environment(runner: NeuronHookRunner) -> RunEnvironment:
    """The device, weights dtype, model revision and library versions of ``runner``."""
    from openinterp import __version__

    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    config = getattr(runner.model, "config", None)
    return RunEnvironment(
        device=str(runner.device),
        dtype=str(next(runner.model.parameters()).dtype).removeprefix("torch."),
        model_revision=getattr(runner, "model_revision", None)
        or getattr(config, "_commit_hash", None),
        openinterp_version=__version__,
        torch_version=torch.__version__,
        transformers_version=transformers_version,
    )


def top_activating_examples(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 20,
    batch_tokens: int | None = None,
) -> tuple[list[ActivatingExample], list[float]]:
    """Scan ``texts``; return the ``k`` strongest snippets and every token's pole strength.

    A thin wrapper over :func:`openinterp.harness.scan.scan_neuron`, which also
    gathers correlated neurons and the confound profile.
    """
    from openinterp.harness.scan import BATCH_TOKENS, scan_neuron

    batch_tokens = batch_tokens or BATCH_TOKENS
    result = scan_neuron(runner, texts, layer, neuron_idx, pole, k=k, batch_tokens=batch_tokens)
    return result.examples, result.values


def activation_histogram(values: Sequence[float], bins: int = 30) -> Histogram:
    tensor = torch.tensor(list(values) or [0.0], dtype=torch.float32)
    low, high = float(tensor.min()), float(tensor.max())
    if high <= low:
        high = low + 1e-6
    counts = torch.histc(tensor, bins=bins, min=low, max=high)
    edges = torch.linspace(low, high, bins + 1)
    return Histogram(
        bin_edges=[round(float(e), 6) for e in edges],
        counts=[int(c) for c in counts],
    )
