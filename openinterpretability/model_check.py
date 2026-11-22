"""Check that a model can join the registry, before it is configured (#3370).

``open-interpretability model-check <name>`` runs these checks, and
``docs/adding-a-model.md`` gives the process around them. Each check gives a
:class:`Finding`:

- **Layout.** The hooks find the layers, the MLP down projection, the attention output,
  the norm before the MLP and the final norm. The finding also gives the MLP's family,
  gated or plain, because it changes what a negative pole means, and whether the block is
  parallel.
- **Smoke.** One forward pass records the last layer's neurons. Ablating the strongest
  pole changes the logits, and the final norm reads a residual stream of the model's width.
- **Config.** The model is in ``[tool.openinterpretability] models``, it has a licence
  entry, and CI's ``ALLOWED_MODELS`` lists it. Before the model is configured, these are
  warnings, so that the check can run first.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.harness.hooks import NeuronHookRunner, _try_path, ablate_pole
from openinterpretability.planner import configured_licenses, configured_models

# The gate projection of a gated MLP (SwiGLU or GeGLU). The paths are those of Llama,
# Qwen2, Mistral and Gemma, of Mixtral-style experts, and of Phi-3's fused projection.
GATE_PATHS = ("gate_proj", "w1", "gate_up_proj")
SMOKE_TEXT = "The cat sat on the mat because the sun had warmed it."
CI_WORKFLOW = Path(".github/workflows/ci.yml")


@dataclass(frozen=True)
class Finding:
    """One check's result: ``status`` is ``pass``, ``warn`` or ``fail``."""

    item: str
    status: str
    detail: str

    def line(self) -> str:
        """The finding as one line of the report.

        Returns:
            The status in capitals, the item and the detail.

        Examples:
            >>> Finding("final norm", "fail", "none found").line()
            'FAIL  final norm: none found'
        """
        return f"{self.status.upper():5} {self.item}: {self.detail}"


def _probe(item: str, read: Callable[[], str]) -> Finding:
    """Pass with what ``read`` gives, or fail with the error it raises.

    Args:
        item: The name of the check.
        read: Gives the detail, or raises when the model lacks what it reads.

    Returns:
        A ``pass`` finding with the detail, or a ``fail`` finding with the error.
    """
    try:
        return Finding(item, "pass", read())
    except (ArchitectureNotSupportedError, AttributeError, IndexError) as exc:
        return Finding(item, "fail", str(exc))


def mlp_family(runner: NeuronHookRunner) -> str:
    """``gated`` or ``plain``, with the activation that the model's config names.

    A gated neuron is ``act(gate(x)) * up(x)``, so its negative pole can be large. A plain
    neuron is ``act(fc(x))``, so with GELU its negative pole is at most about 0.17.

    Args:
        runner: The loaded model.

    Returns:
        ``gated`` or ``plain``, a comma, and the activation's name.

    Examples:
        >>> from transformers import GPT2Config, GPT2LMHeadModel
        >>> config = GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16)
        >>> mlp_family(NeuronHookRunner(GPT2LMHeadModel(config)))
        'plain, gelu_new'
    """
    mlp = runner.mlp(0)
    gated = any(isinstance(_try_path(mlp, path), torch.nn.Module) for path in GATE_PATHS)
    config = getattr(runner.model, "config", None)
    act = getattr(config, "hidden_act", None) or getattr(config, "activation_function", None)
    return f"{'gated' if gated else 'plain'}, {act or 'activation not in the config'}"


def _final_norm(runner: NeuronHookRunner) -> str:
    """The final norm's class name.

    Args:
        runner: The loaded model.

    Returns:
        The class name of the norm before the unembedding.

    Raises:
        ArchitectureNotSupportedError: No path in ``FINAL_NORM_PATHS`` resolves.
    """
    norm = runner.final_norm()
    if norm is None:
        raise ArchitectureNotSupportedError("no final norm found (FINAL_NORM_PATHS)")
    return type(norm).__name__


def _attention(runner: NeuronHookRunner) -> str:
    """The number of heads and their width.

    Args:
        runner: The loaded model.

    Returns:
        ``<n> heads of width <d>``.
    """
    runner.attn_out(0)  # first, so that a missing projection names its paths
    return f"{runner.n_heads} heads of width {runner.head_dim(0)}"


def _residual(runner: NeuronHookRunner) -> str:
    """Whether the block is parallel or sequential.

    Args:
        runner: The loaded model.

    Returns:
        The block's shape, and what it means for the MLP's input.
    """
    from openinterpretability.harness.wiring import parallel_residual

    if parallel_residual(runner):
        return "parallel: attention and the MLP read the same input"
    return "sequential: the MLP reads its own layer's heads"


def layout(runner: NeuronHookRunner) -> list[Finding]:
    """Where the hooks attach, the MLP's family and the block's shape.

    Args:
        runner: The loaded model.

    Returns:
        Seven findings: layers, MLP, MLP family, attention, both norms and the block.

    Examples:
        >>> from transformers import GPT2Config, GPT2LMHeadModel
        >>> config = GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=16)
        >>> found = layout(NeuronHookRunner(GPT2LMHeadModel(config)))
        >>> [f.status for f in found] == ["pass"] * 7
        True
        >>> found[0].detail
        '2 at transformer.h'
    """
    return [
        Finding("layers", "pass", f"{runner.n_layers} at {runner.layer_path}"),
        _probe("MLP", lambda: f"{runner.mlp_width(0)} neurons, read at {runner.down_proj_path}"),
        _probe("MLP family", lambda: mlp_family(runner)),
        _probe("attention", lambda: _attention(runner)),
        _probe("norm before the MLP", lambda: type(runner.mlp_norm(0)).__name__),
        _probe("final norm", lambda: _final_norm(runner)),
        _probe("block", lambda: _residual(runner)),
    ]


def _ablation(runner: NeuronHookRunner, inputs) -> Finding:
    """Ablate the last layer's strongest pole, and check that the logits move.

    Args:
        runner: The loaded model.
        inputs: Text or token ids.

    Returns:
        The ``ablation`` finding, or a failed ``neurons`` finding for values not finite.
    """
    last = runner.n_layers - 1
    acts = runner.layer_activations(inputs, [last])[last]
    if not torch.isfinite(acts).all():
        return Finding("neurons", "fail", f"layer {last} gives values that are not finite")
    peak = acts.flatten(0, -2).abs().max(dim=0).values.argmax().item()
    pole = "positive" if acts[..., peak].max() > 0 else "negative"
    base = runner.run_with_interventions(inputs)
    ablated = runner.run_with_interventions(inputs, {last: ablate_pole(peak, pole)})
    if torch.equal(base, ablated):
        return Finding("ablation", "fail", f"ablating L{last}/N{peak} {pole} leaves the logits")
    return Finding("ablation", "pass", f"ablating L{last}/N{peak} {pole} moves the logits")


def _final_read(runner: NeuronHookRunner, inputs) -> Finding:
    """Read the final norm's input once, and check its width against the residual stream.

    Args:
        runner: The loaded model.
        inputs: Text or token ids.

    Returns:
        The ``final read`` finding.
    """
    norm = runner.final_norm()
    if norm is None:
        return Finding("final read", "fail", "no final norm to read")
    seen: list[int] = []
    handle = norm.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape[-1]))
    try:
        runner.run_with_interventions(inputs)
    finally:
        handle.remove()
    width = runner.neuron_output_direction(0, 0).numel()
    if seen != [width]:
        return Finding("final read", "fail", f"read widths {seen}, residual stream {width}")
    return Finding("final read", "pass", f"the final norm reads a {width}-wide residual stream")


def smoke(runner: NeuronHookRunner, inputs=SMOKE_TEXT) -> list[Finding]:
    """One small run: the neurons record, an ablation moves the logits, the final norm reads.

    Args:
        runner: The loaded model.
        inputs: Text, for a model with a tokenizer, or token ids.

    Returns:
        The ``ablation`` (or ``neurons``) finding, then the ``final read`` finding.

    Examples:
        >>> from transformers import GPT2Config, GPT2LMHeadModel
        >>> _ = torch.manual_seed(0)
        >>> config = GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=16)
        >>> runner = NeuronHookRunner(GPT2LMHeadModel(config).eval())
        >>> [f.status for f in smoke(runner, torch.tensor([1, 2, 3, 4]))]
        ['pass', 'pass']
    """
    return [_ablation(runner, inputs), _final_read(runner, inputs)]


def _ci_models(ci: Path) -> list[str] | None:
    """The ``ALLOWED_MODELS`` of CI's workflow, or None outside a clone.

    Args:
        ci: The path of CI's workflow file.

    Returns:
        The model names, an empty list if the file names none, or None with no file.
    """
    if not ci.is_file():
        return None
    found = re.search(r"^\s*ALLOWED_MODELS:\s*(.+?)\s*$", ci.read_text(), re.M)
    return found.group(1).split() if found else []


def config_findings(
    name: str, pyproject: Path = Path("pyproject.toml"), ci: Path = CI_WORKFLOW
) -> list[Finding]:
    """Whether ``name`` is configured, has a licence entry and is in CI's allow list.

    A gap fails for a configured model and warns for a model that isn't configured yet.

    Args:
        name: The model's Hub name.
        pyproject: The file whose ``[tool.openinterpretability]`` table is read.
        ci: CI's workflow file, which holds ``ALLOWED_MODELS``.

    Returns:
        The ``models``, ``licence`` and ``CI`` findings.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _ = (root / "p.toml").write_text('[tool.openinterpretability]\\nmodels = ["a/x"]\\n')
        >>> [f.status for f in config_findings("a/x", root / "p.toml", root / "ci.yml")]
        ['pass', 'fail', 'warn']
        >>> [f.status for f in config_findings("b/y", root / "p.toml", root / "ci.yml")]
        ['warn', 'warn', 'warn']
    """
    configured = name in configured_models(pyproject)
    gap = "fail" if configured else "warn"
    allowed = _ci_models(ci)
    return [
        Finding("models", "pass" if configured else "warn",
                "configured" if configured else "not in [tool.openinterpretability] models"),
        Finding("licence", "pass" if name in configured_licenses(pyproject) else gap,
                "an entry in [tool.openinterpretability.licenses]"),
        Finding("CI", "warn" if allowed is None else "pass" if name in allowed else gap,
                f"ALLOWED_MODELS in {ci}" if allowed is not None else f"no {ci} here"),
    ]  # fmt: skip


def check(runner: NeuronHookRunner, name: str, inputs=SMOKE_TEXT) -> list[Finding]:
    """Every finding for the model that ``runner`` holds, under its Hub ``name``.

    The smoke run is skipped when the layout fails, since its hooks would not attach.

    Args:
        runner: The loaded model.
        name: The model's Hub name, for the config checks.
        inputs: Text or token ids for the smoke run.

    Returns:
        The layout findings, then the smoke findings if the layout passed, then the config's.

    Examples:
        >>> from transformers import GPT2Config, GPT2LMHeadModel
        >>> config = GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=16)
        >>> runner = NeuronHookRunner(GPT2LMHeadModel(config).eval())
        >>> found = check(runner, "openai-community/gpt2", torch.tensor([1, 2, 3]))
        >>> len(found), any(f.status == "fail" for f in found)
        (12, False)
    """
    found = layout(runner)
    if not any(f.status == "fail" for f in found):
        found += smoke(runner, inputs)
    return found + config_findings(name)
