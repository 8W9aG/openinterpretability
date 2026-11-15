"""Explain one answer of a model as fully as the registry allows (issue #3250).

``open-interpretability explain "<prompt>"`` attributes the logit of the model's answer to every
MLP neuron pole. It labels the poles from the registry, and says how much of the answer is
explained. A sparse registry is fine: the output's job is also to show how little is
explained so far.

- *Attribute.* One prompt has no corrupted twin, so the baseline is each neuron's mean
  activation over a corpus sample. At each position, a neuron's contribution is
  (a − mean) · ∂logit/∂a. That is first order, like ``attribution_patching``. Each
  contribution goes to the pole that the sign of ``a`` names at that position. The
  first token, an attention sink, is left out, as scans leave it out.
- *Explain.* Each pole gets its newest record's status and explanation, or *stopped*
  (a scan found it dead or default-on), or *unstudied*. Only a VERIFIED explanation is
  shown as one.
- *Gaps.* The share of the total |attribution| that VERIFIED poles hold.
- *Check.* Mean-ablate single top poles, then the VERIFIED ones together, and measure
  the change in the answer's logit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from openinterp.core.cache_dirs import cache_root
from openinterp.harness.hooks import NeuronHookRunner

MEANS_ENV = "OPENINTERP_MEANS"  # the cache of per-neuron means, or "off"
WINDOW = 128  # tokens per window of the mean's corpus sample
MEANS_VERSION = 2  # in the cache key: 2 leaves each window's first token out


@dataclass
class PoleAttribution:
    """One pole's attribution to the answer, with what the registry says of it."""

    layer: int
    neuron: int
    pole: str
    attribution: float  # summed (a − mean) · grad over the positions where it is this pole
    strength: float  # the pole's peak strength on the prompt
    status: str = "unstudied"  # VERIFIED, NEEDS_REVISION, REJECTED, stopped, unstudied
    explanation: str | None = None
    max_activation: float | None = None
    ablation_change: float | None = None  # target logit change when mean-ablated alone

    @property
    def label(self) -> str:
        """``L3/N7+``: the pole within its model."""
        return f"L{self.layer}/N{self.neuron}{'+' if self.pole == 'positive' else '−'}"


@dataclass
class Explanation:
    """Everything ``open-interpretability explain`` reports for one prompt."""

    prompt: str
    position: int
    token: str
    logit: float
    poles: list[PoleAttribution]  # the top poles, by |attribution|
    total: float  # the sum of |attribution| over every pole of every layer
    verified: float  # the same sum over VERIFIED poles only
    joint: dict = field(default_factory=dict)  # the VERIFIED poles ablated together

    @property
    def verified_share(self) -> float:
        """The share of the total |attribution| that VERIFIED poles hold."""
        return self.verified / self.total if self.total else 0.0


def _layer_acts(runner: NeuronHookRunner, encoded, layers) -> dict[int, torch.Tensor]:
    _, cache = runner.run_with_cache(encoded, layers)
    return cache


def neuron_means(runner: NeuronHookRunner, model: str, texts: Sequence[str], tokens: int):
    """Each layer's mean MLP activation per neuron, ``{layer: tensor[d_mlp]}``, cached.

    The sample is the first ``tokens`` tokens of ``texts``, in windows of :data:`WINDOW`.
    Each window's first token is left out, as the attribution leaves out the prompt's.
    """
    tag = f"{model}|{tokens}|{WINDOW}|{MEANS_VERSION}|"
    key = hashlib.sha256(tag.encode() + "\n".join(texts).encode())
    root = cache_root(MEANS_ENV, "means")
    path = root / f"{key.hexdigest()[:24]}.pt" if root else None
    if path is not None and path.exists():
        return torch.load(path)
    ids = [i for t in texts for i in runner.tokenizer.encode(t)][:tokens]
    windows = [ids[i : i + WINDOW] for i in range(0, len(ids) - WINDOW + 1, WINDOW)]
    layers = list(range(runner.n_layers))
    sums = {la: torch.zeros(runner.mlp_width(la)) for la in layers}
    count = 0
    for window in windows:
        acts = _layer_acts(runner, torch.tensor([window]), layers)
        for la in layers:  # a window's first token is an attention sink: leave it out
            sums[la] += acts[la][0, 1:].float().sum(0).cpu()
        count += len(window) - 1
    means = {la: s / max(count, 1) for la, s in sums.items()}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(means, path)
    return means


def attribute(runner: NeuronHookRunner, prompt: str, means: dict, token: int, position: int):
    """Per pole, ``{(layer, neuron, pole): (attribution, peak strength)}``, and the logit.

    The metric is the logit of ``token`` at ``position``. The prompt's first token is left out.
    """
    layers = sorted(means)
    captured: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(act: torch.Tensor) -> None:
            captured[layer] = act

        return hook

    def ensure_grad(_module, _args, output):
        return output if output.requires_grad else output.detach().requires_grad_(True)

    encoded = runner.encode(prompt)
    handle = runner.model.get_input_embeddings().register_forward_hook(ensure_grad)
    try:
        with torch.enable_grad(), runner.hooks({la: capture(la) for la in layers}):
            logit = runner.forward(encoded)[0, position, token]
            grads = torch.autograd.grad(logit, [captured[la] for la in layers])
    finally:
        handle.remove()
    out = {}
    for la, grad in zip(layers, grads, strict=True):
        # The first token is an attention sink, active far beyond any scan's scale whatever
        # the prompt. The protocol leaves it out of scans, and so does this.
        act = captured[la].detach()[0, 1:].float().cpu()
        contribution = (act - means[la]) * grad[0, 1:].float().cpu()
        for pole, mask in (("positive", act > 0), ("negative", act < 0)):
            score = (contribution * mask).sum(0)
            peak = (act.abs() * mask).amax(0)
            for n in torch.nonzero(mask.any(0)).flatten().tolist():
                out[(la, n, pole)] = (float(score[n]), float(peak[n]))
    return out, float(logit.detach())


def mean_ablation(poles: Sequence[tuple[int, int, str]], means: dict):
    """``interventions`` that set each pole to its neuron's mean where the pole is active.

    The first token keeps its value, as the attribution leaves it out.
    """
    by_layer: dict[int, list[tuple[int, str]]] = {}
    for layer, neuron, pole in poles:
        by_layer.setdefault(layer, []).append((neuron, pole))

    def at(layer: int):
        def intervene(act: torch.Tensor) -> torch.Tensor:
            act = act.clone()
            for neuron, pole in by_layer[layer]:
                a = act[..., neuron]
                active = a > 0 if pole == "positive" else a < 0
                active[..., 0] = False  # the first token stays, as in the attribution
                mean = means[layer][neuron].to(act.dtype)
                act[..., neuron] = torch.where(active, mean, a)
            return act

        return intervene

    return {layer: at(layer) for layer in by_layer}


@torch.no_grad()
def logits_with(runner: NeuronHookRunner, prompt: str, interventions, position: int):
    """The logits at ``position`` with ``interventions`` applied."""
    with runner.hooks(interventions):
        return runner.forward(runner.encode(prompt))[0, position].float().cpu()


def registry_labels(wiki_dir: Path, model: str) -> dict[str, tuple[str, str | None, float | None]]:
    """``{pole slug: (status, explanation, max_activation)}`` for ``model``'s poles.

    The status is the newest record's verdict. A pole that a scan stopped on, with no
    record, is ``stopped``.
    """
    from openinterp.planner import load_states, stopped

    out: dict[str, tuple[str, str | None, float | None]] = {}
    for ident in stopped(wiki_dir):
        if ident.model_name == model:
            out[ident.slug] = ("stopped", None, None)
    for slug, state in load_states(wiki_dir).items():
        sub = state.record.submission
        if sub.identifier.model_name == model:
            out[slug] = (
                state.record.verdict.status,
                sub.natural_language_explanation,
                sub.max_activation,
            )
    return out


def explain(
    runner: NeuronHookRunner,
    model: str,
    prompt: str,
    means: dict,
    labels: dict,
    *,
    token: int | None = None,
    position: int = -1,
    top: int = 20,
    confirm: int = 10,
) -> Explanation:
    """Attribute, label, measure the VERIFIED share and check by ablation (module docstring)."""
    from openinterp.core.identifiers import NeuronIdentifier

    base = logits_with(runner, prompt, {}, position)
    token = int(base.argmax()) if token is None else token
    scores, logit = attribute(runner, prompt, means, token, position)
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
    total = verified = 0.0
    poles = []
    for rank, ((layer, neuron, pole), (score, strength)) in enumerate(ranked):
        slug = NeuronIdentifier(model_name=model, layer=layer, neuron_idx=neuron, pole=pole).slug
        status, text, max_act = labels.get(slug, ("unstudied", None, None))
        total += abs(score)
        verified += abs(score) if status == "VERIFIED" else 0.0
        if rank < top:
            shown = text if status == "VERIFIED" else None
            poles.append(
                PoleAttribution(layer, neuron, pole, score, strength, status, shown, max_act)
            )
    for p in poles[:confirm]:
        ablated = logits_with(
            runner, prompt, mean_ablation([(p.layer, p.neuron, p.pole)], means), position
        )
        p.ablation_change = float(ablated[token] - base[token])
    chosen = [(p.layer, p.neuron, p.pole) for p in poles if p.status == "VERIFIED"]
    joint = {"poles": len(chosen)}
    if chosen:
        ablated = logits_with(runner, prompt, mean_ablation(chosen, means), position)
        joint |= {
            "logit_change": float(ablated[token] - base[token]),
            "top_token_changed": int(ablated.argmax()) != token,
            "new_top_token": runner.tokenizer.decode([int(ablated.argmax())]),
        }
    return Explanation(
        prompt, position, runner.tokenizer.decode([token]), logit, poles, total, verified, joint
    )
