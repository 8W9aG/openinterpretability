"""Explain one answer of a model as fully as the registry allows (issues #3250, #3617).

``open-interpretability explain "<prompt>"`` attributes the logit of the model's answer to
every MLP neuron pole and every attention head. It labels the units from the registry, and
says how much of the answer is explained. A sparse registry is fine: the output's job is
also to show how little is explained so far.

- *Attribute.* One prompt has no corrupted twin, so the baseline is each neuron's mean
  activation over a corpus sample. At each position, a neuron's contribution is
  (a − mean) · ∂logit/∂a. That is first order, like ``attribution_patching``. Each
  contribution goes to the pole that the sign of ``a`` names at that position. The
  first token, an attention sink, is left out, as scans leave it out. A head's
  contribution is (z − z̄) · ∂logit/∂z over its slice of the attention output's input.
  A head has no pole.
- *Explain.* Each unit gets its newest record's status and explanation, or *stopped*
  (a scan found it dead or default-on), or *unstudied*. Only a VERIFIED explanation is
  shown as one.
- *Gaps.* The share of the total |attribution| that VERIFIED units hold.
- *Check.* Mean-ablate single top units, then the VERIFIED ones together, and measure
  the change in the answer's logit.
- *Paths.* The same rule attributes each top unit's own contribution to the units
  upstream of it. The logit's gradient stays fixed. The result is a set of first-order
  edges between the native units, as in the attribution graphs of Ameisen et al. (2025).
  Unlike those graphs, they need no replacement model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from openinterpretability.core.cache_dirs import cache_root
from openinterpretability.explain_groups import HEAD, GroupAttribution, GroupDef
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.progress import Progress

MEANS_ENV = "OPENINTERPRETABILITY_MEANS"  # the cache of per-neuron means, or "off"
WINDOW = 128  # tokens per window of the mean's corpus sample
MEANS_VERSION = 3  # in the cache key: 2 leaves each window's first token out; 3 adds heads


@dataclass
class PoleAttribution:
    """One unit's attribution to the answer, with what the registry says of it.

    The unit is a neuron pole, or an attention head when ``pole`` is :data:`HEAD`. Then
    ``neuron`` holds the head's index.
    """

    layer: int
    neuron: int
    pole: str
    attribution: float  # summed (a − mean) · grad over the positions where it is this pole
    strength: float  # the pole's peak strength on the prompt
    status: str = "unstudied"  # VERIFIED, NEEDS_REVISION, REJECTED, stopped, unstudied
    explanation: str | None = None
    max_activation: float | None = None
    ablation_change: float | None = None  # target logit change when mean-ablated alone
    hub: dict | None = None  # a HubNote, when the unit is a hub's center (#4472)

    @property
    def label(self) -> str:
        """``L3/N7+`` for a pole, ``L3/H2`` for a head."""
        if self.pole == HEAD:
            return f"L{self.layer}/H{self.neuron}"
        return f"L{self.layer}/N{self.neuron}{'+' if self.pole == 'positive' else '−'}"

    @property
    def key(self) -> tuple[int, int, str]:
        """``(layer, neuron or head, pole or HEAD)``, the key of :func:`attribute`."""
        return self.layer, self.neuron, self.pole


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
    joint: dict = field(default_factory=dict)  # the VERIFIED units ablated together
    edges: list[dict] = field(default_factory=list)  # {source, target, weight} between top units
    groups: list[GroupAttribution] = field(default_factory=list)  # all of them, by |attribution|
    order: list = field(default_factory=list, repr=False)  # every unit, ranked (_ranked)
    token_id: int = -1  # the answer token's id
    net: float = 0.0  # the signed attribution of every unit, summed (the page's waterfall)

    @property
    def verified_share(self) -> float:
        """The share of the total unit |attribution| that is explained (:func:`_ranked`)."""
        return self.verified / self.total if self.total else 0.0


@dataclass
class Means:
    """The attribution's baseline: each layer's mean neuron activation and head output."""

    neurons: dict[int, torch.Tensor]  # layer → [d_mlp]
    heads: dict[int, torch.Tensor]  # layer → [n_heads × head_dim]; empty without heads


def has_heads(runner: NeuronHookRunner) -> bool:
    """Whether the runner can address the model's attention heads."""
    from openinterpretability.harness.hooks import ArchitectureNotSupportedError

    try:
        return runner.attn_out(0) is not None and runner.n_heads > 0
    except ArchitectureNotSupportedError:
        return False


def _observe(store: dict[int, torch.Tensor], layer: int):
    def hook(act: torch.Tensor) -> None:
        store[layer] = act

    return hook


def _captured(runner: NeuronHookRunner, encoded, heads: bool):
    """Logits, and each layer's neuron activations and head outputs, from one forward."""
    neurons: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    layers = range(runner.n_layers)
    head_hooks = {la: _observe(outputs, la) for la in layers} if heads else None
    with runner.hooks({la: _observe(neurons, la) for la in layers}, None, head_hooks):
        logits = runner.forward(encoded)
    return logits, neurons, outputs


@torch.no_grad()
def unit_means(runner: NeuronHookRunner, model: str, texts: Sequence[str], tokens: int) -> Means:
    """Each layer's mean neuron activation and head output, cached (:class:`Means`).

    The sample is the first ``tokens`` tokens of ``texts``, in windows of :data:`WINDOW`.
    Each window's first token is left out, as the attribution leaves out the prompt's.
    """
    tag = f"{model}|{tokens}|{WINDOW}|{MEANS_VERSION}|"
    key = hashlib.sha256(tag.encode() + "\n".join(texts).encode())
    root = cache_root(MEANS_ENV, "means")
    path = root / f"{key.hexdigest()[:24]}.pt" if root else None
    if path is not None and path.exists():
        return Means(**torch.load(path))
    ids = [i for t in texts for i in runner.tokenizer.encode(t)][:tokens]
    windows = [ids[i : i + WINDOW] for i in range(0, len(ids) - WINDOW + 1, WINDOW)]
    heads = has_heads(runner)
    sums: list[dict[int, torch.Tensor]] = [{}, {}]
    progress = Progress("windows read for the mean activations", len(windows))
    for window in windows:
        encoded = runner.encode(torch.tensor([window]))
        for store, acts in zip(sums, _captured(runner, encoded, heads)[1:], strict=True):
            for la, act in acts.items():  # a window's first token is an attention sink
                store[la] = store.get(la, 0) + act[0, 1:].float().sum(0).cpu()
        progress.step()
    count = max(sum(len(w) - 1 for w in windows), 1)
    means = Means(*({la: s / count for la, s in store.items()} for store in sums))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"neurons": means.neurons, "heads": means.heads}, path)
    return means


class Graph:
    """One prompt's forward pass with its graph kept, for attribution and paths.

    It keeps each layer's neuron activations and head outputs, and their mean offsets
    from :class:`Means`, with the first token left out.
    """

    def __init__(self, runner: NeuronHookRunner, prompt: str, means: Means, token, position):
        def ensure_grad(_module, _args, output):
            return output if output.requires_grad else output.detach().requires_grad_(True)

        handle = runner.model.get_input_embeddings().register_forward_hook(ensure_grad)
        try:
            with torch.enable_grad():
                logits, self.neurons, self.heads = _captured(
                    runner, runner.encode(prompt), bool(means.heads)
                )
        finally:
            handle.remove()
        self.runner, self.means = runner, means
        self.logit = logits[0, position, token]
        self.grads = self.gradients(self.logit)  # ∂logit/∂unit, kept for the paths

    def tensors(self) -> list[tuple[str, int, torch.Tensor]]:
        """``(kind, layer, tensor)`` for every captured site; kind is ``N`` or ``H``."""
        return [("N", la, t) for la, t in sorted(self.neurons.items())] + [
            ("H", la, t) for la, t in sorted(self.heads.items())
        ]

    def gradients(self, target: torch.Tensor) -> dict[tuple[str, int], torch.Tensor]:
        """``∂target/∂site`` at every site, past the first token, on the CPU."""
        sites = self.tensors()
        grads = torch.autograd.grad(
            target, [t for *_, t in sites], retain_graph=True, allow_unused=True
        )
        return {
            (kind, la): (torch.zeros_like(t) if g is None else g)[0, 1:].float().cpu()
            for (kind, la, t), g in zip(sites, grads, strict=True)
        }

    def offset(self, kind: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The site's values past the first token, and their offset from the mean."""
        store, mean = (
            (self.neurons, self.means.neurons) if kind == "N" else (self.heads, self.means.heads)
        )
        act = store[layer].detach()[0, 1:].float().cpu()
        return act, act - mean[layer]

    def scores(self, grads: dict[tuple[str, int], torch.Tensor]) -> dict:
        """Per unit, ``{(layer, index, pole or HEAD): (attribution, peak)}`` under ``grads``."""
        out = {}
        for kind, layer, _ in self.tensors():
            act, offset = self.offset(kind, layer)
            contribution = offset * grads[(kind, layer)]
            if kind == "H":
                out |= self._head_scores(layer, offset, contribution)
                continue
            for pole, mask in (("positive", act > 0), ("negative", act < 0)):
                score, peak = (contribution * mask).sum(0), (act.abs() * mask).amax(0)
                for n in torch.nonzero(mask.any(0)).flatten().tolist():
                    out[(layer, n, pole)] = (float(score[n]), float(peak[n]))
        return out

    def _head_scores(self, layer: int, offset: torch.Tensor, contribution: torch.Tensor):
        """A head's score sums its slice; its peak is the largest ``|W_O (z − z̄)|``."""
        out = {}
        d = self.runner.head_dim(layer)
        for h in range(self.runner.n_heads):
            cols = slice(h * d, (h + 1) * d)
            w_o = self.runner.head_output_weights(layer, h).detach().float().cpu()
            peak = (offset[:, cols] @ w_o.T).norm(dim=-1).amax()
            out[(layer, h, HEAD)] = (float(contribution[:, cols].sum()), float(peak))
        return out

    def contribution(self, unit: tuple[int, int, str]) -> torch.Tensor:
        """The unit's attribution as a function of its value, with ∂logit/∂unit held fixed."""
        layer, index, pole = unit
        if pole == HEAD:
            d = self.runner.head_dim(layer)
            cols = slice(index * d, (index + 1) * d)
            value = self.heads[layer][0, 1:, cols]
            grad = self.grads[("H", layer)][:, cols]
        else:
            value = self.neurons[layer][0, 1:, index]
            active = (value > 0) if pole == "positive" else (value < 0)
            grad = self.grads[("N", layer)][:, index] * active.cpu()
        dtype = torch.promote_types(value.dtype, torch.float32)  # at least float32, as the grads
        return (value.to(dtype) * grad.to(value.device, dtype)).sum()


def attribute(runner: NeuronHookRunner, prompt: str, means: Means, token: int, position: int):
    """Per unit, ``{(layer, index, pole or HEAD): (attribution, peak strength)}``, and the logit.

    The metric is the logit of ``token`` at ``position``. The prompt's first token is left out.
    """
    graph = Graph(runner, prompt, means, token, position)
    return graph.scores(graph.grads), float(graph.logit.detach())


def paths(graph: Graph, units: Sequence[tuple[int, int, str]], targets: int) -> list[dict]:
    """Edges between ``units``: how much each one drives the contribution of the first ``targets``.

    An edge's weight is the source's attribution to the target's contribution
    (:meth:`Graph.contribution`). Only sources upstream of the target get a weight.
    """
    labels = {u: PoleAttribution(*u, 0.0, 0.0).label for u in units}
    edges = []
    for target in units[:targets]:
        weights = graph.scores(graph.gradients(graph.contribution(target)))
        for source in units:
            weight = weights.get(source, (0.0, 0.0))[0]
            if source != target and weight:
                edges.append({"source": labels[source], "target": labels[target], "weight": weight})
    return sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)


def mean_ablation(units: Sequence[tuple[int, int, str]], means: Means, runner=None):
    """``(interventions, head_interventions)`` that set each unit to its mean.

    A pole is set to its neuron's mean where the pole is active, and a head's output to its
    mean everywhere. The first token keeps its value, as the attribution leaves it out.
    Heads need ``runner``, for their width.
    """
    by_layer: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for layer, index, pole in units:
        by_layer.setdefault(("H" if pole == HEAD else "N", layer), []).append((index, pole))

    def neurons_at(layer: int):
        def intervene(act: torch.Tensor) -> torch.Tensor:
            act = act.clone()
            for neuron, pole in by_layer[("N", layer)]:
                a = act[..., neuron]
                active = a > 0 if pole == "positive" else a < 0
                active[..., 0] = False  # the first token stays, as in the attribution
                act[..., neuron] = torch.where(active, means.neurons[layer][neuron].to(act), a)
            return act

        return intervene

    def heads_at(layer: int):
        def intervene(z: torch.Tensor) -> torch.Tensor:
            z, d = z.clone(), runner.head_dim(layer)
            for head, _ in by_layer[("H", layer)]:
                cols = slice(head * d, (head + 1) * d)
                z[..., 1:, cols] = means.heads[layer][cols].to(z)
            return z

        return intervene

    return (
        {la: neurons_at(la) for kind, la in by_layer if kind == "N"},
        {la: heads_at(la) for kind, la in by_layer if kind == "H"},
    )


@torch.no_grad()
def logits_with(runner: NeuronHookRunner, prompt: str, interventions, position: int):
    """The logits at ``position`` with ``interventions`` applied.

    ``interventions`` is a dict of neuron interventions, or a pair from :func:`mean_ablation`.
    """
    neurons, heads = interventions if isinstance(interventions, tuple) else (interventions, {})
    with runner.hooks(neurons, None, heads):
        return runner.forward(runner.encode(prompt))[0, position].float().cpu()


def registry_labels(wiki_dir: Path, model: str, states: dict | None = None) -> dict:
    """``{unit slug: (status, explanation, max_activation)}`` for ``model``'s poles and heads.

    The status is the newest record's verdict. A pole that a scan stopped on, with no
    record, is ``stopped``. ``states`` are ``load_states(wiki_dir)``, when the caller has
    them already: each call reads the whole archive.
    """
    from openinterpretability.planner import load_states, stopped

    out: dict[str, tuple[str, str | None, float | None]] = {}
    for ident in stopped(wiki_dir):
        if ident.model_name == model:
            out[ident.slug] = ("stopped", None, None)
    for slug, state in (load_states(wiki_dir) if states is None else states).items():
        status, record = state.standing  # the standing, not the newest record (#4900)
        sub = record.submission
        if sub.identifier.model_name == model:
            out[slug] = (status, sub.natural_language_explanation, sub.max_activation)
    return out


def _slug(model: str, unit: tuple[int, int, str]) -> str:
    from openinterpretability.core.identifiers import HeadIdentifier, NeuronIdentifier

    layer, index, pole = unit
    if pole == HEAD:
        return HeadIdentifier(model_name=model, layer=layer, head_idx=index).slug
    return NeuronIdentifier(model_name=model, layer=layer, neuron_idx=index, pole=pole).slug


def _ranked(model: str, scores: dict, labels: dict, top: int, covered=frozenset()):
    """The top units, labelled, and the total and explained sums of |attribution|.

    A unit is explained when it is VERIFIED, or in ``covered``: a VERIFIED group has it.
    The last item is every unit in rank order, as ``(unit, |attribution|, status,
    explained)``, for the experiment plan (:mod:`openinterpretability.explain_plan`).
    """
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
    total = verified = 0.0
    units, order = [], []
    for rank, (unit, (score, strength)) in enumerate(ranked):
        status, text, max_act = labels.get(_slug(model, unit), ("unstudied", None, None))
        explained = status == "VERIFIED" or unit in covered
        total += abs(score)
        verified += abs(score) if explained else 0.0
        order.append((unit, abs(score), status, explained))
        if rank < top:
            shown = text if status == "VERIFIED" else None  # only a VERIFIED one is shown
            units.append(PoleAttribution(*unit, score, strength, status, shown, max_act))
    return units, total, verified, order


def _check(runner, prompt, means: Means, units, base, token, position, confirm) -> dict:
    """Mean-ablate the first ``confirm`` units alone, then the VERIFIED ones together."""
    for u in units[:confirm]:
        ablated = logits_with(runner, prompt, mean_ablation([u.key], means, runner), position)
        u.ablation_change = float(ablated[token] - base[token])
    chosen = [u.key for u in units if u.status == "VERIFIED"]
    joint: dict = {"poles": len(chosen)}
    if chosen:
        ablated = logits_with(runner, prompt, mean_ablation(chosen, means, runner), position)
        joint |= {
            "logit_change": float(ablated[token] - base[token]),
            "top_token_changed": int(ablated.argmax()) != token,
            "new_top_token": runner.tokenizer.decode([int(ablated.argmax())]),
        }
    return joint


def _groups(graph, scores, defs, labels: dict, group_labels: dict):
    """The superpositions and circuits by |attribution|, labelled from the registry."""
    from openinterpretability import explain_groups

    found = []
    for g in defs:
        if g.kind == "superposition":
            status, text, _ = labels.get(g.slug, ("unstudied", None, None))
        else:
            status, text = group_labels.get(g.slug, ("unstudied", None))
        shown = text if status == "VERIFIED" else None
        attribution = explain_groups.score(graph, scores, g)
        members = len(g.neurons) + len(g.heads)
        found.append(GroupAttribution(g.kind, g.name, members, attribution, status, shown))
    return sorted(found, key=lambda a: abs(a.attribution), reverse=True)


def _check_groups(runner, prompt, means, found, defs, base, token, position, confirm) -> None:
    """Remove each of the first ``confirm`` groups, and record the change in the logit."""
    from openinterpretability.explain_groups import remove_direction

    by_name = {(g.kind, g.name): g for g in defs}
    for a in found[:confirm]:
        g = by_name[(a.kind, a.name)]
        if g.kind == "superposition":
            hooks = remove_direction(g, means)
        else:
            hooks = mean_ablation(g.units, means, runner)
        a.ablation_change = float(logits_with(runner, prompt, hooks, position)[token] - base[token])


def explain(
    runner: NeuronHookRunner,
    model: str,
    prompt: str,
    means: Means,
    labels: dict,
    *,
    token: int | None = None,
    position: int = -1,
    top: int = 20,
    confirm: int = 10,
    edges: int = 8,
    groups: Sequence[GroupDef] = (),
    group_labels: dict | None = None,
) -> Explanation:
    """Attribute, label, measure the explained share, check by ablation and trace paths.

    The module docstring gives each step. ``edges`` is the number of top units whose
    incoming edges from the other top units are traced. ``groups`` are the model's
    superpositions and circuits (:mod:`openinterpretability.explain_groups`), and ``group_labels``
    gives each circuit's status and explanation.
    """
    base = logits_with(runner, prompt, {}, position)
    token = int(base.argmax()) if token is None else token
    graph = Graph(runner, prompt, means, token, position)
    scores = graph.scores(graph.grads)
    found_groups = _groups(graph, scores, groups, labels, group_labels or {})
    verified_groups = {(a.kind, a.name) for a in found_groups if a.status == "VERIFIED"}
    covered = {u for g in groups if (g.kind, g.name) in verified_groups for u in g.units}
    units, total, verified, order = _ranked(model, scores, labels, top, frozenset(covered))
    found = paths(graph, [u.key for u in units], edges) if edges else []
    logit = float(graph.logit.detach())
    del graph  # the kept graph is the largest thing here
    joint = _check(runner, prompt, means, units, base, token, position, confirm)
    _check_groups(runner, prompt, means, found_groups, groups, base, token, position, confirm)
    decoded = runner.tokenizer.decode([token])
    net = sum(score for score, _ in scores.values())
    return Explanation(
        prompt, position, decoded, logit, units, total, verified, joint, found, found_groups, order,
        token, net,
    )  # fmt: skip
