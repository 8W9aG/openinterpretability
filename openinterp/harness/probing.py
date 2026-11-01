"""Linear probes for a pole's concept, compared with the pole itself.

Labelled text is JSON lines ``{"text": ..., "spans": [[start, end], ...]}``: character
spans where the concept holds, marked by a rule that never sees the model. A token is
positive when its characters overlap a span. Texts are cut into windows as a scan cuts
them, and each window's first token is dropped (it is the attention sink, and some
early-layer poles have a floor there).

At each kept token the probe reads the layer's MLP hidden activation (signed, every
neuron) and the residual stream after the layer's block. It fits:

* a *logistic* probe on standardised features with an L2 penalty, classes weighted
  equally;
* a *mass-mean* probe, the difference of the class means in raw activation space
  (Marks & Tegmark, 2023);
* *k-sparse* logistic probes on the k neurons with the largest standardised class-mean
  difference (Gurnee et al., 2023).

Every score's threshold is fitted on the training tokens and tested on held-out ones.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import torch

from openinterp.core.identifiers import Pole
from openinterp.core.probing import NeuronScore, ProbeScore, SiteProbes, SparsePoint
from openinterp.core.schemas import ACTIVATION_THRESHOLD
from openinterp.core.stats import rank_columns
from openinterp.harness.hooks import NeuronHookRunner, _StopForward, pole_strength

SPARSE_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
NEEDED_MARGIN = 0.02  # a sparse probe "matches" the dense one within this balanced accuracy
DEFAULT_L2 = 1e-3


@dataclass(frozen=True)
class Window:
    ids: list[int]
    labels: list[bool]
    document: int


@dataclass(frozen=True)
class TokenSet:
    hidden: torch.Tensor  # [n, d_mlp]
    residual: torch.Tensor  # [n, d_model]
    labels: torch.Tensor  # [n] bool

    @property
    def counts(self) -> tuple[int, int]:
        pos = int(self.labels.sum())
        return pos, len(self.labels) - pos


def read_labels(path: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append((row["text"], [(int(s), int(e)) for s, e in row.get("spans", [])]))
    return rows


def label_tokens(tokenizer, text: str, spans: Sequence[tuple[int, int]]):
    """Token ids of ``text`` and whether each token overlaps a span (needs a fast tokenizer)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = [any(s < end and start < e for s, e in spans) for start, end in enc["offset_mapping"]]
    return list(enc["input_ids"]), labels


def windows(tokenizer, rows, window_tokens: int) -> list[Window]:
    out = []
    for doc, (text, spans) in enumerate(rows):
        ids, labels = label_tokens(tokenizer, text, spans)
        for start in range(0, len(ids), window_tokens):
            if len(ids) - start > 1:  # a lone token is only a first token
                cut = slice(start, start + window_tokens)
                out.append(Window(ids[cut], labels[cut], doc))
    return out


def choose_tokens(
    wins: Sequence[Window], max_positive: int, max_negative: int, seed: int
) -> list[list[int]]:
    """Positions to keep in each window: every position but the first, sampled down to
    at most ``max_positive`` positive and ``max_negative`` negative tokens."""
    pos, neg = [], []
    for w, win in enumerate(wins):
        for i in range(1, len(win.ids)):
            (pos if win.labels[i] else neg).append((w, i))
    rng = random.Random(seed)
    keep = [[] for _ in wins]
    for group, cap in ((pos, max_positive), (neg, max_negative)):
        for w, i in sorted(rng.sample(group, min(cap, len(group)))):
            keep[w].append(i)
    return keep


@torch.no_grad()
def collect(
    runner: NeuronHookRunner,
    wins: Sequence[Window],
    keep: Sequence[Sequence[int]],
    layer: int,
    batch_size: int = 16,
) -> TokenSet:
    """The hidden activations and residual stream of ``layer`` at the kept positions."""
    hidden, residual, labels = [], [], []
    todo = [(w, k) for w, k in zip(wins, keep, strict=True) if k]
    pad = runner.tokenizer.pad_token_id or 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        width = max(len(w.ids) for w, _ in batch)
        ids = torch.tensor([w.ids + [pad] * (width - len(w.ids)) for w, _ in batch])
        mask = torch.tensor([[1] * len(w.ids) + [0] * (width - len(w.ids)) for w, _ in batch])
        seen = _run_to(runner, layer, ids, mask)
        for row, (win, positions) in enumerate(batch):
            hidden.append(seen["hidden"][row, positions])
            residual.append(seen["residual"][row, positions])
            labels += [win.labels[i] for i in positions]
    return TokenSet(torch.cat(hidden), torch.cat(residual), torch.tensor(labels, dtype=torch.bool))


def _run_to(runner: NeuronHookRunner, layer: int, ids, mask) -> dict[str, torch.Tensor]:
    """Run until ``layer``'s block is done, recording its MLP hidden layer and output."""
    seen: dict[str, torch.Tensor] = {}

    def record_hidden(act):
        seen["hidden"] = act.detach().float().cpu()

    def record_residual(_module, _args, output):
        out = output[0] if isinstance(output, tuple) else output
        seen["residual"] = out.detach().float().cpu()
        raise _StopForward

    handle = runner.layers[layer].register_forward_hook(record_residual)
    try:
        with runner.hooks({layer: record_hidden}), suppress(_StopForward):
            runner.forward({"input_ids": ids.to(runner.device), "attention_mask": mask})
    finally:
        handle.remove()
    return seen


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann–Whitney AUROC with ties counted half (pole strengths are mostly zeros)."""
    ranks = rank_columns(scores[:, None].float())[:, 0]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    tpr = (predicted & labels).sum() / labels.sum()
    tnr = (~predicted & ~labels).sum() / (~labels).sum()
    return float((tpr + tnr) / 2)


def fit_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """The cut ``score >= t`` with the best balanced accuracy on these tokens."""
    values = scores.unique()
    pos, neg = scores[labels].sort().values, scores[~labels].sort().values
    tpr = (len(pos) - torch.searchsorted(pos, values)) / len(pos)
    fpr = (len(neg) - torch.searchsorted(neg, values)) / len(neg)
    return float(values[(tpr + 1 - fpr).argmax()])


def held_out_score(
    train: torch.Tensor, train_y: torch.Tensor, test: torch.Tensor, test_y: torch.Tensor
) -> ProbeScore:
    t = fit_threshold(train, train_y)
    return ProbeScore(
        balanced_accuracy=balanced_accuracy(test >= t, test_y), auroc=auroc(test, test_y)
    )


def standardiser(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(0), x.std(0).clamp(min=1e-6)


def fit_logistic(
    x: torch.Tensor, y: torch.Tensor, l2: float = DEFAULT_L2, steps: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced logistic regression on standardised ``x``; returns the weights in
    raw activation space and the bias, so ``x @ w + b`` is the logit."""
    mean, std = standardiser(x)
    z = (x - mean) / std
    weight = torch.where(y, 0.5 / y.sum(), 0.5 / (~y).sum())
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    target = y.float()

    def closure():
        opt.zero_grad()
        logits = z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, weight=weight, reduction="sum"
        )
        loss = loss + l2 * (w * w).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    w_raw = w.detach() / std
    return w_raw, b.detach() - (mean * w_raw).sum()


def mass_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[y].mean(0) - x[~y].mean(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).clamp(-1, 1))


def site_probes(
    site: str,
    train: torch.Tensor,
    train_y: torch.Tensor,
    test: torch.Tensor,
    test_y: torch.Tensor,
    pole_direction: torch.Tensor,
    l2: float,
) -> SiteProbes:
    w, b = fit_logistic(train, train_y, l2)
    d = mass_mean(train, train_y)
    return SiteProbes(
        site=site,
        width=train.shape[1],
        logistic=held_out_score(train @ w + b, train_y, test @ w + b, test_y),
        mass_mean=held_out_score(train @ d, train_y, test @ d, test_y),
        logistic_cosine=cosine(w, pole_direction),
        mass_mean_cosine=cosine(d, pole_direction),
    )


def separation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each column's |class-mean difference| in units of its standard deviation."""
    _, std = standardiser(x)
    return mass_mean(x, y).abs() / std


def sparse_curve(
    train: torch.Tensor,
    train_y: torch.Tensor,
    test: torch.Tensor,
    test_y: torch.Tensor,
    neuron: int,
    l2: float,
    ks: Iterable[int] = SPARSE_KS,
) -> tuple[list[SparsePoint], int]:
    """k-sparse probes, and the audited neuron's rank by separation (1 = best)."""
    order = separation(train, train_y).argsort(descending=True)
    rank = int((order == neuron).nonzero()[0, 0]) + 1
    points = []
    for k in ks:
        if k > train.shape[1]:
            break
        cols = order[:k]
        w, b = fit_logistic(train[:, cols], train_y, l2)
        points.append(
            SparsePoint(
                k=k,
                score=held_out_score(
                    train[:, cols] @ w + b, train_y, test[:, cols] @ w + b, test_y
                ),
                includes_neuron=rank <= k,
            )
        )
    return points, rank


def neuron_score(
    train: TokenSet, test: TokenSet, neuron: int, pole: Pole, line: float, rank: int
) -> NeuronScore:
    strength = [pole_strength(s.hidden[:, neuron], pole) for s in (train, test)]
    score = held_out_score(strength[0], train.labels, strength[1], test.labels)
    return NeuronScore(
        **score.model_dump(),
        line=line,
        balanced_accuracy_at_line=balanced_accuracy(strength[1] >= line, test.labels),
        rank=rank,
    )


def neurons_needed(points: Sequence[SparsePoint], dense: SiteProbes) -> int | None:
    """Smallest k within the margin of the better dense probe on the hidden layer (a
    logistic probe's threshold can transfer worse across sources than a mass-mean's)."""
    best = max(dense.logistic.balanced_accuracy, dense.mass_mean.balanced_accuracy)
    for p in points:
        if p.score.balanced_accuracy >= best - NEEDED_MARGIN:
            return p.k
    return None


def split_documents(wins: Sequence[Window], test_share: float, seed: int):
    """Hold out a share of the documents (never a window of a training document)."""
    docs = sorted({w.document for w in wins})
    random.Random(seed).shuffle(docs)
    held = set(docs[: max(1, round(len(docs) * test_share))])
    return [w for w in wins if w.document not in held], [w for w in wins if w.document in held]


def compare(
    runner: NeuronHookRunner,
    submission,
    train_wins: Sequence[Window],
    test_wins: Sequence[Window],
    *,
    max_positive: int,
    max_negative: int,
    seed: int,
    l2: float = DEFAULT_L2,
) -> dict:
    """Every score of :class:`ProbeComparison` except what names the labels."""
    ident = submission.identifier
    layer, neuron, pole = ident.layer, ident.neuron_idx, ident.pole
    sets = []
    for i, wins in enumerate((train_wins, test_wins)):
        keep = choose_tokens(wins, max_positive, max_negative, seed + i)
        sets.append(collect(runner, wins, keep, layer))
    train, test = sets
    for name, s in (("training", train), ("held-out", test)):
        if 0 in s.counts:
            raise ValueError(f"the {name} tokens hold only one class: {s.counts}")
    sign = 1.0 if pole == "positive" else -1.0
    basis = torch.zeros(train.hidden.shape[1])
    basis[neuron] = sign
    write = sign * runner.neuron_output_direction(layer, neuron).float().cpu()
    y, yt = train.labels, test.labels
    sites = [
        site_probes("mlp_hidden", train.hidden, y, test.hidden, yt, basis, l2),
        site_probes("residual", train.residual, y, test.residual, yt, write, l2),
    ]
    sparse, rank = sparse_curve(train.hidden, y, test.hidden, yt, neuron, l2)
    line = ACTIVATION_THRESHOLD * submission.max_activation
    (train_pos, train_neg), (test_pos, test_neg) = train.counts, test.counts
    return {
        "slug": ident.slug,
        "explanation": submission.natural_language_explanation,
        "seed": seed,
        "l2": l2,
        "train_positive": train_pos,
        "train_negative": train_neg,
        "test_positive": test_pos,
        "test_negative": test_neg,
        "neuron": neuron_score(train, test, neuron, pole, line, rank),
        "sites": sites,
        "sparse": sparse,
        "neurons_needed": neurons_needed(sparse, sites[0]),
    }
