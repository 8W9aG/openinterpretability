"""Held-out windows and the arithmetic of detection, fuzzing and embedding scores.

The judge only ever sees a window's text, so each window is measured on its own, the
way probes are, rather than inside the snippet it was cut from. Labels are always
re-derived from a fresh measurement, so a task handed to a judge carries no
activations and a stored score can be re-checked (`openinterp reproduce`).
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from openinterp.core.explanation import ExplanationScores, JudgeScore, ScoredWindow
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.units import Unit

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Window:
    text: str
    tokens: list[str]
    activations: list[float]
    highlighted: tuple[int, ...] = ()
    start: int = 0  # 1 for a unit with a first-token floor (protocol 0.8)

    @property
    def peak(self) -> float:
        """Over the measured tokens."""
        return max(self.activations[self.start :], default=0.0)

    def hot(self, threshold: float) -> list[int]:
        """Measured tokens at or above ``threshold``."""
        return [i for i, a in enumerate(self.activations) if i >= self.start and a >= threshold]

    def cold(self, threshold: float) -> list[int]:
        return [i for i, a in enumerate(self.activations) if i >= self.start and a < threshold]


@dataclass(frozen=True)
class HeldOutSample:
    detection: list[Window]
    fuzzing: list[Window]
    quantile_edges: list[float]
    candidate_windows: int
    active_windows: int


def cut_windows(runner: NeuronHookRunner, texts: Sequence[str], window_tokens: int) -> list[str]:
    """Split each text into consecutive ``window_tokens``-token pieces, decoded back to text.

    A trailing piece shorter than half a window is dropped, as are blank pieces.
    """
    tok = runner.tokenizer
    pieces = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        for start in range(0, len(ids), window_tokens):
            chunk = ids[start : start + window_tokens]
            if len(chunk) * 2 < window_tokens and start > 0:
                continue
            piece = tok.decode(chunk)
            if piece.strip():
                pieces.append(piece)
    return pieces


@torch.no_grad()
def measure_windows(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    unit: Unit,
    *,
    batch_size: int = 16,
) -> list[Window]:
    """Per-token strengths of each text, run on its own (no surrounding context)."""
    windows = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = runner.encode(batch)
        strengths = unit.strengths(runner, encoded).float()
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask", torch.ones_like(ids)).bool()
        for row, text in enumerate(batch):
            keep = mask[row]
            windows.append(
                Window(
                    text=text,
                    tokens=[runner.tokenizer.decode([i]) for i in ids[row][keep].tolist()],
                    activations=[round(v, 5) for v in strengths[row][keep].cpu().tolist()],
                    start=int(unit.skip_first),
                )
            )
    return windows


def sample_heldout(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    unit: Unit,
    threshold: float,
    *,
    seed: int,
    exclude: Sequence[str] = (),
    window_tokens: int = 32,
    max_windows: int = 4000,
    quantiles: int = 5,
    per_quantile: int = 5,
) -> HeldOutSample:
    """Draw detection and fuzzing windows from the ``texts`` containing no ``exclude``.

    Active windows (peak ≥ ``threshold``) are sorted by peak and split into ``quantiles``
    equal bins; each bin gives ``per_quantile`` windows to detection and as many to
    fuzzing, so the weakest activations count as much as the top ones. Detection gets as
    many inactive windows as active ones. Fuzzing marks the active tokens of half its
    active windows and as many random inactive tokens in the other half, and adds half
    as many inactive windows with random tokens marked. With no active window at all,
    detection gets ``quantiles × per_quantile`` inactive ones and fuzzing none.
    """
    rng = random.Random(seed)
    # A top example may be a window of a longer document: leave out the whole document.
    held_out = [t for t in texts if not any(e in t for e in exclude)]
    candidates = cut_windows(runner, held_out, window_tokens)
    rng.shuffle(candidates)
    measured = measure_windows(runner, candidates[:max_windows], unit)
    active = sorted((w for w in measured if w.peak >= threshold), key=lambda w: w.peak)
    inactive = [w for w in measured if w.peak < threshold]
    if not measured:
        raise ValueError("no held-out windows: every corpus snippet is a top example")
    if not active:  # the explanation can still be scored on false positives
        drawn = rng.sample(inactive, min(len(inactive), quantiles * per_quantile))
        return HeldOutSample(drawn, [], [], len(measured), 0)
    n = len(active)
    bins = [active[i * n // quantiles : (i + 1) * n // quantiles] for i in range(quantiles)]
    edges = [b[0].peak for b in bins[1:] if b]

    detect_active, fuzz_active = [], []
    for members in bins:
        drawn = rng.sample(members, min(len(members), 2 * per_quantile))
        half = (len(drawn) + 1) // 2
        detect_active += drawn[:half]
        fuzz_active += drawn[half:]
    detection = detect_active + rng.sample(inactive, min(len(inactive), len(detect_active)))

    fuzzing = []
    for n, w in enumerate(fuzz_active):
        hot, cold = w.hot(threshold), w.cold(threshold)
        marks = hot if n % 2 == 0 or len(cold) < len(hot) else rng.sample(cold, len(hot))
        fuzzing.append(replace(w, highlighted=tuple(sorted(marks))))
    rest = [w for w in inactive if w not in detection]
    for w in rng.sample(rest, min(len(rest), len(fuzz_active) // 2)):
        tokens = range(w.start, len(w.tokens))
        marks = rng.sample(tokens, min(len(tokens), rng.randint(1, 3)))
        fuzzing.append(replace(w, highlighted=tuple(sorted(marks))))
    rng.shuffle(detection)
    rng.shuffle(fuzzing)
    return HeldOutSample(detection, fuzzing, edges, len(measured), len(active))


def label_windows(
    windows: Sequence[Window],
    threshold: float,
    edges: Sequence[float],
    *,
    fuzzing: bool,
    judged: Sequence[bool | None] | None = None,
    similarities: Sequence[float | None] | None = None,
) -> list[ScoredWindow]:
    out = []
    for n, w in enumerate(windows):
        hot = set(w.hot(threshold))
        label = bool(hot) and set(w.highlighted) == hot if fuzzing else bool(hot)
        out.append(
            ScoredWindow(
                text=w.text,
                tokens=w.tokens,
                activations=w.activations,
                quantile=bisect.bisect_right(edges, w.peak) if hot else None,
                highlighted=list(w.highlighted),
                label=label,
                judged=judged[n] if judged is not None else None,
                similarity=similarities[n] if similarities is not None else None,
            )
        )
    return out


def judge_score(windows: Sequence[ScoredWindow], quantiles: int) -> JudgeScore | None:
    """Balanced accuracy of the judge's answers, or ``None`` if nothing was judged."""
    judged = [w for w in windows if w.judged is not None]
    pos = [w.judged for w in judged if w.label]
    neg = [w.judged for w in judged if not w.label]
    if not judged:
        return None
    tpr = sum(pos) / len(pos) if pos else None
    fpr = sum(neg) / len(neg) if neg else None
    rates = [r for r in (tpr, None if fpr is None else 1 - fpr) if r is not None]
    by_q: list[float | None] = []
    for q in range(quantiles):
        hits = [w.judged for w in judged if w.label and w.quantile == q]
        by_q.append(sum(hits) / len(hits) if hits else None)
    return JudgeScore(
        balanced_accuracy=sum(rates) / len(rates),
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        recall_by_quantile=by_q,
    )


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Area under the ROC curve (Mann–Whitney U, ties count half); ``None`` with one class."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def finalize_scores(scores: ExplanationScores) -> ExplanationScores:
    """Recompute the summary scores from the stored windows, labels and answers."""
    det = scores.detection_windows
    sims = [w.similarity for w in det]
    emb = auroc(sims, [w.label for w in det]) if det and None not in sims else None
    return scores.model_copy(
        update={
            "detection": judge_score(det, scores.quantiles),
            "fuzzing": judge_score(scores.fuzzing_windows, scores.quantiles),
            "embedding_auroc": emb,
        }
    )


def remeasure(scores: ExplanationScores, runner: NeuronHookRunner, unit: Unit):
    """Re-measure every window and re-derive labels and scores, keeping the answers."""

    def redo(windows: Sequence[ScoredWindow], fuzzing: bool) -> list[ScoredWindow]:
        fresh = measure_windows(runner, [w.text for w in windows], unit)
        pairs = zip(fresh, windows, strict=True)
        fresh = [replace(f, highlighted=tuple(w.highlighted)) for f, w in pairs]
        return label_windows(
            fresh,
            scores.threshold,
            scores.quantile_edges,
            fuzzing=fuzzing,
            judged=[w.judged for w in windows],
            similarities=[w.similarity for w in windows],
        )

    return finalize_scores(
        scores.model_copy(
            update={
                "detection_windows": redo(scores.detection_windows, False),
                "fuzzing_windows": redo(scores.fuzzing_windows, True),
            }
        )
    )


@torch.no_grad()
def embedding_similarities(model_name: str, query: str, texts: Sequence[str]) -> list[float]:
    """Cosine between ``query`` and each text, mean-pooled from a sentence encoder."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    vectors = []
    batch = [query, *texts]
    for start in range(0, len(batch), 32):
        enc = tokenizer(
            batch[start : start + 32], padding=True, truncation=True, return_tensors="pt"
        )
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vectors.append((hidden * mask).sum(1) / mask.sum(1).clamp(min=1))
    emb = torch.nn.functional.normalize(torch.cat(vectors), dim=-1)
    return [round(float(v), 6) for v in emb[1:] @ emb[0]]
