"""Can a gate tell wrong-copy prompts from right-copy prompts (issue #5233)?

Arms G and I of the fix plan (#5220) add a module that a gate switches on. The gate must
open on the copy failures of #5232 and stay shut on the right copies and on ordinary text.
This script fits two gates on pythia-70m and tests them on held-out prompts:

- **Text-keyed.** A probe on the residual stream at the answer position, after each layer:
  the difference-in-means direction of #5074, and an L2-regularised logistic regression.
- **Mechanism-keyed.** A logistic regression on four numbers. Three are the attention of
  the copy heads L3/H3, L4/H0 and L4/H7 from the answer position to the first clause's
  unit. The fourth is the cosine of the two subjects' residual states after layer 2.

Both gates are fitted on train's wrong copies against retain's right copies, and tested on
held-out's. The threshold is set on the fitted data, at :data:`FIT_FPR` false opens on the
right copies. Corpus text gives each gate's false opens on ordinary text. The mechanism
gate has no template positions there, so it gets a proxy: the copy heads' strongest
attention to an earlier token.

    uv run python scripts/copy_gate.py --out docs/analyses/copy-gate.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
import torch

MODEL = "EleutherAI/pythia-70m"
FAMILY = Path("docs/analyses/copy-family.jsonl")
COPY_HEADS = ((3, 3), (4, 0), (4, 7))  # the VERIFIED route's copy heads (#4781)
MISMATCH_LAYER = 2  # where #5074's direction separated the groups best
FIT_FPR = 0.05  # the false-open rate on fitted right copies that sets each threshold
RANDOM_DIRECTIONS = 200
CORPUS_TOKENS = 20_000
BOOTSTRAP = 1000
FOLDS = 5  # out-of-fold scores set each threshold
CORPUS_FIT = 2000  # corpus tokens that a gate may fit on as right answers; the rest test it
SEED = 5233


def auroc(pos: list[float], neg: list[float]) -> float:
    """The chance that a random positive scores above a random negative (ties count half).

    Args:
        pos: The scores of the positives (wrong copies).
        neg: The scores of the negatives (right copies).
    """
    total = 0.0
    for p in pos:
        total += sum(1.0 if p > n else 0.5 if p == n else 0.0 for n in neg)
    return total / (len(pos) * len(neg))


def interval(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """A 95% bootstrap interval of the AUROC, resampling each group.

    Args:
        pos: The scores of the positives.
        neg: The scores of the negatives.
    """
    rng = random.Random(SEED)
    values = sorted(
        auroc(rng.choices(pos, k=len(pos)), rng.choices(neg, k=len(neg))) for _ in range(BOOTSTRAP)
    )
    return values[int(0.025 * BOOTSTRAP)], values[int(0.975 * BOOTSTRAP) - 1]


def threshold(neg: list[float], fpr: float = FIT_FPR) -> float:
    """The score above which at most ``fpr`` of the negatives fall.

    Args:
        neg: The negatives' scores on the fitted data.
        fpr: The share of negatives allowed above it.
    """
    ranked = sorted(neg, reverse=True)
    return ranked[min(len(ranked) - 1, int(fpr * len(ranked)))]


def logistic(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-2) -> tuple[torch.Tensor, float]:
    """An L2-regularised logistic regression on standardised ``x``: weights and bias.

    Args:
        x: The features, one row per prompt.
        y: 1 for a wrong copy, 0 for a right one.
        l2: The weight of the L2 penalty.
    """
    torch.manual_seed(SEED)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)
    weight = torch.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # balance the groups

    def step():
        opt.zero_grad()
        logits = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weight)
        loss = loss + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(step)
    return w.detach(), float(b.detach())


class Scaler:
    """Standardise features with the fitted data's mean and spread."""

    def __init__(self, x: torch.Tensor):
        """Keep the mean and the spread of ``x``.

        Args:
            x: The fitted features.
        """
        self.mean, self.std = x.mean(0), x.std(0).clamp_min(1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` standardised.

        Args:
            x: Features to standardise.
        """
        return (x - self.mean) / self.std


def family() -> list[dict]:
    """The family's prompts with their kind and split."""
    rows = [json.loads(line) for line in FAMILY.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r.get("table") == "prompt"]


def unit_position(tokens: list[str], unit: str) -> int:
    """The position of the first clause's unit: the first token that is the unit.

    Args:
        tokens: The prompt's tokens.
        unit: The copied unit, with its leading space.
    """
    return next(i for i, t in enumerate(tokens) if t == unit)


def measure(runner, row: dict) -> dict:
    """The residual states, the copy heads' attention and the subject cosine of one prompt.

    Args:
        runner: The loaded pythia-70m.
        row: A family prompt.
    """
    enc = runner.encode(row["prompt"])
    ids = enc["input_ids"][0].tolist()
    tokens = [runner.tokenizer.decode([i]) for i in ids]
    with torch.no_grad():
        out = runner.model(**enc, output_hidden_states=True, output_attentions=True)
    unit = unit_position(tokens, row["copied"])
    has = [i for i, t in enumerate(tokens) if t == " has"]  # "A <X> has ... a <Y> has <m>"
    x_end, y_end = has[0] - 1, has[-1] - 1  # each subject's last token
    states = out.hidden_states
    mid = states[MISMATCH_LAYER + 1][0]
    cosine = float(torch.nn.functional.cosine_similarity(mid[x_end], mid[y_end], dim=0))
    attention = [float(out.attentions[la][0, h, -1, unit]) for la, h in COPY_HEADS]
    resid = torch.stack([s[0, -1].float() for s in states[1:]])
    return {"resid": resid, "attention": attention, "cosine": cosine}


def corpus_states(runner, texts: list[str], tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each layer's residual state, and the copy heads' strongest earlier attention, per token.

    Args:
        runner: The loaded pythia-70m.
        texts: Corpus documents.
        tokens: How many corpus tokens to read.
    """
    resid, attn, seen = [], [], 0
    for text in texts:
        enc = runner.encode(text[:2000])
        if enc["input_ids"].shape[1] < 3:
            continue
        with torch.no_grad():
            out = runner.model(**enc, output_hidden_states=True, output_attentions=True)
        n = enc["input_ids"].shape[1]
        resid.append(torch.stack([s[0, 1:].float() for s in out.hidden_states[1:]], dim=1))
        heads = [out.attentions[la][0, h, 1:, 1:] for la, h in COPY_HEADS]  # not the first token
        attn.append(torch.stack([a.max(dim=-1).values for a in heads], dim=1))
        seen += n - 1
        if seen >= tokens:
            break
    return torch.cat(resid)[:tokens], torch.cat(attn)[:tokens]


def _probe(kind: str, x: torch.Tensor, y: torch.Tensor):
    """A scoring function fitted on ``x`` and ``y``: wrong copies score high.

    Args:
        kind: ``diff_means`` or ``logistic``.
        x: The fitted rows.
        y: 1 for a wrong copy, 0 for a right copy or a corpus token.
    """
    if kind == "diff_means":
        d = x[y == 1].mean(0) - x[y == 0].mean(0)
        return lambda z: z @ d
    scale = Scaler(x)
    w, b = logistic(scale(x), y)
    return lambda z: scale(z) @ w + b


def _out_of_fold(kind: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each fitted row's score from a probe fitted without its fold (:data:`FOLDS` folds).

    Args:
        kind: The probe's kind.
        x: The fitted rows.
        y: Their labels.
    """
    fold = torch.arange(len(y)) % FOLDS
    fold = fold[torch.randperm(len(y), generator=torch.Generator().manual_seed(SEED))]
    out = torch.empty(len(y))
    for f in range(FOLDS):
        held = fold == f
        out[held] = _probe(kind, x[~held], y[~held])(x[held])
    return out


def _share(opened: torch.Tensor) -> float:
    """The share of rows where the gate opened, to 4 decimals.

    Args:
        opened: True where the gate opened.
    """
    return round(float(opened.float().mean()), 4)


def _recall_at(pos: list[float], neg: list[float], fpr: float = FIT_FPR) -> float:
    """The held-out ROC's recall where at most ``fpr`` of the negatives open.

    A reading of the curve, not a threshold that a gate could set without held-out data.

    Args:
        pos: The wrong copies' scores.
        neg: The right copies' scores.
        fpr: The share of negatives allowed open.
    """
    cut = threshold(neg, fpr)
    return sum(p > cut for p in pos) / len(pos)


def text_rows(fit, test, corpus_resid, n_layers: int) -> list[dict]:
    """The text-keyed gate after each layer: three probes, the null, and the operating point.

    ``logistic_corpus`` also fits on the first :data:`CORPUS_FIT` corpus tokens as right
    answers. Its threshold keeps both right copies and those tokens at :data:`FIT_FPR`.
    Every threshold comes from out-of-fold scores, so it isn't fitted to the rows it reads.
    The corpus false-open rate is measured on the other corpus tokens.

    Args:
        fit: The fitted prompts' measures, with ``wrong`` set.
        test: The held-out prompts' measures, with ``wrong`` set.
        corpus_resid: Each corpus token's residual states, layer by layer.
        n_layers: The model's layer count.
    """
    rng = torch.Generator().manual_seed(SEED)
    rows = []
    y_fit = torch.tensor([float(m["wrong"]) for m in fit])
    for layer in range(n_layers):
        xf = torch.stack([m["resid"][layer] for m in fit])
        xt = torch.stack([m["resid"][layer] for m in test])
        c_fit, c_test = corpus_resid[:CORPUS_FIT, layer], corpus_resid[CORPUS_FIT:, layer]
        nulls = []
        for _ in range(RANDOM_DIRECTIONS):  # a random direction, signed on the fitted data
            r = torch.randn(xf.shape[1], generator=rng)
            sign = 1.0 if auroc(*_split(xf @ r, fit)) >= 0.5 else -1.0
            nulls.append(auroc(*_split(sign * (xt @ r), test)))
        null95 = sorted(nulls)[int(0.95 * RANDOM_DIRECTIONS) - 1]
        for name in ("diff_means", "logistic", "logistic_corpus"):
            kind = "diff_means" if name == "diff_means" else "logistic"
            if name == "logistic_corpus":
                x = torch.cat([xf, c_fit])
                y = torch.cat([y_fit, torch.zeros(len(c_fit))])
            else:
                x, y = xf, y_fit
            oof = _out_of_fold(kind, x, y)
            cut = threshold(oof[y == 0][: int((y_fit == 0).sum())].tolist())
            if name == "logistic_corpus":
                cut = max(cut, threshold(oof[len(y_fit) :].tolist()))
            probe = _probe(kind, x, y)
            pos_t, neg_t = _split(probe(xt), test)
            low, high = interval(pos_t, neg_t)
            rows.append({"gate": "text", "probe": name, "layer": layer,
                         "auroc": round(auroc(pos_t, neg_t), 4), "low": round(low, 4),
                         "high": round(high, 4), "null_95": round(null95, 4),
                         "recall_at_5": round(_recall_at(pos_t, neg_t), 4),
                         "recall": round(sum(p > cut for p in pos_t) / len(pos_t), 4),
                         "right_fpr": round(sum(n > cut for n in neg_t) / len(neg_t), 4),
                         "corpus_fpr": _share(probe(c_test) > cut)})  # fmt: skip
    return rows


def _split(scores: torch.Tensor, measures: list[dict]) -> tuple[list[float], list[float]]:
    """The scores of the wrong copies and of the right copies.

    Args:
        scores: One score per prompt.
        measures: The prompts' measures, with ``wrong`` set.
    """
    values = scores.tolist()
    pos = [v for v, m in zip(values, measures, strict=True) if m["wrong"]]
    neg = [v for v, m in zip(values, measures, strict=True) if not m["wrong"]]
    return pos, neg


def mechanism_rows(fit, test, corpus_attn) -> list[dict]:
    """The mechanism-keyed gate, and its attention-only part, which has a corpus proxy.

    Args:
        fit: The fitted prompts' measures.
        test: The held-out prompts' measures.
        corpus_attn: Each corpus token's strongest earlier attention, per copy head.
    """
    y_fit = torch.tensor([float(m["wrong"]) for m in fit])
    rows = []
    for name, cols in (("attention_and_mismatch", 4), ("attention_only", 3)):

        def feats(ms, cols=cols):
            return torch.tensor([[*m["attention"], m["cosine"]][:cols] for m in ms])

        xf, xt = feats(fit), feats(test)
        oof = _out_of_fold("logistic", xf, y_fit)
        cut = threshold(oof[y_fit == 0].tolist())
        scale = Scaler(xf)
        w, b = logistic(scale(xf), y_fit)
        pos_t, neg_t = _split(scale(xt) @ w + b, test)
        low, high = interval(pos_t, neg_t)
        corpus = None
        if cols == 3:  # no template positions in corpus text: the attention-only proxy
            proxy = corpus_attn[CORPUS_FIT:]
            corpus = round(float(((scale(proxy) @ w + b) > cut).float().mean()), 4)
        rows.append({"gate": "mechanism", "probe": name, "auroc": round(auroc(pos_t, neg_t), 4),
                     "low": round(low, 4), "high": round(high, 4),
                     "recall_at_5": round(_recall_at(pos_t, neg_t), 4),
                     "recall": round(sum(p > cut for p in pos_t) / len(pos_t), 4),
                     "right_fpr": round(sum(n > cut for n in neg_t) / len(neg_t), 4),
                     "corpus_fpr": corpus, "weights": [round(float(v), 4) for v in w]})  # fmt: skip
    return rows


def _counts(measures: list[dict]) -> dict:
    """How many wrong and right copies ``measures`` holds.

    Args:
        measures: The prompts' measures, with ``wrong`` set.
    """
    return {"wrong": sum(m["wrong"] for m in measures),
            "right": sum(not m["wrong"] for m in measures)}  # fmt: skip


@click.command()
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("docs/analyses/copy-gate.jsonl")
)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
def main(out: Path, corpus: Path) -> None:
    """Measure the family, fit both gates, and write their held-out and corpus results.

    Args:
        out: Where to write the rows.
        corpus: The corpus whose text gives the false opens on ordinary text.
    """
    from openinterpretability.cli import _load_runner
    from openinterpretability.evidence import read_corpus

    runner = _load_runner(MODEL, "cpu", "float32", attn_implementation="eager")
    rows = family()
    fit = [r for r in rows if (r["kind"], r["split"]) in {("wrong_copy", "train"),
                                                           ("right_copy", "retain")}]  # fmt: skip
    test = [r for r in rows if r["split"] == "heldout"]
    measured = {}
    for r in fit + test:
        measured[r["prompt"]] = {**measure(runner, r), "wrong": r["kind"] == "wrong_copy"}
    click.echo(f"measured {len(measured)} prompts")
    fit_m, test_m = [measured[r["prompt"]] for r in fit], [measured[r["prompt"]] for r in test]
    resid, attn = corpus_states(runner, read_corpus(corpus)[0], CORPUS_TOKENS)
    click.echo(f"read {resid.shape[0]} corpus tokens")
    meta = {"table": "metadata", "issue": 5233, "model": MODEL, "family": str(FAMILY),
            "fit": _counts(fit_m), "test": _counts(test_m),
            "corpus_tokens": int(resid.shape[0]), "fit_fpr": FIT_FPR, "seed": SEED,
            "copy_heads": [f"L{la}/H{h}" for la, h in COPY_HEADS], "mismatch_layer": MISMATCH_LAYER,
            "random_directions": RANDOM_DIRECTIONS, "bootstrap": BOOTSTRAP}  # fmt: skip
    out_rows = [meta, *text_rows(fit_m, test_m, resid, runner.n_layers),
                *mechanism_rows(fit_m, test_m, attn)]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
    click.echo(f"wrote {len(out_rows)} rows → {out}")


if __name__ == "__main__":
    main()
