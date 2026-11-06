"""What one side of a layer's activations says, beyond token identity.

The numbers behind the "Other decoding targets" section of
docs/analyses/default-on-poles.md. ``scripts/default_on_decode.py`` predicts token identity.
This script keeps its ridge readout and its random-feature null, and predicts two others:

* ``next-token``: the model's own log-probabilities of the 200 commonest tokens, and the
  entropy of its distribution, on the corpus sample's windows.
* ``concepts``: human labels of `openinterp/concepts.py`, for each token of each document.

Token identity explains much of each target, so each row also has a lookup baseline that
reads only the current token. ``concepts`` also scores the ambiguous tokens: the token
types that have the label and do not have it, in the fit documents.

    uv run python scripts/default_on_targets.py next-token --out default-on-targets.jsonl
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent))

from default_on_decode import explained, random_negative_side, ridge
from default_on_poles import (
    GPT2,
    WINDOW_TOKENS,
    batched,
    load_split,
    read_mlp_inputs,
    read_pre_activations,
    write_row,
)

from openinterp.cli import _load, resolve_device
from openinterp.concepts import CONCEPT_PRESETS
from openinterp.concepts import load_split as load_concept_split
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

NEXT_TOKEN_CLASSES = 200
NULL_DRAWS = 3
NULL_SEED_STEP = 1000  # the first draw's seed is the layer, as in default_on_decode.py
BOOTSTRAPS = 100
BOOTSTRAP_SEED = 0
CONCEPT_LAYERS = (0, 3, 6, 9, 11)
CONCEPT_DATASETS = ("ud-upos", "ud-deprel", "conll2003-ner", "ag_news", "language-identification")
CONCEPT_MAX_CHARS = 600  # about one window, so that truncation loses few labels
MAX_FIT_TOKENS = 40_000
MAX_TEST_TOKENS = 30_000
FIT_SHARE = 0.6
MAX_LABELS = 20
MIN_POSITIVES = 100  # tokens with the label, in the fit rows and in the test rows
MIN_AMBIGUOUS = 50  # test tokens on each side of an ambiguous-token score
SPLIT_SEED = 0


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the area under the ROC curve, with tied scores at their mean rank."""
    positive = labels.bool()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    _, inverse, counts = scores.unique(return_inverse=True, return_counts=True)
    ranks = (counts.cumsum(0).double() - (counts.double() - 1) / 2)[inverse]
    return ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def token_lookup(tokens: torch.Tensor, y: torch.Tensor, tokens_test: torch.Tensor) -> torch.Tensor:
    """Return each test token's mean target in the fit rows, shrunk by one row to the mean.

    A token that the fit rows don't have gets the mean of all fit rows.
    """
    vocab = int(max(tokens.max(), tokens_test.max())) + 1
    totals = torch.zeros(vocab, y.shape[1]).index_add_(0, tokens, y)
    counts = tokens.bincount(minlength=vocab).unsqueeze(1)
    return ((totals + y.mean(0)) / (counts + 1))[tokens_test]


def read_sides(runner, fit, test, layer: int, keep_fit=None, keep_test=None) -> dict:
    """Return each side's activations on fit and test rows, with the rows that ``keep`` marks."""
    gelu = torch.nn.functional.gelu
    h = read_pre_activations(runner, fit, [layer])[layer]
    h_test = read_pre_activations(runner, test, [layer])[layer]
    x = read_mlp_inputs(runner, fit, layer).float()
    x_test = read_mlp_inputs(runner, test, layer).float()
    if keep_fit is not None:
        h, x, h_test, x_test = h[keep_fit], x[keep_fit], h_test[keep_test], x_test[keep_test]
    a, b = gelu(h), gelu(h_test)
    sides = {
        "negative": (a.clamp(max=0), b.clamp(max=0)),
        "positive": (a.clamp(min=0), b.clamp(min=0)),
        "both": (a, b),
    }
    for draw in range(NULL_DRAWS):
        seed = layer + draw * NULL_SEED_STEP
        sides[f"null_{draw}"] = random_negative_side(h, x, h_test, x_test, seed)
    return sides


def resample_groups(groups: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return row indices for one bootstrap draw of whole groups, with replacement."""
    names, inverse = groups.unique(return_inverse=True)
    order = inverse.argsort()
    starts = torch.cat([torch.tensor([0]), inverse.bincount().cumsum(0)])
    picks = torch.randint(len(names), (len(names),), generator=generator).tolist()
    return torch.cat([order[starts[g] : starts[g + 1]] for g in picks])


def bootstrap_interval(metric, groups: torch.Tensor) -> list[float]:
    """Return the 95% interval of ``metric(rows)`` over bootstrap draws of ``groups``."""
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    values = [metric(resample_groups(groups, generator)) for _ in range(BOOTSTRAPS)]
    low, high = torch.tensor(values).quantile(torch.tensor([0.025, 0.975])).tolist()
    return [low, high]


@dataclass
class NextTokenTargets:
    """The model's next-token distribution on fit and test rows, and the rows' tokens."""

    y_fit: torch.Tensor  # [rows, classes + 1]: log-probabilities, then entropy
    y_test: torch.Tensor
    tokens_fit: torch.Tensor
    tokens_test: torch.Tensor
    groups_test: torch.Tensor


@torch.no_grad()
def distribution_rows(runner, ids: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """Return log-probabilities of ``classes`` and the entropy, first tokens left out."""
    rows = []
    for batch in batched(ids):
        logits = runner.forward({"input_ids": batch.to(runner.device)})[:, 1:].float().cpu()
        log_probs = logits.log_softmax(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)
        rows.append(torch.cat([log_probs[..., classes], entropy], -1).reshape(-1, len(classes) + 1))
    return torch.cat(rows)


def next_token_targets(runner, split) -> NextTokenTargets:
    """Return the distribution targets of ``split``."""
    classes = split.fit.reshape(-1).bincount().topk(NEXT_TOKEN_CLASSES).indices
    seq = split.test.shape[1] - 1
    return NextTokenTargets(
        distribution_rows(runner, split.fit, classes),
        distribution_rows(runner, split.test, classes),
        split.fit[:, 1:].reshape(-1),
        split.test[:, 1:].reshape(-1),
        split.test_docs.repeat_interleave(seq),
    )


def _distribution_scores(prediction: torch.Tensor, target: torch.Tensor, floor) -> dict:
    """Return a prediction's scores. ``floor`` is the lookup baseline's prediction."""
    error = (target[:, :-1] - prediction[:, :-1]).square().sum()
    floor_error = (target[:, :-1] - floor[:, :-1]).square().sum()
    top = prediction[:, :-1].argmax(1) == target[:, :-1].argmax(1)
    return {
        "log_prob_r2": explained(prediction[:, :-1], target[:, :-1]),
        "entropy_r2": explained(prediction[:, -1:], target[:, -1:]),
        "top_agreement": top.float().mean().item(),
        "error_left_after_lookup": (error / floor_error).item(),
    }


def next_token_row(runner, split, targets: NextTokenTargets, layer: int) -> dict:
    """Return how well each side predicts the model's next-token distribution."""
    sides = read_sides(runner, split.fit, split.test, layer)
    floor = token_lookup(targets.tokens_fit, targets.y_fit, targets.tokens_test)
    row = {"part": "targets_next_token", "layer": layer}
    row["token_lookup"] = _distribution_scores(floor, targets.y_test, floor)
    predictions = {name: ridge(a, targets.y_fit, b) for name, (a, b) in sides.items()}
    for name, prediction in predictions.items():
        row[name] = _distribution_scores(prediction, targets.y_test, floor)

    def margin(side: str):
        def metric(rows: torch.Tensor) -> float:
            target = targets.y_test[rows, :-1]
            null = explained(predictions["null_0"][rows, :-1], target)
            return explained(predictions[side][rows, :-1], target) - null

        return metric

    for side in ("negative", "positive"):
        row[f"{side}_minus_null_interval"] = bootstrap_interval(margin(side), targets.groups_test)
    return row


@dataclass
class TokenRows:
    """A set of documents as padded windows, with the labels of the tokens that it keeps."""

    ids: torch.Tensor  # [documents, WINDOW_TOKENS]
    keep: torch.Tensor  # [documents × (WINDOW_TOKENS − 1)]: not a pad and not a first token
    y: torch.Tensor  # [kept tokens, labels]
    tokens: torch.Tensor  # [kept tokens]
    groups: torch.Tensor  # [kept tokens]: the document of each


def label_tokens(offsets, spans, labels: list[str]) -> torch.Tensor:
    """Return ``[tokens, labels]``: 1 where a token's characters overlap a span of the label."""
    index = {label: i for i, label in enumerate(labels)}
    y = torch.zeros(len(offsets), len(labels))
    for span in spans:
        if span.label not in index:
            continue
        for position, (start, end) in enumerate(offsets):
            if start < span.end and end > span.start:
                y[position, index[span.label]] = 1
    return y


def token_rows(tokenizer, docs, labels: list[str], max_tokens: int) -> TokenRows:
    """Return ``docs`` up to ``max_tokens`` kept tokens. Each is cut to one window."""
    ids, keep, ys, tokens, groups = [], [], [], [], []
    pad = tokenizer.eos_token_id
    kept = 0
    for doc, (text, spans) in enumerate(docs):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=WINDOW_TOKENS,
            return_offsets_mapping=True,
        )
        row = encoded["input_ids"]
        if len(row) < 2:
            continue
        ids.append(row + [pad] * (WINDOW_TOKENS - len(row)))
        keep.append([i < len(row) for i in range(1, WINDOW_TOKENS)])
        ys.append(label_tokens(encoded["offset_mapping"], spans, labels)[1:])
        tokens.extend(row[1:])
        groups.extend([doc] * (len(row) - 1))
        kept += len(row) - 1
        if kept >= max_tokens:
            break
    return TokenRows(
        torch.tensor(ids),
        torch.tensor(keep).reshape(-1),
        torch.cat(ys),
        torch.tensor(tokens),
        torch.tensor(groups),
    )


def ambiguous_rows(fit: TokenRows, test: TokenRows, column: int) -> torch.Tensor:
    """Return a mask of test tokens whose type has the label, and lacks it, in the fit rows."""
    vocab = int(max(fit.tokens.max(), test.tokens.max())) + 1
    positive = fit.y[:, column].bool()
    with_label = fit.tokens[positive].bincount(minlength=vocab) > 0
    without_label = fit.tokens[~positive].bincount(minlength=vocab) > 0
    return (with_label & without_label)[test.tokens]


def _concept_scores(prediction, y, masks) -> tuple[list[float], list[float | None]]:
    """Return each label's AUROC on all rows, and on its ambiguous rows if they are enough."""
    overall, ambiguous = [], []
    for column, mask in enumerate(masks):
        overall.append(auroc(prediction[:, column], y[:, column]))
        positives = int(y[mask, column].sum())
        enough = min(positives, int(mask.sum()) - positives) >= MIN_AMBIGUOUS
        ambiguous.append(auroc(prediction[mask, column], y[mask, column]) if enough else None)
    return overall, ambiguous


def _mean(values) -> float | None:
    """Return the mean of the values that are not ``None``, or ``None`` if there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def concept_row(runner, fit: TokenRows, test: TokenRows, labels, layer: int) -> dict:
    """Return each side's AUROC for each label, against the lookup and the null."""
    sides = read_sides(runner, fit.ids, test.ids, layer, fit.keep, test.keep)
    masks = [ambiguous_rows(fit, test, column) for column in range(len(labels))]
    predictions = {name: ridge(a, fit.y, b) for name, (a, b) in sides.items()}
    predictions["token_lookup"] = token_lookup(fit.tokens, fit.y, test.tokens)
    row = {"part": "targets_concept", "layer": layer, "labels": labels}
    row["positives"] = test.y.sum(0).int().tolist()
    for name, prediction in predictions.items():
        overall, ambiguous = _concept_scores(prediction, test.y, masks)
        row[name] = {"auroc": overall, "auroc_ambiguous": ambiguous}

    def margin(side: str, use_ambiguous: bool):
        def metric(rows: torch.Tensor) -> float:
            y, values = test.y[rows], []
            for column, mask in enumerate(masks):
                chosen = mask[rows] if use_ambiguous else torch.ones(len(rows)).bool()
                if use_ambiguous and row[side]["auroc_ambiguous"][column] is None:
                    continue
                labelled = y[chosen, column]
                real = auroc(predictions[side][rows][chosen, column], labelled)
                null = auroc(predictions["null_0"][rows][chosen, column], labelled)
                values.append(real - null)
            return _mean(values) or 0.0

        return metric

    for side in ("negative", "positive"):
        for use_ambiguous, key in ((False, "all"), (True, "ambiguous")):
            row[f"{side}_minus_null_{key}_interval"] = bootstrap_interval(
                margin(side, use_ambiguous), test.groups
            )
    return row


def concept_rows(tokenizer, dataset: str) -> tuple[TokenRows, TokenRows, list[str], dict]:
    """Return fit rows, test rows, the labels that have enough tokens, and the data's source."""
    loaded = load_concept_split(dataset, max_chars=CONCEPT_MAX_CHARS)
    labels = [label for label, _ in loaded.counts().most_common(MAX_LABELS)]
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(len(loaded.docs), generator=generator).tolist()
    docs = [loaded.docs[i] for i in order]
    cut = int(len(docs) * FIT_SHARE)
    fit = token_rows(tokenizer, docs[:cut], labels, MAX_FIT_TOKENS)
    test = token_rows(tokenizer, docs[cut:], labels, MAX_TEST_TOKENS)
    enough = (fit.y.sum(0) >= MIN_POSITIVES) & (test.y.sum(0) >= MIN_POSITIVES)
    fit.y, test.y = fit.y[:, enough], test.y[:, enough]
    kept = [label for label, keep in zip(labels, enough.tolist(), strict=True) if keep]
    source = {
        "dataset": dataset,
        "hub": loaded.dataset,
        "revision": loaded.revision,
        "license": loaded.license,
        "fit_tokens": len(fit.tokens),
        "test_tokens": len(test.tokens),
    }
    return fit, test, kept, source


@click.group()
def cli():
    """Decode one side of the activations against targets other than token identity."""


@cli.command("next-token")
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="auto", show_default=True)
@click.option("--layers", type=int, multiple=True, help="Default: every layer.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def next_token(corpus, device, layers, out):
    """Append each layer's next-token distribution row to ``--out``."""
    runner = _load(GPT2, resolve_device(device), torch.float32, None, None)
    split = load_split(runner.tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    targets = next_token_targets(runner, split)
    for layer in layers or range(runner.n_layers):
        row = next_token_row(runner, split, targets, layer)
        write_row(out, {"model": GPT2, "sample": split.sample, **row})
        click.echo(f"next-token layer {layer} done", err=True)


@cli.command()
@click.option("--dataset", "datasets", multiple=True, type=click.Choice(sorted(CONCEPT_PRESETS)))
@click.option("--device", default="auto", show_default=True)
@click.option("--layers", type=int, multiple=True, default=CONCEPT_LAYERS, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def concepts(datasets, device, layers, out):
    """Append each dataset and layer's concept row to ``--out``."""
    runner = _load(GPT2, resolve_device(device), torch.float32, None, None)
    for dataset in datasets or CONCEPT_DATASETS:
        fit, test, labels, source = concept_rows(runner.tokenizer, dataset)
        click.echo(f"{dataset}: {labels}, {source}", err=True)
        for layer in layers:
            row = concept_row(runner, fit, test, labels, layer)
            write_row(out, {"model": GPT2, **source, **row})
            click.echo(f"{dataset} layer {layer} done", err=True)


if __name__ == "__main__":
    cli()
