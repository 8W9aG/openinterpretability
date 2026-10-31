"""Null calibration: how often the review verifies a claim that should fail.

The protocol's false-positive rate, measured on two kinds of target that should never be
VERIFIED (``docs/analyses/null-calibration.md``):

* ``shuffle``: each record's own probes and activations, with the hypothesis's labels
  (hard negative or positive holdout) permuted among its measured probes, so the
  "hypothesis" predicts nothing about the unit. No model is run; only the tokenizer, for
  the confound baselines.
* ``swap``: each VERIFIED neuron pole's hypothesis, counterfactuals and probe set, scored
  against a random other neuron of the same layer and pole. The other neuron is scanned
  as its own audit would scan it (same corpus, sample and windows), and the review is
  re-run on it under the current policy, without causal metrics or wiring (not gated).

Each verdict is also read *without the proposer's gates* (counterfactual reproduction and
lookahead consistency): a proposer measures both before submitting, so they catch
transcription errors, not wrong hypotheses (``docs/PROTOCOL.md`` §4).

    A=docs/analyses/null-calibration
    uv run python scripts/null_calibration.py shuffle --out $A-shuffle.jsonl
    uv run python scripts/null_calibration.py swap --out $A-swap.jsonl
    uv run python scripts/null_calibration.py report $A-shuffle.jsonl $A-swap.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import click

from openinterp.agents.reviewer import (
    DEFAULT_POLICY,
    _probe_score,
    decide,
    lookahead_result,
    measure_counterfactuals,
    measure_probes,
    probes_from_verdict,
    review_metrics,
    scan_active_share,
    settle_first_token_floor,
)
from openinterp.core.schemas import AuditRecord

PROBE_KINDS = ("hard_negative", "positive_holdout")
PROPOSER_GATES = ("counterfactual_reproduction", "lookahead_consistent")
PROTOCOLS = ("0.7", "0.8")  # records with a corpus sample and a confound profile


def native_records(wiki_dir: Path):
    """(path, record) for every native neuron-pole record judged under ``PROTOCOLS``."""
    for path in sorted(wiki_dir.glob("**/*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        ident = record.submission.identifier
        if ident.kind == "mlp_neuron_pole" and record.verdict.policy.protocol_version in (
            PROTOCOLS
        ):
            yield path, record


def probe_verified(gates: dict[str, bool]) -> bool:
    """Every gate but the proposer's passes."""
    return all(ok for gate, ok in gates.items() if gate not in PROPOSER_GATES)


def outcome_row(path: Path, record: AuditRecord) -> dict:
    ident = record.submission.identifier
    return {
        "record": str(path),
        "model": ident.model_name,
        "layer": ident.layer,
        "neuron": ident.neuron_idx,
        "pole": ident.pole,
        "status": record.verdict.status,
        "selection": record.submission.provenance.selection,
    }


@click.group()
def cli() -> None:
    """Null calibration of the review (docs/analyses/null-calibration.md)."""


# --- shuffle ------------------------------------------------------------------------


@cli.command()
@click.option("--wiki-dir", default="wiki/experiments", type=click.Path(path_type=Path))
@click.option("--permutations", default=200, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def shuffle(wiki_dir, permutations, seed, out):
    """Permute each record's probe labels and re-decide from its stored activations."""
    from transformers import AutoTokenizer

    tokenizers: dict[str, object] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for path, record in native_records(wiki_dir):
            model = record.submission.identifier.model_name
            if model not in tokenizers:
                tokenizers[model] = AutoTokenizer.from_pretrained(model)
            row = shuffle_record(record, tokenizers[model], permutations, seed)
            if row is not None:
                fh.write(json.dumps(outcome_row(path, record) | row) + "\n")
    click.echo(f"Wrote {out}")


def shuffle_record(record: AuditRecord, tokenizer, permutations: int, seed: int):
    """Counts of VERIFIED (all gates, and without the proposer's) over label permutations,
    and the unpermuted verdict re-derived the same way as a check. ``None`` if the record
    has no measured probes or no scale."""
    sub = record.submission
    results = record.verdict.adversarial_test_results
    probes = [r for r in results if r.kind in PROBE_KINDS and r.measured]
    if not probes or not sub.max_activation:
        return None
    baselines = shuffle_baselines(sub, tokenizer, [r.prompt for r in probes], DEFAULT_POLICY)
    labels = [bool(r.expected_active) for r in probes]
    rng = random.Random(f"{seed}/{sub.identifier.slug}/{record.verdict.timestamp}")
    status, gates = relabel(record, probes, labels, baselines)
    counts = Counter()
    for _ in range(permutations):
        permuted, permuted_gates = relabel(
            record, probes, rng.sample(labels, len(labels)), baselines
        )
        counts["verified"] += permuted == "VERIFIED"
        counts["probe_verified"] += probe_verified(permuted_gates)
    original = {"rederived": status, "rederived_probe": probe_verified(gates)}
    return {"permutations": permutations, **counts, **original}


def relabel(record: AuditRecord, probes, labels, baselines):
    """The verdict and gates with ``labels`` as the probes' expected activity, from the
    activity the record measured."""
    from openinterp.agents.reviewer import AdversarialProbe

    sub, policy = record.submission, DEFAULT_POLICY
    actual = [r.passed == bool(r.expected_active) for r in probes]
    relabelled = [
        r.model_copy(update={"expected_active": e, "passed": a == e})
        for r, e, a in zip(probes, labels, actual, strict=True)
    ]
    others = [r for r in record.verdict.adversarial_test_results if r not in probes]
    baseline = None
    if baselines is not None:
        as_probes = [
            AdversarialProbe(text=r.prompt, expected_active=e, rationale="")
            for r, e in zip(probes, labels, strict=True)
        ]
        scored = {n: _probe_score(as_probes, p, actual, policy) for n, p in baselines.items()}
        best = max(sorted(scored), key=lambda n: scored[n])
        baseline = (best, round(scored[best], 6))
    metrics = review_metrics(
        relabelled + others,
        policy,
        independent=record.verdict.reviewer_agent_model != sub.submitting_agent_model,
        baseline=baseline,
        active_share=scan_active_share(sub, policy),
        first_token_floor=sub.first_token_floor,
    )
    return decide(metrics, policy), metrics.gates


def shuffle_baselines(sub, tokenizer, texts, policy) -> dict[str, list[bool]] | None:
    """The confound baselines' predictions, or ``None`` without a confound profile (the
    gate then fails, as in a review)."""
    from openinterp.harness.confounds import baseline_predictions

    if sub.confounds is None:
        return None
    threshold = policy.activation_threshold * sub.max_activation
    skip_first = bool(sub.first_token_floor)
    found = baseline_predictions(sub.confounds, tokenizer, texts, threshold, skip_first)
    return found | {"never active": [False] * len(texts)}


# --- swap ---------------------------------------------------------------------------


@cli.command()
@click.option("--wiki-dir", default="wiki/experiments", type=click.Path(path_type=Path))
@click.option(
    "--corpus",
    "corpora",
    multiple=True,
    default=["corpus.jsonl"],
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Corpus files records may have scanned (matched by SHA-256; the builtin sample "
    "is always available). Records whose corpus isn't given are skipped.",
)
@click.option("--model", "models", multiple=True, help="Only these models (default: all).")
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def swap(wiki_dir, corpora, models, seed, device, out):
    """Score each VERIFIED pole's hypothesis against a random neuron of its layer."""
    import openinterp.harness.scan
    from openinterp.cli import _load_runner
    from openinterp.cli_evidence import open_scan_cache
    from openinterp.evidence import read_corpus

    openinterp.harness.scan.PartnerSearch = _NoPartners
    by_sha = {read_corpus(None)[2]: None} | {read_corpus(p)[2]: p for p in corpora}
    targets, skipped = verified_units(wiki_dir, models), Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["record"] for line in out.open()} if out.exists() else set()
    cache, runners, samples = open_scan_cache(True), {}, {}
    with out.open("a") as fh:
        for path, record in targets:
            prov = record.submission.provenance
            if prov.corpus_sha256 not in by_sha:
                skipped[prov.corpus] += 1
                continue
            if str(path) in done:
                continue
            model = record.submission.identifier.model_name
            if model not in runners:
                runners[model] = _load_runner(model, device, "float32", prov.model_revision)
            corpus = by_sha[prov.corpus_sha256]
            row = swap_record(record, runners[model], corpus, seed, cache, samples)
            fh.write(json.dumps(outcome_row(path, record) | row) + "\n")
            fh.flush()
            click.echo(f"{record.submission.identifier} -> N{row['null_neuron']}: {row['outcome']}")
    for corpus, n in skipped.items():
        click.echo(f"skipped {n} record(s) scanned on {corpus}: corpus not given", err=True)


class _NoPartners:
    """Stands in for the scan's partner search, which feeds only joint ablation (not
    gated, and skipped here) and costs most of a replayed scan."""

    def __init__(self, width: int) -> None:
        pass

    def update(self, target, layer_acts) -> None:
        pass

    def partners(self, exclude, active_threshold, *, k=10):
        return []


def verified_units(wiki_dir: Path, models):
    """The newest VERIFIED record of each neuron pole, as (path, record)."""
    newest: dict[str, tuple[Path, AuditRecord]] = {}
    for path, record in native_records(wiki_dir):
        ident = record.submission.identifier
        if record.verdict.status != "VERIFIED" or (models and ident.model_name not in models):
            continue
        seen = newest.get(ident.slug)
        if seen is None or seen[1].verdict.timestamp < record.verdict.timestamp:
            newest[ident.slug] = (path, record)
    return [newest[slug] for slug in sorted(newest)]


def swap_record(record: AuditRecord, runner, corpus, seed: int, cache, samples) -> dict:
    """Scan a random other neuron of the record's layer and pole and review the record's
    submission and probes against it."""
    from openinterp.evidence import Unauditable, check_default_on, run_scan

    sub = record.submission
    ident = sub.identifier
    rng = random.Random(f"{seed}/{ident.slug}")
    neuron = rng.choice([n for n in range(runner.mlp_width(ident.layer)) if n != ident.neuron_idx])
    null = ident.model_copy(update={"neuron_idx": neuron})
    request = scan_request(sub, null, corpus)
    texts = scan_texts(runner, request, samples)
    scan = run_scan(runner, request, null, None, texts, cache)
    row: dict = {"null_neuron": neuron}
    if not scan.examples or scan.examples[0].max_activation <= 0:
        return row | {"outcome": "dead"}
    try:
        check_default_on(null, scan.values, scan.examples[0].max_activation)
    except Unauditable:
        return row | {"outcome": "default_on"}
    return row | review_null(record, runner, null_submission(sub, null, scan))


def scan_request(sub, null, corpus):
    from openinterp.evidence import ScanRequest

    prov = sub.provenance
    sample = prov.corpus_sample
    return ScanRequest(
        model_name=null.model_name,
        layer=null.layer,
        neuron=null.neuron_idx,
        pole=null.pole,
        corpus=corpus,
        top_k=max(len(sub.top_examples), 1),
        dtype=prov.dtype,
        window_tokens=prov.snippet_tokens,
        stride=prov.snippet_stride,
        max_tokens=sample.max_tokens if sample else 0,
        sample_seed=sample.seed if sample else 0,
    )


def scan_texts(runner, request, samples):
    """The corpus sample ``request`` reads, drawn once per corpus, sample and windows."""
    from openinterp.evidence import load_texts

    key = (request.corpus, request.max_tokens, request.sample_seed, request.window_tokens)
    key += (request.stride,)
    if key not in samples:
        samples[key] = load_texts(runner, request)
    return samples[key]


def null_submission(sub, null, scan):
    """The submission with everything its scan measured replaced by the null neuron's."""
    from openinterp.harness.hooks import activation_histogram

    return sub.model_copy(
        update={
            "identifier": null,
            "max_activation": scan.examples[0].max_activation,
            "top_examples": scan.examples,
            "activation_histogram": activation_histogram(scan.values),
            "first_token_floor": scan.first_token_floor,
            "confounds": scan.confounds,
            "correlated_neurons": scan.correlated,
        }
    )


def review_null(record: AuditRecord, runner, sub) -> dict:
    """``evaluate_submission`` without causal metrics or wiring, which aren't gated."""
    from openinterp.agents.reviewer import confound_baseline

    policy = DEFAULT_POLICY
    probes = probes_from_verdict(record.verdict)
    sub, unit, scale = settle_first_token_floor(sub, runner, policy)
    threshold = policy.activation_threshold * scale
    results, measured, actual = measure_probes(runner, unit, probes, scale, threshold)
    results += measure_counterfactuals(runner, unit, sub, scale, policy)
    results.append(lookahead_result(runner, unit, sub))
    metrics = review_metrics(
        results,
        policy,
        independent=record.verdict.reviewer_agent_model != sub.submitting_agent_model,
        baseline=confound_baseline(sub, runner, measured, actual, threshold, policy),
        active_share=scan_active_share(sub, policy),
        first_token_floor=sub.first_token_floor,
    )
    return {
        "outcome": decide(metrics, policy),
        "probe_verified": probe_verified(metrics.gates),
        "gates": metrics.gates,
        "specificity": metrics.specificity,
        "sensitivity": metrics.sensitivity,
        "balanced_accuracy": metrics.balanced_accuracy,
        "active_probes": sum(actual),
        "measured_probes": len(actual),
    }


# --- report -------------------------------------------------------------------------


@cli.command()
@click.argument("shuffle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("swap_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def report(shuffle_file, swap_file):
    """Print the Markdown tables of docs/analyses/null-calibration.md."""
    shuffled = [json.loads(line) for line in shuffle_file.open()]
    swapped = [json.loads(line) for line in swap_file.open()]
    click.echo(shuffle_table(shuffled))
    click.echo(swap_table(swapped))
    click.echo(gate_table(swapped))


def wilson(k: int, n: int, z: float = 1.96) -> str:
    """``k/n`` as a percentage with its 95% Wilson interval."""
    if not n:
        return "—"
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5 / (1 + z2 / n)
    return f"{100 * p:.1f}% [{100 * max(centre - half, 0):.1f}, {100 * (centre + half):.1f}]"


def short(model: str) -> str:
    return {"openai-community/gpt2": "GPT-2", "Qwen/Qwen2.5-0.5B": "Qwen"}.get(model, model)


def shuffle_table(rows) -> str:
    """Per model and original verdict: records, the mean per-record VERIFIED rate over
    permutations, and the share of records any permutation verified."""
    groups = defaultdict(list)
    for r in rows:
        groups[(short(r["model"]), r["status"])].append(r)
        groups[(short(r["model"]), "all")].append(r)
    lines = [
        "| Model | Original verdict | Records | VERIFIED | Without proposer's gates "
        "| Records ever VERIFIED | Re-derived = recorded |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (model, status), group in sorted(groups.items()):
        rate = sum(r.get("verified", 0) / r["permutations"] for r in group) / len(group)
        probe = sum(r.get("probe_verified", 0) / r["permutations"] for r in group) / len(group)
        ever = sum(r.get("verified", 0) > 0 for r in group)
        same = sum(r["rederived"] == r["status"] for r in group)
        lines.append(
            f"| {model} | {status} | {len(group)} | {100 * rate:.2f}% | {100 * probe:.2f}% "
            f"| {ever} | {same}/{len(group)} |"
        )
    return "\n".join(lines) + "\n"


def swap_table(rows) -> str:
    """Per model and pole: outcomes of the swapped reviews and the false-positive rates."""
    groups = defaultdict(list)
    for r in rows:
        groups[(short(r["model"]), r["pole"])].append(r)
        groups[(short(r["model"]), "both")].append(r)
    lines = [
        "| Model | Pole | Hypotheses | Dead | Default-on | REJECTED | NEEDS_REVISION | VERIFIED "
        "| False positives | Without proposer's gates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (model, pole), group in sorted(groups.items()):
        out = Counter(r["outcome"] for r in group)
        probe = sum(bool(r.get("probe_verified")) for r in group)
        lines.append(
            f"| {model} | {pole} | {len(group)} | {out['dead']} | {out['default_on']} "
            f"| {out['REJECTED']} | {out['NEEDS_REVISION']} | {out['VERIFIED']} "
            f"| {wilson(out['VERIFIED'], len(group))} | {wilson(probe, len(group))} |"
        )
    return "\n".join(lines) + "\n"


def gate_table(rows) -> str:
    """How often each gate failed on the swapped reviews that reached the review."""
    reviewed = [r for r in rows if "gates" in r]
    failed = Counter(g for r in reviewed for g, ok in r["gates"].items() if not ok)
    lines = ["| Gate | Failed |", "| --- | --- |"]
    for gate in reviewed[0]["gates"] if reviewed else ():
        lines.append(f"| `{gate}` | {failed[gate]}/{len(reviewed)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cli()
