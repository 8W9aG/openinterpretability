"""A frozen, versioned benchmark for tools that explain neurons (docs/benchmark.md).

The benchmark is a set of native MLP neuron poles and probe texts. Each probe has a label:
whether the model measured the pole as active on it. An explainer writes one explanation
per unit. A judge applies each explanation to the unit's probes. The score is the
balanced accuracy of the judge's labels against the measured ones.

:func:`freeze` builds a version from the registry's records. It takes only units that a
random draw chose, so that the scores are not read as covering a hand-picked set. A hash
of each unit's slug puts the unit in the dev split or the test split. The dev split
publishes the probes and labels that the records hold. Those records are public, so the
test split uses fresh probes instead. A maintainer measures them
(``openinterp benchmark measure``) and keeps them out of the repository. The manifest
publishes only their SHA-256 (:func:`seal`).

This module loads no model. ``openinterp/cli_benchmark.py`` has the commands.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from openinterp.builder.wiki import experiment_id
from openinterp.core.schemas import AuditRecord, ReviewPolicy
from openinterp.core.stats import balanced_accuracy, effective_counts

NAME = "openinterp-explainer-benchmark"
LICENSE = "CC-BY-4.0"  # the records' licence (wiki/LICENSE.md)
DEFAULT_DIR = Path("benchmark")
TEST_FRACTION = 0.2
MIN_PER_CLASS = 3  # active and inactive probes a unit needs, so that both rates exist
PROBE_KINDS = ("hard_negative", "positive_holdout")
CHANCE = 0.5  # the score of a unit that has no explanation
BOOTSTRAP_SAMPLES = 2000

MANIFEST = "manifest.json"
UNITS = "units.jsonl"
DEV_PROBES = "dev/probes.jsonl"
DEV_EXPLANATIONS = "dev/registry_explanations.jsonl"
TEST_EXPLANATIONS = "test/registry_explanations.jsonl"


@dataclass
class Unit:
    """One benchmark unit: a neuron pole, its split and what measures it."""

    unit: str  # the slug, e.g. ``Qwen--Qwen2.5-0.5B/L3/N7-positive``
    model: str
    model_revision: str | None
    layer: int
    neuron: int
    pole: str
    split: str
    records: list[str] = field(default_factory=list)  # experiment IDs, oldest first
    # What fresh test probes are measured against: the newest record's scan maximum and
    # threshold, and whether the first token is measured.
    scale: float | None = None
    threshold: float | None = None
    first_token_floor: bool | None = None


def split_of(slug: str, version: str, test_fraction: float = TEST_FRACTION) -> str:
    """Return ``"test"`` or ``"dev"`` for ``slug``, from a hash salted with ``version``."""
    digest = hashlib.sha256(f"{version}:{slug}".encode()).hexdigest()
    return "test" if int(digest[:8], 16) / 0x1_0000_0000 < test_fraction else "dev"


ORDERS = ("slug", "shuffled")  # the orders that `benchmark score --limit` takes units in


def ordered(units: Iterable[Unit], order: str, version: str) -> list[Unit]:
    """``units`` in slug order, or in a shuffled order fixed by ``version`` (issue #3302).

    A shuffled unit's place is a hash of its slug, salted with ``version`` and its split.
    So the first N units of a split are a random sample of it, the same one on every run.
    Adding units to a later version doesn't move the others within it.
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; use one of {', '.join(ORDERS)}")
    if order == "slug":
        return sorted(units, key=lambda u: u.unit)

    def key(u: Unit) -> str:
        return hashlib.sha256(f"order:{version}:{u.split}:{u.unit}".encode()).hexdigest()

    return sorted(units, key=key)


def probe_labels(record: AuditRecord) -> list[tuple[str, bool, str]]:
    """Return each measured reviewer probe in ``record`` as (text, active, kind).

    A probe passed when the model's activity matched the hypothesis's label. So the
    activity is the label when the probe passed, and the opposite when it failed.
    """
    rows = []
    for r in record.verdict.adversarial_test_results:
        if r.kind in PROBE_KINDS and r.measured and r.expected_active is not None:
            rows.append((r.prompt, r.expected_active == r.passed, r.kind))
    return rows


def _eligible(records: Sequence[AuditRecord]) -> bool:
    first = records[0].submission
    prov = first.provenance
    return first.identifier.kind == "mlp_neuron_pole" and bool(prov and prov.selection == "random")


def _unit_probes(slug: str, records: Sequence[AuditRecord]) -> list[dict]:
    """Every distinct probe text over ``records``, dropping texts whose labels disagree."""
    seen: dict[str, dict] = {}
    conflicts: set[str] = set()
    for record in records:
        for text, active, kind in probe_labels(record):
            row = {
                "unit": slug,
                "text": text,
                "active": active,
                "kind": kind,
                "record": experiment_id(record),
            }
            if text in seen and seen[text]["active"] != active:
                conflicts.add(text)
            seen.setdefault(text, row)
    return [row for text, row in seen.items() if text not in conflicts]


def _explanation(slug: str, records: Sequence[AuditRecord]) -> dict:
    """The registry's baseline for a unit: its newest VERIFIED explanation, else its newest."""
    verified = [r for r in records if r.verdict.status == "VERIFIED"]
    best = (verified or list(records))[-1]
    return {
        "unit": slug,
        "explanation": best.submission.natural_language_explanation,
        "status": best.verdict.status,
        "record": experiment_id(best),
    }


def _unit(slug: str, records: Sequence[AuditRecord], split: str) -> Unit:
    newest = records[-1].submission
    ident = newest.identifier
    policy = records[-1].verdict.policy or ReviewPolicy()
    scale = newest.max_activation
    return Unit(
        unit=slug,
        model=ident.model_name,
        model_revision=newest.provenance.model_revision if newest.provenance else None,
        layer=ident.layer,
        neuron=ident.neuron_idx,
        pole=ident.pole,
        split=split,
        records=[experiment_id(r) for r in records],
        scale=scale,
        threshold=None if scale is None else round(policy.activation_threshold * scale, 6),
        first_token_floor=newest.first_token_floor,
    )


@dataclass
class Frozen:
    """What :func:`freeze` returns, before it is written."""

    manifest: dict
    units: list[Unit]
    dev_probes: list[dict]
    explanations: dict[str, list[dict]]  # per split


def freeze(
    records: Iterable[AuditRecord],
    version: str,
    *,
    test_fraction: float = TEST_FRACTION,
    min_per_class: int = MIN_PER_CLASS,
) -> Frozen:
    """Select the units in ``records``, split them and collect the dev probes.

    ``records`` must be oldest first, as :func:`openinterp.builder.wiki.load_archive`
    returns them. A unit qualifies when its first record was a random draw of a native
    neuron pole. Its probes must also have at least ``min_per_class`` of each label.
    """
    by_unit: dict[str, list[AuditRecord]] = {}
    for record in records:
        by_unit.setdefault(record.submission.identifier.slug, []).append(record)
    units, dev_probes = [], []
    explanations: dict[str, list[dict]] = {"dev": [], "test": []}
    skipped = {"not_random_neuron_pole": 0, "too_few_probes": 0}
    for slug in sorted(by_unit):
        group = by_unit[slug]
        if not _eligible(group):
            skipped["not_random_neuron_pole"] += 1
            continue
        probes = _unit_probes(slug, group)
        active = sum(p["active"] for p in probes)
        if min(active, len(probes) - active) < min_per_class:
            skipped["too_few_probes"] += 1
            continue
        split = split_of(slug, version, test_fraction)
        units.append(_unit(slug, group, split))
        explanations[split].append(_explanation(slug, group))
        if split == "dev":
            dev_probes.extend(probes)
    manifest = _manifest(version, test_fraction, min_per_class, skipped, units, dev_probes)
    return Frozen(manifest, units, dev_probes, explanations)


def _manifest(version, test_fraction, min_per_class, skipped, units, dev_probes) -> dict:
    return {
        "name": NAME,
        "version": version,
        "license": LICENSE,
        "selection": {
            "unit_kind": "mlp_neuron_pole",
            "first_record_selection": "random",
            "min_probes_per_label": min_per_class,
            "skipped_units": skipped,
        },
        "split": {"test_fraction": test_fraction, "salt": version},
        "units": {s: sum(u.split == s for u in units) for s in ("dev", "test")},
        "units_per_model": _per_model(units),
        "dev_probes": len(dev_probes),
        "newest_record": max((u.records[-1] for u in units), default=None),
        "test_labels_sha256": None,
        "test_probes": None,
    }


def _per_model(units: Sequence[Unit]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for u in units:
        counts = out.setdefault(u.model, {"dev": 0, "test": 0})
        counts[u.split] += 1
    return dict(sorted(out.items()))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write ``rows`` to ``path``, one JSON object per line, keys sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """Read the JSON objects in ``path``, one per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write(frozen: Frozen, out_dir: Path) -> None:
    """Write a frozen version to ``out_dir`` (e.g. ``benchmark/v1``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, frozen.manifest)
    write_jsonl(out_dir / UNITS, (u.__dict__ for u in frozen.units))
    write_jsonl(out_dir / DEV_PROBES, frozen.dev_probes)
    write_jsonl(out_dir / DEV_EXPLANATIONS, frozen.explanations["dev"])
    write_jsonl(out_dir / TEST_EXPLANATIONS, frozen.explanations["test"])


def write_manifest(out_dir: Path, manifest: dict) -> None:
    """Write ``manifest.json`` in ``out_dir``."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / MANIFEST).write_text(text, encoding="utf-8")


def read_manifest(bench_dir: Path) -> dict:
    """Read ``manifest.json`` in ``bench_dir``."""
    return json.loads((Path(bench_dir) / MANIFEST).read_text(encoding="utf-8"))


def read_units(bench_dir: Path) -> dict[str, Unit]:
    """Read the units of a version, by slug."""
    return {row["unit"]: Unit(**row) for row in read_jsonl(Path(bench_dir) / UNITS)}


def labels_sha256(path: Path) -> str:
    """The SHA-256 of a labels file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(bench_dir: Path, labels_path: Path) -> dict:
    """Sort the private test labels, and put their hash and counts in the manifest.

    Returns the new manifest. Fails if the labels name a unit outside the test split, or
    leave a test unit with too few probes of a label.
    """
    units = read_units(bench_dir)
    manifest = read_manifest(bench_dir)
    rows = sorted(read_jsonl(labels_path), key=lambda r: (r["unit"], r["text"]))
    _check_test_labels(rows, units, manifest["selection"]["min_probes_per_label"])
    write_jsonl(labels_path, rows)
    manifest["test_labels_sha256"] = labels_sha256(labels_path)
    manifest["test_probes"] = len(rows)
    write_manifest(Path(bench_dir), manifest)
    return manifest


def _check_test_labels(rows: Sequence[dict], units: dict[str, Unit], minimum: int) -> None:
    counts: dict[str, list[int]] = {}
    for row in rows:
        unit = units.get(row["unit"])
        if unit is None or unit.split != "test":
            raise ValueError(f"{row['unit']}: not a test unit of this version")
        counts.setdefault(row["unit"], [0, 0])[bool(row["active"])] += 1
    missing = sorted(u.unit for u in units.values() if u.split == "test" and u.unit not in counts)
    if missing:
        raise ValueError(f"{len(missing)} test units have no labels, e.g. {missing[0]}")
    thin = sorted(slug for slug, c in counts.items() if min(c) < minimum)
    if thin:
        raise ValueError(f"{len(thin)} test units have under {minimum} probes of a label: {thin}")


def load_labels(bench_dir: Path, split: str, labels_path: Path | None) -> dict[str, list[dict]]:
    """Return the probes of ``split``, by unit.

    The test split's labels come from ``labels_path``, which must match the manifest's hash.
    """
    if split == "dev":
        rows = read_jsonl(Path(bench_dir) / DEV_PROBES)
    else:
        expected = read_manifest(bench_dir)["test_labels_sha256"]
        if expected is None:
            raise ValueError("this version has no sealed test labels yet")
        if labels_path is None:
            raise ValueError("the test split needs --labels, the private labels file")
        if labels_sha256(labels_path) != expected:
            raise ValueError(f"{labels_path} does not match the manifest's test_labels_sha256")
        rows = read_jsonl(labels_path)
    by_unit: dict[str, list[dict]] = {}
    for row in rows:
        by_unit.setdefault(row["unit"], []).append(row)
    return by_unit


def shuffled(probes: Sequence[dict], slug: str) -> list[dict]:
    """``probes`` in an order fixed by ``slug``, so that no label shows in the order."""
    out = sorted(probes, key=lambda p: p["text"])
    random.Random(slug).shuffle(out)
    return out


def unit_score(
    probes: Sequence[dict], predicted: Sequence[bool], cluster_threshold: float
) -> dict[str, float | int | None]:
    """Balanced accuracy of ``predicted`` against the probes' labels.

    Near-duplicate probes count once, as in the review (docs/PROTOCOL.md §4).
    """

    def rate(active: bool):
        pairs = zip(probes, predicted, strict=True)
        rows = [(p["text"], pr == active) for p, pr in pairs if p["active"] is active]
        return effective_counts([t for t, _ in rows], [ok for _, ok in rows], cluster_threshold)

    inactive, active = rate(False), rate(True)
    return {
        "balanced_accuracy": round(balanced_accuracy(inactive, active), 6),
        "specificity": None if inactive.rate is None else round(inactive.rate, 6),
        "sensitivity": None if active.rate is None else round(active.rate, 6),
        "active_probes": active.raw,
        "inactive_probes": inactive.raw,
    }


def summarize(
    rows: Sequence[dict], units: Sequence[Unit], seed: int = 0, split_units: int | None = None
) -> dict:
    """The split's score: mean balanced accuracy over every unit, chance where missing.

    A unit without an explanation scores :data:`CHANCE`, so leaving hard units out does not
    raise the mean. The interval is a bootstrap over units. ``split_units`` is the size of
    the whole split, when ``units`` are only part of it.
    """
    scored = {r["unit"]: r["balanced_accuracy"] for r in rows}
    values = [scored.get(u.unit, CHANCE) for u in units]
    summary = {
        "units": len(units),
        "explained": sum(u.unit in scored for u in units),
        "mean_balanced_accuracy": _mean(values),
        "ci95": _bootstrap(values, seed),
        "per_model": {},
        "split_units": len(units) if split_units is None else split_units,
    }
    for model in sorted({u.model for u in units}):
        mine = [scored.get(u.unit, CHANCE) for u in units if u.model == model]
        summary["per_model"][model] = {"units": len(mine), "mean_balanced_accuracy": _mean(mine)}
    return summary


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _bootstrap(values: Sequence[float], seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [round(means[int(0.025 * len(means))], 6), round(means[int(0.975 * len(means))], 6)]


def score(
    probes_by_unit: dict[str, list[dict]],
    explanations: dict[str, str],
    judge: Callable[[str, str, list[str]], list[bool] | None],
    *,
    done: dict[str, dict] | None = None,
    cluster_threshold: float = ReviewPolicy().probe_cluster_threshold,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Score each explained unit; ``judge(unit, explanation, texts)`` says which texts fire.

    Units in ``done`` (from an earlier, interrupted run) are not judged again. A judge that
    has no answer yet returns ``None``. Then the unit is left out of the rows, and a later
    run scores it.
    """
    rows = []
    for slug in sorted(probes_by_unit):
        if slug not in explanations:
            continue
        if done and slug in done:
            rows.append(done[slug])
            continue
        probes = shuffled(probes_by_unit[slug], slug)
        predicted = judge(slug, explanations[slug], [p["text"] for p in probes])
        if predicted is None:
            continue
        row = {"unit": slug, **unit_score(probes, predicted, cluster_threshold)}
        rows.append(row)
        if on_row:
            on_row(row)
    return rows
