"""Reproduce an explanation record before it merges (issue #4793).

``open-interpretability reproduce`` takes explanation records
(``wiki/explanations/<model>/<prompt id>/<UTC>.json``) next to experiment records. It runs
every measurement that the page shows again, with the model revision and dtype that the
record pins, and makes no LLM calls:

1. The means, from the record's corpus and token budget. The corpus is found by its
   SHA-256, or built again from its stored source (``corpus --same-as``).
2. The attribution of each top unit and group, and the explained and blocked shares. The
   explained and blocked units are the record's own sets, as the registry stood when it was
   written. A record that a change adds must also agree with the registry in the checkout
   (:mod:`openinterpretability.explanation_registry`).
3. Prompt coverage: clean, ablated and kept, the raw value, the random baseline from its
   stored seed, the coverage by kind, and the minimal set.
4. The figures' data: the tokens, the logit, the net attribution, the edges, the reading,
   the ablation table and the random draws.

Numbers must agree within ``atol × scale + rtol × |recorded|``, as for experiments. The
scale of an attribution is the largest top unit's |attribution|. The scale of a share or a
coverage is 1. A reader takes some values as conclusions, so these must agree exactly:

- the answer token and the explained units;
- the top token of the ablation table;
- whether coverage overshoots, and whether the page cautions about coverage.

The minimal set is greedy, and a tie between two units can go either way on another
machine. So a different minimal set is a warning, not a mismatch.

The narration is LLM text and is not reproduced. Each decimal or percentage that it quotes
is looked for among the record's numbers (:func:`unquoted_numbers`). The narrator also
read numbers that the record doesn't keep, such as the plan's gains. So a number that
isn't there is a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import copysign
from pathlib import Path

from openinterpretability.core.explanation_records import ExplanationRecord
from openinterpretability.reproduce import Tolerance

SHARE_SCALE = 1.0  # shares and coverages are fractions


@dataclass
class ExplanationReport:
    """The result of reproducing one explanation record."""

    path: Path | None
    record: ExplanationRecord
    tolerance: Tolerance
    mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compared: int = 0

    @property
    def ok(self) -> bool:
        """True when nothing differed beyond tolerance, and every conclusion agreed."""
        return not self.mismatches

    def number(self, what: str, recorded, reproduced, scale: float, *, warn=False) -> None:
        """Compare one number; a difference is a mismatch, or with ``warn`` a warning."""
        self.compared += 1
        if not self.tolerance.close(recorded, reproduced, scale):
            message = f"{what}: recorded {recorded}, reproduced {reproduced}"
            (self.warnings if warn else self.mismatches).append(message)

    def same(self, what: str, recorded, reproduced, *, warn=False) -> None:
        """Compare one conclusion, which must agree exactly."""
        self.compared += 1
        if recorded != reproduced:
            message = f"{what}: recorded {recorded!r}, reproduced {reproduced!r}"
            (self.warnings if warn else self.mismatches).append(message)


def not_reproducible(record: ExplanationRecord) -> str | None:
    """Why ``record`` can't be reproduced by this harness, or ``None``."""
    from openinterpretability.explain import MEANS_VERSION, WINDOW

    inputs = record.inputs
    if inputs is None:
        return "it was written before explanation records kept their inputs (#4793)"
    if (inputs.means_window, inputs.means_version) != (WINDOW, MEANS_VERSION):
        return (f"its means were read in windows of {inputs.means_window} tokens, version "
                f"{inputs.means_version}; this harness reads {WINDOW}, "
                f"version {MEANS_VERSION}")  # fmt: skip
    return None


def means_corpus(record: ExplanationRecord, corpora: dict[str, list[str]]) -> list[str]:
    """The documents of the record's means corpus. Raises ``ValueError`` if they can't be had.

    ``corpora`` maps a corpus file's SHA-256 to its documents. Otherwise the built-in
    corpus is used when its hash matches, or the corpus is built again from its source.
    """
    from openinterpretability.evidence import read_corpus

    inputs = record.inputs
    sha = inputs.means_corpus_sha256
    if sha in corpora:
        return corpora[sha]
    builtin = read_corpus(None)
    if builtin[2] == sha:
        return builtin[0]
    if inputs.means_corpus_source is None:
        raise ValueError(f"its means corpus (SHA-256 {sha}) is not given, and it has no source")
    return read_corpus(_rebuilt(inputs.means_corpus_source))[0]


def _rebuilt(source) -> Path:
    """The corpus that ``source`` describes, built once into the corpus cache."""
    from openinterpretability.core.cache_dirs import cache_root
    from openinterpretability.corpus_rebuild import rebuild

    root = cache_root("OPENINTERPRETABILITY_REBUILT_CORPORA", "rebuilt-corpora")
    if root is None:
        raise ValueError("the corpus cache is off, so its means corpus can't be built again")
    out = root / f"{source.sha256}.jsonl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        rebuild(source, out)
    return out


def reproduce_explanation(record: ExplanationRecord, runner, texts: list[str], tolerance,
                          path: Path | None = None, groups=None) -> ExplanationReport:  # fmt: skip
    """Measure ``record`` again with ``runner`` and compare (module docstring).

    ``texts`` are the means corpus's documents. ``groups`` are the model's superpositions
    and circuits; by default the repository's (:mod:`openinterpretability.explain_groups`).
    """
    from openinterpretability import explain_groups
    from openinterpretability.explain import explain, unit_means
    from openinterpretability.explain_coverage import prompt_coverage

    inputs, model = record.inputs, record.model_name
    report = ExplanationReport(path, record, tolerance)
    if groups is None:
        groups = explain_groups.superpositions(model) + explain_groups.circuits(model)
    means = unit_means(runner, model, texts, inputs.means_tokens)
    result = explain(runner, model, record.prompt, means, {}, token=inputs.token_id,
                     position=record.position, top=inputs.top, edges=inputs.edges,
                     groups=groups)  # fmt: skip
    order = _with_explained(report, result, inputs.explained)
    report.same("answer", record.answer, result.token)
    _shares(report, record, result, order, inputs.blocked)
    _units(report, record, result)
    draws = {"draws": inputs.random_draws, "seed": inputs.random_seed}
    coverage = prompt_coverage(runner, record.prompt, means, inputs.token_id, record.position,
                               order, **draws)  # fmt: skip
    _coverage(report, record, coverage)
    if record.figures is not None:
        _figures(report, record, runner, means, result, coverage, groups)
    if record.narration is not None:
        missing = unquoted_numbers(record)
        if missing:
            report.warnings.append(
                "the narration quotes numbers that the record doesn't hold: " + ", ".join(missing)
            )
    return report


def _with_explained(report: ExplanationReport, result, explained: list[str]) -> list:
    """``result.order``, with the record's explained set in place of the registry's."""
    from openinterpretability.explain import PoleAttribution

    labels = {u: PoleAttribution(*u, 0.0, 0.0).label for u, *_ in result.order}
    order = [(u, a, s, labels[u] in set(explained)) for u, a, s, _ in result.order]
    found = [labels[u] for u, _, _, done in order if done]
    report.same("explained units", sorted(explained), sorted(found))
    result.verified = sum(a for _, a, _, done in order if done)
    result.order = order
    return order


def _shares(report, record, result, order, blocked: list[str]) -> None:
    from openinterpretability.explain import PoleAttribution

    total = result.total or 1.0
    held = sum(a for u, a, _, _ in order if PoleAttribution(*u, 0.0, 0.0).label in set(blocked))
    report.number("explained share", record.explained_share, result.verified_share, SHARE_SCALE)
    report.number("blocked share", record.blocked_share, held / total, SHARE_SCALE)


def _scale(record) -> float:
    """The largest top unit's |attribution|: the scale of an attribution."""
    return max((abs(u.attribution) for u in record.units), default=1.0) or 1.0


def _units(report, record, result) -> None:
    """Each top unit's and group's attribution and share."""
    scale = _scale(record)
    found = {u.label: u for u in result.poles}
    total = result.total or 1.0
    attributions = {u: a for u, a, *_ in result.order}
    labels = {_label(u): u for u in attributions}
    groups = {g.name: g for g in result.groups}
    for unit in record.units:
        if unit.kind in ("pole", "head"):
            if unit.unit not in labels:
                report.same(f"{unit.unit}", "attributed", "no attribution")
                continue
            again = found.get(unit.unit)  # out of the top: the order keeps |attribution|
            value = (
                again.attribution
                if again
                else copysign(attributions[labels[unit.unit]], unit.attribution)
            )
            report.number(f"{unit.unit} attribution", unit.attribution, value, scale)
            report.number(f"{unit.unit} share", unit.share, abs(value) / total, SHARE_SCALE)
        elif unit.unit in groups:
            g = groups[unit.unit]
            report.number(f"{unit.unit} attribution", unit.attribution, g.attribution, scale)
        else:
            report.warnings.append(f"{unit.unit}: its definition is no longer in the repository")


def _label(unit) -> str:
    from openinterpretability.explain import PoleAttribution

    return PoleAttribution(*unit, 0.0, 0.0).label


def _coverage(report, record, coverage: dict) -> None:
    """Prompt coverage, its random baseline and its detail."""
    report.number("prompt coverage", record.prompt_coverage, coverage["coverage"], SHARE_SCALE)
    report.number("raw coverage", record.raw_coverage, coverage["raw"], SHARE_SCALE)
    report.number("random coverage", record.random_coverage, coverage["random"], SHARE_SCALE)
    again = record.model_copy(update={"prompt_coverage": coverage["coverage"],
                                      "raw_coverage": coverage["raw"],
                                      "random_coverage": coverage["random"]})  # fmt: skip
    report.same("coverage overshoots", record.overshoot, again.overshoot)
    report.same("coverage caution", record.coverage_caution, again.coverage_caution)
    detail, found = record.coverage_detail, record.detail(coverage)
    if detail is None or found is None:
        report.same("coverage detail", detail is None, found is None)
        return
    for name in ("random_spread", "heads_only", "poles_only"):
        report.number(f"coverage {name}", getattr(detail, name), getattr(found, name), SHARE_SCALE)
    report.same("explained heads by layer", detail.heads_kept, found.heads_kept)
    report.same("minimal set", detail.minimal_units, found.minimal_units, warn=True)
    report.number("minimal set's coverage", detail.minimal_coverage, found.minimal_coverage,
                  SHARE_SCALE, warn=detail.minimal_units != found.minimal_units)  # fmt: skip


def _figures(report, record, runner, means, result, coverage, groups) -> None:
    """The figures' data, measured again (:mod:`openinterpretability.explain_figure_data`)."""
    from types import SimpleNamespace

    from openinterpretability.explain_figure_data import figure_data

    blocked = SimpleNamespace(blocked_units=[{"target": b} for b in record.inputs.blocked])
    want, got = record.figures, figure_data(runner, means, result, coverage, blocked, groups)
    scale = _scale(record)
    report.same("prompt tokens", want.tokens, got.tokens)
    report.number("answer logit", want.logit, got.logit, abs(want.logit))
    report.number("net attribution", want.net, got.net, scale)
    edges = {(e["source"], e["target"]): e["weight"] for e in got.edges}
    for e in want.edges:
        weight = edges.get((e["source"], e["target"]), 0.0)
        report.number(f"edge {e['source']} → {e['target']}", e["weight"], weight, scale)
    for label, values in want.reading.items():
        again = got.reading.get(label)
        if again is None or len(again) != len(values):
            report.same(f"{label} reading", len(values), None if again is None else len(again))
            continue
        peak = max(map(abs, values), default=1.0) or 1.0
        for i, (a, b) in enumerate(zip(values, again, strict=True)):
            report.number(f"{label} reading at token {i}", a, b, peak)
    if want.ablation and got.ablation:
        report.same("top token of the ablation table", want.ablation[0]["token"],
                    got.ablation[0]["token"])  # fmt: skip
    after = {row["token"]: row for row in got.ablation}
    for row in want.ablation:
        found = after.get(row["token"], {})
        for side in ("before", "after"):
            report.number(f"{row['token']!r} {side} ablation", row[side], found.get(side),
                          SHARE_SCALE)  # fmt: skip
    report.same("random draws", len(want.random_draws), len(got.random_draws))
    for i, (a, b) in enumerate(zip(want.random_draws, got.random_draws, strict=False)):
        report.number(f"random draw {i}", a, b, SHARE_SCALE)


NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+%?|(?<![\w.])\d+%")


def unquoted_numbers(record: ExplanationRecord) -> list[str]:
    """The decimals and percentages in the narration that no number in the record gives."""
    texts = [s.get("text", "") for s in record.narration.sections] + [record.narration.paragraph]
    values = _values(record.model_dump(mode="json", exclude={"narration"}))
    missing = []
    for text in texts:
        for quoted in NUMBER.findall(text):
            if not _found(quoted, values) and quoted not in missing:
                missing.append(quoted)
    return missing


def _values(data) -> list[float]:
    """Every number in ``data``, a record's JSON, and in each string of it."""
    if isinstance(data, bool):
        return []
    if isinstance(data, int | float):
        return [float(data)]
    if isinstance(data, str):
        return [float(x.replace("−", "-").rstrip("%")) for x in NUMBER.findall(data)]
    items = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
    return [v for item in items for v in _values(item)]


def _found(quoted: str, values: list[float]) -> bool:
    """True when some value rounds to ``quoted`` at its precision, as a fraction or as itself."""
    text = quoted.replace("−", "-")
    percent = text.endswith("%")
    number = float(text.rstrip("%"))
    places = len(text.rstrip("%").partition(".")[2])
    half = 0.5 * 10**-places + 1e-9
    for v in values:
        for candidate in (v * 100, v) if percent else (v,):
            if abs(abs(candidate) - abs(number)) <= half:
                return True
    return False


def is_explanation(path: Path) -> bool:
    """True when ``path`` is an explanation record, under ``wiki/explanations/``."""
    from openinterpretability.core.explanation_records import EXPLANATIONS_WIKI

    return EXPLANATIONS_WIKI in Path(path).parts[-4:-3]


def check_path(reproducer, path: Path) -> tuple[str, list[str], list[str]]:
    """(status, mismatches, warnings) for the explanation record at ``path``.

    ``reproducer`` is ``cli_reproduce._Reproducer``: its allow-list, corpora, runners and
    the records new since the base ref. A record from before #4793 has no inputs, so it is
    skipped. A new one fails.
    """
    from pydantic import ValidationError

    from openinterpretability.explanation_registry import registry_mismatches
    from openinterpretability.half_precision import reproduction_dtype

    try:
        record = ExplanationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return "FAILED", ["invalid explanation record"], []
    why = not_reproducible(record)
    if why and path.resolve() in reproducer.new:
        return "FAILED", [f"a new explanation record must be reproducible: {why}"], []
    if why:
        return f"NOT REPRODUCIBLE: {why}", [], []
    if path.resolve() in reproducer.new:  # its sets come from its author (#5173)
        wrong = registry_mismatches(record, Path(path).resolve().parents[3])
        if wrong:
            return "FAILED", wrong, []
    env = record.inputs.environment
    reason = reproducer.model_refusal(record.model_name, env.model_revision)
    if reason:
        return "FAILED", [reason], []
    try:
        texts = means_corpus(record, reproducer.scan_corpora)
    except ValueError as exc:
        return "FAILED", [f"its means corpus can't be had: {exc}"], []
    dtype = reproduction_dtype(env.dtype)
    runner = reproducer.load(record.model_name, env.model_revision, dtype, None)
    tolerance = reproducer.tolerance.for_dtype(env.dtype)
    report = reproduce_explanation(record, runner, texts, tolerance, path)
    return ("FAILED" if report.mismatches else "REPRODUCED"), report.mismatches, report.warnings


def inputs_of(runner, ask, result, planned, corpus: tuple, mean_tokens: int):
    """What ``reproduce`` needs to measure an explanation again, as ``explain`` stores it.

    ``ask`` is ``cli_explain.Ask``, for ``top`` and ``edges``. ``corpus`` is the means
    corpus's ``(SHA-256, CorpusSource or None)``.
    """
    from openinterpretability.core.explanation_records import ExplanationInputs
    from openinterpretability.explain import MEANS_VERSION, WINDOW
    from openinterpretability.explain_coverage import DRAWS, SEED
    from openinterpretability.harness.environment import run_environment

    explained = [_label(u) for u, _, _, done in result.order if done]
    return ExplanationInputs(
        environment=run_environment(runner), token_id=result.token_id,
        means_corpus_sha256=corpus[0], means_corpus_source=corpus[1], means_tokens=mean_tokens,
        means_window=WINDOW, means_version=MEANS_VERSION, top=ask.top, edges=ask.edges,
        explained=explained, blocked=[b["target"] for b in planned.blocked_units],
        random_seed=SEED, random_draws=DRAWS,
    )  # fmt: skip
