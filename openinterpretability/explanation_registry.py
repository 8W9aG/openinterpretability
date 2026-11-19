"""A new explanation record must agree with the registry it was written against (#5173).

``reproduce`` measures an explanation record again with the record's own explained and
blocked unit sets (:mod:`openinterpretability.reproduce_explanation`). That keeps an old
record reproducible after the registry changes. But a new record's sets come from the
person who wrote it, so they must also agree with the registry in the checkout:

- each explained unit is VERIFIED, or is a member of a VERIFIED superposition or circuit,
  as :func:`openinterpretability.explain.explain` decides;
- each blocked unit is neither explained nor unstudied, as the experiment plan decides
  (:mod:`openinterpretability.explain_plan`);
- each listed unit's status is the registry's, and its explanation is the registry's
  explanation when the unit is VERIFIED, and none otherwise.

:func:`registry_mismatches` returns each disagreement. ``reproduce`` runs it only on the
records that a change adds, before it loads a model.
"""

from __future__ import annotations

from pathlib import Path

UNSTUDIED = ("unstudied", None, None)


def registry_mismatches(record, wiki_dir: Path, groups=None) -> list[str]:
    """Each way that ``record`` disagrees with the registry under ``wiki_dir``."""
    from openinterpretability import explain as ex
    from openinterpretability import explain_groups
    from openinterpretability.planner import load_states

    model = record.model_name
    labels = ex.registry_labels(wiki_dir, model, load_states(wiki_dir))
    circuit_labels = explain_groups.circuit_labels(wiki_dir, model)
    if groups is None:
        groups = explain_groups.superpositions(model) + explain_groups.circuits(model)

    def group_label(g) -> tuple[str, str | None]:
        if g.kind == "superposition":
            return labels.get(g.slug, UNSTUDIED)[:2]
        return circuit_labels.get(g.slug, ("unstudied", None))

    covered = {u for g in groups if group_label(g)[0] == "VERIFIED" for u in g.units}
    out = _sets(record, model, labels, covered)
    by_name = {g.name: g for g in groups}
    for u in record.units:
        if u.kind in ("pole", "head"):
            status, text = _status(model, labels, u.unit)
        elif u.unit in by_name:
            status, text = group_label(by_name[u.unit])
        else:
            out.append(f"{u.unit}: no such {u.kind} in the repository")
            continue
        shown = text if status == "VERIFIED" else None
        if (u.status, u.explanation) != (status, shown):
            out.append(f"{u.unit}: the record says {u.status}, the registry says {status}")
    return out


def _sets(record, model: str, labels: dict, covered: set) -> list[str]:
    """The explained and blocked sets, against the registry."""
    from openinterpretability.core.answer_records import parse_unit

    out = []
    for name, labels_in, allowed in (("explained", record.inputs.explained, True),
                                     ("blocked", record.inputs.blocked, False)):  # fmt: skip
        for label in labels_in:
            try:
                unit = parse_unit(label)
            except ValueError:
                out.append(f"{label}: not a unit label, in the {name} set")
                continue
            status, _ = _status(model, labels, label)
            explained = status == "VERIFIED" or unit in covered
            if explained != allowed or (not allowed and status == "unstudied"):
                out.append(f"{label} is in the {name} set, but the registry has it {status}")
    return out


def _status(model: str, labels: dict, label: str) -> tuple[str, str | None]:
    """A pole's or head's registry status and explanation, from its label."""
    from openinterpretability import explain as ex
    from openinterpretability.core.answer_records import parse_unit

    try:
        unit = parse_unit(label)
    except ValueError:
        return "unknown", None
    return labels.get(ex._slug(model, unit), UNSTUDIED)[:2]
