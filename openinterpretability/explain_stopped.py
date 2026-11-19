"""The stopped units of an explained prompt, ranked by the coverage they would add (#4951).

A scan stops on a dead or default-on unit, and a studied unit can have no follow-up. Such a
unit is *blocked*: the experiment plan can't raise it (:mod:`openinterpretability.explain_plan`).
:func:`rank` measures each blocked unit's coverage gain, as
:func:`openinterpretability.explain_coverage.add_gains` does for a planned unit. It gives
each its reason: ``default_on``, ``dead`` or ``no follow-up``.

The first :data:`~openinterpretability.inverse_claims.INVERSE_CAP` default-on neuron poles
by gain get an inverse experiment (:mod:`openinterpretability.inverse_claims`). Each comes
with the ``audit --inverse`` command that starts it. A dead unit has no claim to make, so
it is listed, not studied.
"""

from __future__ import annotations

from pathlib import Path

POOL = 60  # blocked units measured for their gain, in order of their share of attribution


def outcomes(wiki_dir: Path) -> dict[str, str]:
    """Each stopped unit's slug and its newest scan outcome: ``default_on`` or ``dead``."""
    from openinterpretability.core.draws import load_draws, load_stops

    out = {}
    for d in sorted((*load_draws(wiki_dir), *load_stops(wiki_dir)), key=lambda d: d.timestamp):
        out[d.identifier.slug] = d.outcome
    return out


def rank(runner, prompt: str, means, token: int, position: int, order, planned, wiki_dir: Path,
         model: str) -> list[dict]:  # fmt: skip
    """Each blocked unit's ``{target, unit, reason, share, gain}``, highest gain first."""
    from openinterpretability.core.answer_records import parse_unit
    from openinterpretability.explain import _slug
    from openinterpretability.explain_coverage import _log_prob, _share, keep_hooks

    found = outcomes(Path(wiki_dir))
    keep = [unit for unit, _, _, explained in order if explained]
    clean = _log_prob(runner, prompt, token, position)
    ablated = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, []))

    def coverage(units) -> float | None:
        hooks = keep_hooks(runner, means, units)
        return _share(clean, ablated, _log_prob(runner, prompt, token, position, hooks))

    now = coverage(keep)
    rows = []
    for b in sorted(planned.blocked_units, key=lambda b: -b["share"])[:POOL]:
        unit = parse_unit(b["target"])
        reason = found.get(_slug(model, unit), "no follow-up")
        after = coverage([*keep, unit])
        gain = None if after is None or now is None else round(after - now, 6)
        rows.append({"target": b["target"], "unit": unit, "reason": reason,
                     "share": b["share"], "gain": gain})  # fmt: skip
    return sorted(rows, key=lambda r: (r["gain"] is None, -(r["gain"] or 0.0)))


def inverse_experiments(rows: list[dict], model: str, cap: int | None = None) -> list[dict]:
    """The first ``cap`` default-on neuron poles by gain, with the command that audits each."""
    from openinterpretability.inverse_claims import INVERSE_CAP

    picks = [r for r in rows if r["reason"] == "default_on" and r["unit"][2] != "head"]
    out = []
    for r in picks[: INVERSE_CAP if cap is None else cap]:
        layer, index, pole = r["unit"]
        command = (f"open-interpretability audit --model {model} --layer {layer} "
                   f"--neuron {index} --pole {pole} --inverse --selection attribution "
                   f"--evidence-out exchange/<id>/ev.json")  # fmt: skip
        out.append({**r, "command": command})
    return out


def lines(rows: list[dict], model: str, top: int = 10) -> list[str]:
    """How ``explain`` prints the stopped units, and the inverse experiments it plans."""
    if not rows:
        return []
    out = [f"Stopped units, by the coverage they would add ({len(rows)} measured):"]
    for r in rows[:top]:
        gain = "gain not measured" if r["gain"] is None else f"{r['gain']:+.1%} coverage"
        out.append(f"  {r['target']:<14} {gain:<18} {r['reason']}")
    for e in inverse_experiments(rows, model):
        out.append(f"  inverse experiment for {e['target']}:\n    $ {e['command']}")
    return out
