"""Unit statuses on the site, best first, and their Tailwind classes.

The layer map, the pieces bars and the status badges all read this table (templates
through the ``statuses`` global). CONTESTED is a standing (:mod:`openinterp.builder.standing`);
the rest are ``VerdictStatus`` values (:mod:`openinterp.core.schemas`), except STOPPED.
STOPPED marks a neuron whose scans all stopped, dead or default-on, so that it has no
experiment (:class:`openinterp.builder.tree.StoppedNeuron`). It comes last, so any verdict
in the same cell wins.
"""

from openinterp.builder.palette import STATUS_HEX, token

STOPPED = "STOPPED"  # a neuron whose scans all stopped


# Status: (map and bar fill, badge). The colours are builder/palette.py's.
def _badge(status: str) -> str:
    colour = token(status)
    return f"bg-{colour}/15 text-slate-900 ring-1 ring-inset ring-{colour} dark:text-slate-100"


_CLASSES = {
    **{s: (f"fill-{token(s)}", _badge(s)) for s in STATUS_HEX},
    STOPPED: (
        "fill-slate-400 dark:fill-slate-500",
        "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    ),
}
# Statuses whose map and bar cells are hatched as well as coloured, so they don't rely on
# colour alone (``status_cell`` in ``_macros.html``).
HATCHED = ("REJECTED",)

# A neuron on the layer map shows its best pole's status, in this order.
MAP_STATUSES = tuple(_CLASSES)
STATUS_FILL = {s: fill for s, (fill, _) in _CLASSES.items()}
STATUS_BADGE = {s: badge for s, (_, badge) in _CLASSES.items()}
