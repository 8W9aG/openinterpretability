"""Unit statuses on the site, best first, and their Tailwind classes.

The layer map, the pieces bars and the status badges all read this table (templates
through the ``statuses`` global). CONTESTED is a standing (:mod:`openinterp.builder.standing`);
the rest are ``VerdictStatus`` values (:mod:`openinterp.core.schemas`), except STOPPED.
STOPPED marks a neuron whose scans all stopped, dead or default-on, so that it has no
experiment (:class:`openinterp.builder.tree.StoppedNeuron`). It comes last, so any verdict
in the same cell wins.
"""

STOPPED = "STOPPED"  # a neuron whose scans all stopped

# Status: (map and bar fill, badge).
_CLASSES = {
    "VERIFIED": (
        "fill-emerald-500",
        "bg-emerald-100 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100",
    ),
    "CONTESTED": ("fill-sky-500", "bg-sky-100 text-sky-900 dark:bg-sky-900 dark:text-sky-100"),
    "NEEDS_REVISION": (
        "fill-amber-500",
        "bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100",
    ),
    "REJECTED": ("fill-rose-500", "bg-rose-100 text-rose-900 dark:bg-rose-900 dark:text-rose-100"),
    STOPPED: (
        "fill-slate-400 dark:fill-slate-500",
        "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    ),
}

# A neuron on the layer map shows its best pole's status, in this order.
MAP_STATUSES = tuple(_CLASSES)
STATUS_FILL = {s: fill for s, (fill, _) in _CLASSES.items()}
STATUS_BADGE = {s: badge for s, (_, badge) in _CLASSES.items()}
