"""Unit statuses on the site, best first, and their Tailwind classes.

The layer map, the pieces bars and the status badges all read this table (templates
through the ``statuses`` global). CONTESTED is a standing
(:mod:`openinterpretability.builder.standing`); the rest are ``VerdictStatus`` values
(:mod:`openinterpretability.core.schemas`), except STOPPED. STOPPED marks a neuron whose scans all
stopped, dead or default-on, so that it has no experiment
(:class:`openinterpretability.builder.tree.StoppedNeuron`). It comes last, so any verdict in the
same cell wins.
"""

from openinterpretability.builder.palette import STATUS_HEX, badge_text, token

STOPPED = "STOPPED"  # a neuron whose scans all stopped


# Status: (map and bar fill, badge). The colours are builder/palette.py's.
def _badge(status: str) -> str:
    """A filled badge, with text that stays readable on the fill."""
    return f"bg-{token(status)} text-[{badge_text(STATUS_HEX[status])}]"


_CLASSES = {
    **{s: (f"fill-{token(s)}", _badge(s)) for s in STATUS_HEX},
    STOPPED: (
        "fill-slate-400 dark:fill-slate-500",
        "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    ),
}

# A neuron on the layer map shows its best pole's status, in this order.
MAP_STATUSES = tuple(_CLASSES)
STATUS_FILL = {s: fill for s, (fill, _) in _CLASSES.items()}
STATUS_BADGE = {s: badge for s, (_, badge) in _CLASSES.items()}

# A status grid's cell (#5797, #5836). On a layer page a cell is one neuron, CELL_HEIGHT px
# high in its status's colour. On a model's grid a cell is a block of neurons. It stacks its
# studied neurons by status, NEURON_PX px each, with SEGMENT_GAP px between two statuses. So
# a cell's height is how many it holds.
CELL_HEIGHT = 10
NEURON_PX = 3
SEGMENT_GAP = 1
