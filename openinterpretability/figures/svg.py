"""Draw the figures of the analyses in docs/analyses as SVG, with no plotting library.

An SVG file is text, so a diff shows each change, and a test can compare it with its data.
Each figure sets its own colours for light and dark themes. GitHub and editors show both.
"""

import html
import math
from collections.abc import Iterable, Sequence

WIDTH = 760
AXIS_ROOM = 48
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#6b6a66",
    "neutral": "#8a8985",
    "wash": "#d9d8d2",
    "grid": "#e8e7e2",
    "axis": "#bdbcb5",
    "c1": "#2a78d6",
    "c2": "#eb6834",
    "c3": "#1baf7a",
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#a3a299",
    "neutral": "#8f8e86",
    "wash": "#3f3f3b",
    "grid": "#2c2c2a",
    "axis": "#55554f",
    "c1": "#3987e5",
    "c2": "#d95926",
    "c3": "#199e70",
}
RULES = """
.bg{fill:var(--surface)}
text{font-family:FONT_STACK;fill:var(--ink2);font-size:12px}
.title{fill:var(--ink);font-size:17px;font-weight:600}
.sub{font-size:13px}
.head{fill:var(--ink);font-size:13px;font-weight:600}
.ink{fill:var(--ink)}
.small{fill:var(--muted);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.dip{stroke:var(--neutral);stroke-width:1}
.curve{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.range{stroke-width:2;stroke-linecap:round}
.ring{stroke:var(--surface);stroke-width:2}
.open{fill:var(--surface);stroke-width:2}
.soft{opacity:.25}
.ridge path{vector-effect:non-scaling-stroke}
@media (prefers-reduced-motion: reduce){*{animation:none!important}}
.bg2{fill:var(--surface)}
.mesh{fill:none;stroke:var(--surface);stroke-width:.7;stroke-opacity:.8}
.contour{stroke:var(--ink);stroke-width:2;stroke-linecap:round}
.dipfloor{stroke:var(--ink);stroke-width:1.4;stroke-opacity:.75}
.fc1{fill:var(--c1)}.fc2{fill:var(--c2)}.fc3{fill:var(--c3)}
.fneutral{fill:var(--neutral)}.fwash{fill:var(--wash)}
.sc1{stroke:var(--c1)}.sc2{stroke:var(--c2)}.sc3{stroke:var(--c3)}
.sneutral{stroke:var(--neutral)}.swash{stroke:var(--wash)}
""".replace("FONT_STACK", FONT)


def _variables(colours: dict[str, str]) -> str:
    """Return the CSS variables for one theme."""
    return ";".join(f"--{name}:{value}" for name, value in colours.items())


def number(value: float, places: int = 2) -> str:
    """Return ``value`` as text with a true minus sign, and no minus sign on a zero."""
    text = f"{value:.{places}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def gelu(x: float) -> float:
    """Return the exact GELU of ``x``."""
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def relu(x: float) -> float:
    """Return the ReLU of ``x``."""
    return max(x, 0.0)


LEAKY_SLOPE = 0.1  # the slope of leaky ReLU below 0, large enough to differ from ReLU


def leaky_relu(x: float) -> float:
    """Return the leaky ReLU of ``x``, with slope ``LEAKY_SLOPE`` below 0."""
    return x if x > 0 else LEAKY_SLOPE * x


def softplus(x: float) -> float:
    """Return the softplus of ``x``: log(1 + e^x)."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def silu(x: float) -> float:
    """Return the SiLU of ``x``: x times the logistic sigmoid of x."""
    return x / (1 + math.exp(-x))


def normal(x: float, mean: float, sd: float) -> float:
    """Return the density of a normal distribution at ``x``."""
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def sample(start: float, stop: float, count: int = 161) -> list[float]:
    """Return ``count`` evenly spaced values from ``start`` to ``stop``."""
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


class Figure:
    """One SVG image: a surface, a finding as its title, and marks in pixel coordinates."""

    def __init__(self, title: Sequence[str], subtitle: Sequence[str], alt: str):
        """Add the text at the top. ``alt`` is the image's text for a reader."""
        self.bottom = 0.0
        self.motion = ""
        self.alt = alt
        self.marks: list[str] = []
        y = 30
        for line in title:
            self.text(20, y, line, "title")
            y += 23
        y -= 3
        for line in subtitle:
            self.text(20, y, line, "sub")
            y += 18
        self.body_top = y + 6

    def text(self, x: float, y: float, content: str, cls: str = "", anchor: str = "start") -> None:
        """Add one line of text."""
        style = f' class="{cls}"' if cls else ""
        end = "" if anchor == "start" else f' text-anchor="{anchor}"'
        self.marks.append(
            f'<text x="{x:.1f}" y="{y:.1f}"{style}{end}>{html.escape(content)}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        """Add one straight line."""
        self.marks.append(f'<path class="{cls}" d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}"/>')

    def path(self, points: Iterable[tuple[float, float]], cls: str, close: bool = False) -> None:
        """Add a line through ``points``, and close it to a shape if ``close`` is true."""
        steps = "L".join(f"{x:.1f} {y:.1f}" for x, y in points)
        self.marks.append(f'<path class="{cls}" d="M{steps}{"Z" if close else ""}"/>')

    def shaded(self, points: list[tuple[float, float]], cls: str, opacity: float) -> None:
        """Add a closed shape with its own strength of fill."""
        steps = "L".join(f"{x:.1f} {y:.1f}" for x, y in points)
        self.marks.append(f'<path class="{cls}" fill-opacity="{opacity:.2f}" d="M{steps}Z"/>')

    def dot(self, x: float, y: float, radius: float, cls: str) -> None:
        """Add one circle."""
        self.marks.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}"/>')

    def bar(self, x: float, y: float, width: float, height: float, cls: str) -> None:
        """Add one rectangle with slightly round corners. It may have a negative height."""
        top = min(y, y + height)
        self.marks.append(
            f'<rect class="{cls}" x="{x:.1f}" y="{top:.1f}" width="{max(width, 0.5):.1f}" '
            f'height="{abs(height):.1f}" rx="1.5"/>'
        )

    def legend(self, y: float, entries: Sequence[tuple[str, str]], x: float = 20) -> None:
        """Add a row of labelled dots. Each entry is a dot's class and its label."""
        for cls, label in entries:
            self.dot(x + 5, y - 4, 5, cls)
            self.text(x + 15, y, label)
            x += 26 + 6.6 * len(label)

    def render(self) -> str:
        """Return the SVG document. It is stable text, so reruns give identical files."""
        style = (
            f":root{{{_variables(LIGHT)}}}"
            f"@media (prefers-color-scheme: dark){{:root{{{_variables(DARK)}}}}}"
            f"{RULES}{self.motion}"
        )
        body = "\n".join(self.marks)
        height = round(self.bottom + AXIS_ROOM)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" role="img">\n'
            f"<title>{html.escape(self.alt)}</title>\n<style>{style}</style>\n"
            f'<rect class="bg" width="{WIDTH}" height="{height}" rx="10"/>\n{body}\n</svg>\n'
        )


class Panel:
    """A rectangle of a figure, with the scales that map data to its pixels."""

    def __init__(
        self,
        figure: Figure,
        box: tuple[float, float, float, float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ):
        """Keep the box as left, top, width and height, with the two data ranges."""
        self.figure = figure
        self.left, self.top, self.width, self.height = box
        self.x_range = x_range
        self.y_range = y_range
        figure.bottom = max(figure.bottom, self.top + self.height)

    def x(self, value: float) -> float:
        """Return the pixel of a data value on the x axis."""
        low, high = self.x_range
        return self.left + (value - low) / (high - low) * self.width

    def y(self, value: float) -> float:
        """Return the pixel of a data value on the y axis."""
        low, high = self.y_range
        return self.top + self.height - (value - low) / (high - low) * self.height

    def grid_x(self, ticks: Sequence[float], places: int = 1, thousands: bool = False) -> None:
        """Add a vertical hairline at each tick, with its value below the panel."""
        for tick in ticks:
            self.figure.line(self.x(tick), self.top, self.x(tick), self.top + self.height, "grid")
            text = f"{tick / 1000:.0f}k" if thousands and tick else number(tick, places)
            self.figure.text(self.x(tick), self.top + self.height + 15, text, "small", "middle")

    def grid_y(self, ticks: Sequence[float], places: int = 1, labels: bool = True) -> None:
        """Add a horizontal hairline at each tick, with its value left of the panel."""
        for tick in ticks:
            self.figure.line(self.left, self.y(tick), self.left + self.width, self.y(tick), "grid")
            if labels:
                self.figure.text(
                    self.left - 7, self.y(tick) + 4, number(tick, places), "small", "end"
                )

    def x_title(self, content: str) -> None:
        """Add the x axis's name below its tick labels."""
        centre = self.left + self.width / 2
        self.figure.text(centre, self.top + self.height + 31, content, "small", "middle")

    def head(self, content: str) -> None:
        """Add the panel's name above it."""
        self.figure.text(self.left, self.top - 8, content, "head")

    def curve(self, xs: Iterable[float], ys: Iterable[float], cls: str) -> None:
        """Add a line through the data points."""
        self.figure.path([(self.x(a), self.y(b)) for a, b in zip(xs, ys, strict=True)], cls)

    def vertical(self, value: float, cls: str) -> None:
        """Add a vertical line over the full height of the panel."""
        self.figure.line(self.x(value), self.top, self.x(value), self.top + self.height, cls)

    def horizontal(self, value: float, cls: str) -> None:
        """Add a horizontal line over the full width of the panel."""
        self.figure.line(self.left, self.y(value), self.left + self.width, self.y(value), cls)


def columns(
    figure: Figure,
    count: int,
    left: float,
    top: float,
    height: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    gap: float = 18,
    right: float = 24,
) -> list[Panel]:
    """Return ``count`` panels of equal width in one row, with the same two ranges."""
    width = (WIDTH - left - right - gap * (count - 1)) / count
    return [
        Panel(figure, (left + i * (width + gap), top, width, height), x_range, y_range)
        for i in range(count)
    ]
