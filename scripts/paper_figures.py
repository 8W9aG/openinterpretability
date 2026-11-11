"""Write each paper's media and the interactive page's data. It needs no model.

The source of each file is the results.jsonl of a paper in papers/.

Run it after a change to the results. ``--check`` writes nothing, and fails if a file differs.
"""

import argparse
import json
import sys
from pathlib import Path

import figures_default_on
import figures_default_on_decoding
import figures_depth
import figures_explorer
import figures_spd_input
import figures_syntax
import figures_why_rest
import figures_why_rest_equal
import figures_why_rest_push

PAPERS = Path(__file__).parents[1] / "papers"


def _rows(paper: str, name: str = "results.jsonl") -> list[dict]:
    """Return the rows of one paper's results, or of another results file of the paper."""
    with (PAPERS / paper / name).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def _other_models(paper: str) -> dict[str, list[dict]]:
    """Return the rows of each results-<model>.jsonl of a paper, by the model's short name."""
    found = {}
    for path in sorted((PAPERS / paper).glob("results-*.jsonl")):
        rows = _rows(paper, path.name)
        model = next(r["model"] for r in rows if r["kind"] == "sample")
        found[model.split("/")[-1]] = rows
    return found


def build_all() -> dict[str, str]:
    """Return each figure's SVG text by its path under papers/."""
    poles = _rows("default-on-poles")
    resting = _rows("why-neurons-rest")
    dip = next(row["dip"] for row in poles if row.get("part") == "band_population")
    papers = {
        "default-on-poles": {
            "rest.svg": figures_default_on.rest(poles, dip),
            "clamps.svg": figures_default_on.clamps(),
            "cost.svg": figures_default_on.cost(poles),
            "kl-ratio.svg": figures_default_on.kl_ratio(poles),
            "tails.svg": figures_default_on_decoding.tails(poles),
            "decoding.svg": figures_default_on_decoding.decoding(poles),
            "ridges.svg": figures_depth.ridges(poles, dip),
        },
        "spd-input-projections": figures_spd_input.build(_rows("spd-input-projections")),
        "syntax-positive-controls": figures_syntax.build(
            _rows("syntax-positive-controls"), _other_models("syntax-positive-controls")
        ),
        "why-neurons-rest": {
            "surface.svg": figures_depth.surface(resting, dip),
            **figures_why_rest.build(resting, dip),
            "push.svg": figures_why_rest_push.push(resting),
            "equal-rest.svg": figures_why_rest_equal.equal_rest(resting),
        },
    }
    return {
        f"{paper}/media/{name}": figure.render()
        for paper, figures in papers.items()
        for name, figure in figures.items()
    }


def generated() -> dict[str, str]:
    """Return every generated file's text by its path under papers/."""
    poles = _rows("default-on-poles")
    dip = next(row["dip"] for row in poles if row.get("part") == "band_population")
    data = figures_explorer.data_js(poles, _rows("why-neurons-rest"), dip)
    return {**build_all(), "below-the-dip/data.js": data}


def main() -> int:
    """Write the figures, or compare them with the files. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if a file differs.")
    args = parser.parse_args()
    stale = []
    for name, text in generated().items():
        path = PAPERS / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text, encoding="utf-8")
    for name in stale:
        print(f"Stale figure: {name}. Run scripts/paper_figures.py.", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
