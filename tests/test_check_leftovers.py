"""The template's files hold no word of this project, past the ratchet's baseline (#6631)."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
leftovers = importlib.import_module("check_leftovers")


def test_every_template_path_exists():
    missing = [p for p in leftovers.lines(leftovers.PATHS) if not (ROOT / p).exists()]
    assert not missing


def test_the_repository_matches_its_baseline():
    over, lower = leftovers.compare(leftovers.counts(), leftovers.read_baseline())
    assert not over, over  # a new leftover word in a template file
    assert not lower, f"lower the baseline: python3 scripts/check_leftovers.py --update {lower}"


def project(tmp_path: Path) -> Path:
    """A repository with two template files, one of them with two leftover words."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "template_paths.txt").write_text("# the template\ndocs\nREADME.md\n")
    (scripts / "leftover_terms.txt").write_text("(?i)\\bwidgets?\\b\n")
    (scripts / "leftovers_baseline.txt").write_text("9 a file that names widgets\n")  # exempt
    (tmp_path / "docs" / "__pycache__").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("A widget. Two Widgets.")
    (tmp_path / "docs" / "logo.png").write_text("widget")  # binary, never read
    (tmp_path / "docs" / "__pycache__" / "a.md").write_text("widget")
    (tmp_path / "README.md").write_text("Nothing here.")
    return tmp_path


def test_the_counts_read_the_listed_text_files_only(tmp_path):
    assert leftovers.counts(project(tmp_path)) == {"docs/a.md": 2}


def test_the_check_fails_over_the_baseline_and_asks_to_lower_it_under(tmp_path, capsys):
    root = project(tmp_path)
    assert leftovers.main(["--update"], root) == 0
    baseline = root / "scripts" / "leftovers_baseline.txt"
    assert leftovers.read_baseline(baseline) == {"docs/a.md": 2}
    assert baseline.read_text().startswith("# The leftover words")
    assert leftovers.main([], root) == 0
    (root / "docs" / "a.md").write_text("A widget, three widgets, one more widget.")
    assert leftovers.main([], root) == 1
    assert "docs/a.md: 3 hits, the baseline allows 2" in capsys.readouterr().out
    (root / "docs" / "a.md").write_text("No word left.")
    assert leftovers.main([], root) == 1
    assert "--update" in capsys.readouterr().out
