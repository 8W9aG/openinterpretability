"""Changelog fragments (scripts/changelog.py): the lint check and the release step."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "changelog", Path(__file__).parents[1] / "scripts" / "changelog.py"
)
changelog = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(changelog)

HEAD = "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n"
OLD = "## [0.0.1] - 2026-01-01\n\n### Added\n\n- **First.**\n"


def _setup(tmp_path, unreleased="", **fragments):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(HEAD + unreleased + OLD)
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    (folder / "README.md").write_text("How to add a fragment.\n")
    for name, text in fragments.items():
        (folder / name.replace("__", ".")).write_text(text)
    return log, folder


def test_valid_fragments_pass(tmp_path):
    log, folder = _setup(tmp_path, **{"3490__changed__md": "- **A change** (#3490).\n"})
    assert changelog.problems(log, folder) == []


def test_bad_names_empty_fragments_and_unreleased_entries_fail(tmp_path):
    log, folder = _setup(
        tmp_path,
        unreleased="### Added\n\n- **Edited by hand.**\n\n",
        **{"3490__tweaked__md": "- **x**\n", "12__fixed__md": "no bullet\n"},
    )
    found = changelog.problems(log, folder)
    assert len(found) == 3
    assert any("3490.tweaked.md" in p for p in found)
    assert any("12.fixed.md" in p and "bullet" in p for p in found)
    assert any("Unreleased section must be empty" in p for p in found)


def test_collect_groups_orders_dates_and_deletes(tmp_path):
    log, folder = _setup(
        tmp_path,
        **{
            "9__security__md": "- **Pinned.**\n",
            "2__added__md": "- **Second.**\n",
            "1__added__md": "- **First new.**\n",
            "3__removed__md": "- **Gone.**\n",
        },
    )
    assert (
        changelog.collect(
            "0.0.2", "2026-11-14", "Ships with protocol version **0.8**.", log, folder
        )
        == 4
    )
    text = log.read_text()
    assert text.startswith(
        HEAD + "## [0.0.2] - 2026-11-14\n\nShips with protocol version **0.8**.\n"
    )
    new = text[len(HEAD) : text.index("## [0.0.1]")]
    assert new.index("### Added") < new.index("### Removed") < new.index("### Security")
    assert new.index("First new.") < new.index("Second.")
    assert [p.name for p in folder.iterdir()] == ["README.md"]
    assert changelog.problems(log, folder) == []  # Unreleased is empty again
    assert OLD in text
