"""The STE linter skips a paper's reference list (scripts/ste_prose.py, issue #3137)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ste_prose", Path(__file__).parents[1] / "scripts" / "ste_prose.py"
)
ste_prose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ste_prose)


def test_only_the_reference_section_is_blanked_and_lines_are_kept():
    text = "# T\n\nProse.\n\n## References\n\n- A, B. (2020). *Long*.\n\n## Next\n\nMore.\n"
    out = ste_prose.without_references(text)
    assert out.count("\n") == text.count("\n")
    assert "Long" not in out and "Prose." in out and "More." in out
    assert "## References" in out and "## Next" in out
