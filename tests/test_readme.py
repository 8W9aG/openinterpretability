"""The README says how to do two things and nothing more (#5843)."""

import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_the_readme_has_two_tasks_and_a_links_section():
    text = README.read_text()
    assert re.findall(r"^## (.+)$", text, re.M) == [
        "Before you start", "Run an experiment", "Explain an answer", "More",
    ]  # fmt: skip
    assert len(text.splitlines()) <= 45


def test_the_readme_lists_no_option():
    # Options belong in docs/experiments.md, so the project's commands run with their defaults.
    # The fork step's --clone is GitHub's flag, not one of ours.
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(), re.S)
    ours = [
        line
        for b in blocks
        for line in b.splitlines()
        if "open-interpretability" in line or "claude" in line
    ]
    assert len(blocks) == 2 and len(ours) == 2
    assert not [line for line in ours if " --" in line]
