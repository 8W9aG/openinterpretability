"""The explain skill's steps run in an order that works (.claude/skills/explain, #5964)."""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "explain" / "SKILL.md"


def test_the_explanation_is_recorded_before_the_answer_experiment_and_narrated_after():
    # `answers propose-prompt` needs a recorded explanation of the prompt, and the narrator
    # reads the answer experiment. So: record, then the answer experiment, then narrate.
    text = SKILL.read_text(encoding="utf-8")
    record = text.index("--target <target> --record --yes")
    answer = text.index("**Run the answer experiment.**")
    narrate = text.index("--level long --record --yes")
    assert record < answer < narrate


def test_the_steps_are_numbered_in_order():
    import re

    numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", SKILL.read_text(), re.M)]
    assert numbers == list(range(1, len(numbers) + 1))
