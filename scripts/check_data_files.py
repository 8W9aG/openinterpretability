"""Fail on a JSON, JSONL, notebook or YAML file that does not parse or breaks a style rule.

JSON, JSONL and notebooks must be UTF-8, parse, have no duplicate keys and end with a
newline. A JSONL file has one JSON value on each line and no blank lines. YAML files
and the YAML front matter of Markdown files must pass yamllint, with the rules in
`.yamllint.yml`. Front matter has no line limit, because a skill's description is one line.

The records under `wiki/` are exempt: `open-interpretability check` validates each one against its
schema, and they are write-once.

Run with `make lint`. CI runs it in the lint step.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

from yamllint import linter
from yamllint.config import YamlLintConfig

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / ".yamllint.yml"

JSON_PATTERNS = ("*.json", "*.ipynb")
JSONL_PATTERNS = ("*.jsonl",)
YAML_PATTERNS = ("*.yml", "*.yaml")
MARKDOWN_PATTERNS = ("*.md",)

EXEMPT = (
    "wiki/*",  # generated records and pages, validated by `open-interpretability check`
    ".claude/skills/asd-ste100/*",  # vendored as is from danyuchn/asd-ste100-skill
)

FRONT_MATTER = re.compile(r"---\n(.*?\n)---\n", re.DOTALL)


def tracked_files() -> list[str]:
    """Every tracked or new file that has a pattern in this module and is not exempt."""
    patterns = JSON_PATTERNS + JSONL_PATTERNS + YAML_PATTERNS + MARKDOWN_PATTERNS
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *patterns],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    names = [f for f in out.splitlines() if not any(fnmatch(f, p) for p in EXEMPT)]
    return [f for f in names if (REPO / f).exists()]


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Make a dict from ``pairs``, as `json` does. Raise ValueError on a duplicate key.

    The `json` module keeps the last value of a duplicate key and gives no warning.
    """
    duplicates = [key for key, count in Counter(key for key, _ in pairs).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate key {', '.join(map(repr, duplicates))}")
    return dict(pairs)


def json_problems(name: str, raw: bytes, lines: bool) -> list[str]:
    """One line for each problem in the JSON text ``raw``, or in each line if ``lines``."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        return [f"{name}: not UTF-8: {error}"]
    problems = [] if not raw or raw.endswith(b"\n") else [f"{name}: no newline at the end"]
    chunks = enumerate(text.splitlines(), 1) if lines else [(1, text)]
    for number, chunk in chunks:
        where = f"{name}:{number}" if lines else name
        if lines and not chunk.strip():
            problems.append(f"{where}: blank line")
            continue
        try:
            json.loads(chunk, object_pairs_hook=reject_duplicate_keys)
        except ValueError as error:
            problems.append(f"{where}: {error}")
    return problems


def yaml_problems(name: str, text: str, config: YamlLintConfig, first_line: int = 1) -> list[str]:
    """One line for each yamllint finding in ``text``.

    ``first_line`` is the line of the file where ``text`` starts, for front matter.
    """
    return [
        f"{name}:{p.line + first_line - 1}:{p.column}: {p.desc} ({p.rule})"
        for p in linter.run(text, config)
    ]


def file_problems(name: str, file_config: YamlLintConfig, matter: YamlLintConfig) -> list[str]:
    """One line for each problem in the file ``name``, by the rules for its type."""
    path = REPO / name
    if any(fnmatch(name, p) for p in JSON_PATTERNS):
        return json_problems(name, path.read_bytes(), lines=False)
    if any(fnmatch(name, p) for p in JSONL_PATTERNS):
        return json_problems(name, path.read_bytes(), lines=True)
    text = path.read_text(encoding="utf-8")
    if any(fnmatch(name, p) for p in YAML_PATTERNS):
        return yaml_problems(name, text, file_config)
    match = FRONT_MATTER.match(text)
    return yaml_problems(name, match[1], matter, first_line=2) if match else []


def yaml_configs() -> tuple[YamlLintConfig, YamlLintConfig]:
    """The yamllint rules for YAML files, and the same rules without a line limit."""
    file_config = YamlLintConfig(file=str(CONFIG))
    matter = YamlLintConfig(f"extends: {CONFIG}\nrules:\n  line-length: disable\n")
    return file_config, matter


def main() -> int:
    """Print every problem; return 1 if there is one."""
    file_config, matter = yaml_configs()
    problems = [p for name in tracked_files() for p in file_problems(name, file_config, matter)]
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
