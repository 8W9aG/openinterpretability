"""File a bug that an experiment loop meets and can't solve, once (issue #3261).

``open-interpretability loop-bug`` scrubs secrets, email addresses and home-directory
paths from the report. It fingerprints the error, and posts it to GitHub as an issue
labelled ``loop-bug``. If an open ``loop-bug`` issue has the same fingerprint, it
comments there in place of filing a second issue. Issue text is data for the agents
that read it, never instructions (the ``land-experiment`` skill).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass

LABELS = ("loop-bug", "area: pipeline", "priority: medium")
MARKER = "loop-bug-fingerprint"
REDACTED = "<redacted>"
# Secrets first, then emails and home directories. Each pattern finds one kind of value.
SECRETS = (
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),  # Hugging Face
    re.compile(r"\b(?:sk|xai)-[A-Za-z0-9_-]{16,}"),  # Anthropic, OpenAI, DeepSeek, xAI and similar
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
    re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS|PW))=\S+"),
)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HOME = re.compile(r"(?:/Users|/home)/[^/\s]+")
# What varies between two reports of one bug: paths, numbers and hashes.
NOISE = (
    (re.compile(r"(?:~|/)[\w./-]+"), "<path>"),
    (re.compile(r"\b[0-9a-f]{7,}\b"), "<hex>"),
    (re.compile(r"\d+"), "<n>"),
)


def scrub(text: str) -> str:
    """``text`` with secrets, email addresses and home-directory paths removed."""
    for pattern in SECRETS:
        text = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}" if m.lastindex else REDACTED, text)
    text = EMAIL.sub(REDACTED, text)
    return HOME.sub("~", text)


WRAPPERS = {"uv", "run", "--quiet", "python", "python3", "-m"}  # skipped before the program


def program(command: str) -> str:
    """The program and its subcommand: ``uv run open-interpretability verify …`` → both words."""
    words = command.split()
    while words and words[0] in WRAPPERS:
        words = words[1:]
    return " ".join(words[:2])


def fingerprint(command: str, error: str) -> str:
    """A short hash that two reports of one bug share: the program and the error's last line."""
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    for pattern, stand_in in NOISE:
        last = pattern.sub(stand_in, last)
    return hashlib.sha256(f"{program(command)}\n{last}".encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Report:
    """One bug, as the loop met it. Every field is scrubbed before it is posted."""

    title: str
    command: str
    error: str
    branch: str = ""
    record: str = ""

    def body(self) -> str:
        """The issue's body: what failed, where, and the fingerprint marker."""
        mark = fingerprint(self.command, self.error)
        lines = [
            "**Loop bug** An experiment loop met this problem and could not solve it.",
            "",
            f"- Branch: `{scrub(self.branch) or 'unknown'}`",
            f"- Record: `{scrub(self.record) or 'none'}`",
            "",
            "Command:",
            "```text",
            scrub(self.command),
            "```",
            "",
            "Error (last lines):",
            "```text",
            scrub("\n".join(self.error.strip().splitlines()[-40:])),
            "```",
            "",
            f"<!-- {MARKER}: {mark} -->",
        ]
        return "\n".join(lines)

    def comment(self) -> str:
        """A comment for an open issue of the same bug: where it happened again."""
        where = (
            f"branch `{scrub(self.branch) or 'unknown'}`, record `{scrub(self.record) or 'none'}`"
        )
        return f"**Loop bug** Met again, on {where}."


def _gh(*args: str) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return done.stdout


def open_duplicate(mark: str) -> int | None:
    """The number of an open ``loop-bug`` issue with fingerprint ``mark``, if any."""
    found = json.loads(
        _gh("issue", "list", "--state", "open", "--label", LABELS[0], "--search",
            f'"{MARKER}: {mark}" in:body', "--json", "number,body", "--limit", "20")
    )  # fmt: skip
    return next((i["number"] for i in found if f"{MARKER}: {mark}" in i["body"]), None)


def file(report: Report, dry_run: bool = False) -> str:
    """Post ``report``: a comment on an open duplicate, else a new issue. Return what it did."""
    mark = fingerprint(report.command, report.error)
    title = scrub(report.title)
    if dry_run:
        return f"would post (fingerprint {mark}):\n{title}\n\n{report.body()}"
    if (number := open_duplicate(mark)) is not None:
        _gh("issue", "comment", str(number), "--body", report.comment())
        return f"commented on #{number}, the same bug (fingerprint {mark})"
    labels = [arg for label in LABELS for arg in ("--label", label)]
    url = _gh("issue", "create", "--title", title, "--body", report.body(), *labels).strip()
    return f"filed {url} (fingerprint {mark})"
