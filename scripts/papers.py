"""Each paper's status and citation file, checked in lint (issue #3150).

A paper is a **preprint** while work on it is left, and it says which issues hold it
open. It is **published** when no work is left. After that its text and results don't
change. Later corrections go in a ``## Addenda`` section at the end. Each paper has a
``CITATION.cff`` (CFF 1.2.0) whose ``version`` agrees with its status line. A published
paper's file holds the SHA-256 of its text above the addenda and of its results files.
A change to them then fails the check.

A preprint's line must name only open issues (#5395). ``check --closing`` takes the issues a
pull request closes, and fails if a line still names one. ``check --online`` asks GitHub for
the state of every issue on every line, and fails on a closed one. Plain ``check`` stays
offline, for ``make lint``.

    uv run python scripts/papers.py check
    uv run python scripts/papers.py check --closing 4250,5394   # CI on a PR (#5395)
    uv run python scripts/papers.py check --online              # the nightly run (#5395)
    uv run python scripts/papers.py new <slug> --work "#N (what is left)"
    uv run python scripts/papers.py publish <slug>

A published paper goes to the alphaXiv AI Track (#5259). alphaXiv has no upload API, so
Muse (``docs/muse.md``) makes the upload from the project account. ``submit --issue`` opens
the issue, labelled ``muse``, that asks it to. ``submit``
writes the package: the PDF that ``make build`` prints, and ``submission.md`` with each field
of the form. After the upload, ``alphaxiv`` records the paper's URL in its ``CITATION.cff``,
outside the frozen hash.

    make build && uv run python scripts/papers.py submit <slug>
    uv run python scripts/papers.py submit <slug> --issue   # the issue for Muse (label muse)
    uv run python scripts/papers.py alphaxiv <slug> https://www.alphaxiv.org/abs/<id>
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.builder.research import (  # noqa: E402
    _ENTRY,
    ADDENDA,
    ALPHAXIV,
    ALPHAXIV_URL,
    STATUS,
    _read_paper,
    alphaxiv_url,
    paper_status,
)

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
GROUP = {"name": "Open Interpretability"}
SITE = "https://open-interpretability.com/research/"
REPO = "https://github.com/8W9aG/openinterpretability"
PREPRINT = "preprint"
PUBLISHED = "1.0"
DIST = ROOT / "dist"  # where make build writes the site, with each paper's PDF
SUBMISSIONS = ROOT / "build" / "alphaxiv"  # the packages that submit writes
LICENCE = "CC BY 4.0"  # the maintainer's choice on #5259 (decision 2)
AI_TRACK = "https://www.alphaxiv.org/ai-track"
MUSE_LABELS = ("muse", "area: housekeeping")  # Muse posts it; next_issue.py never takes it
CATEGORIES = ("cs.LG", "cs.AI")
DATASET = re.compile(r"https://huggingface\.co/datasets/[\w.\-]+/[\w.\-]+")
DISCLOSURE = (
    "AI agents (Claude models in Claude Code) ran the experiments and wrote the text. The "
    "paper's text names which models proposed, reviewed, wrote and narrated the work."
)


def entries(papers: Path = PAPERS) -> list[tuple[str, str]]:
    """Each listed paper's (date, slug), from ``papers/README.md``."""
    return [(d, s) for d, s in _ENTRY.findall((papers / "README.md").read_text(encoding="utf-8"))]


def title(text: str) -> str:
    """The paper's level-1 heading."""
    return next(line[2:].strip() for line in text.splitlines() if line.startswith("# "))


def frozen_hash(folder: Path) -> str:
    """SHA-256 of the text above the addenda and of each results file, in name order."""
    digest = hashlib.sha256()
    text = (folder / "README.md").read_text(encoding="utf-8").split(ADDENDA)[0].rstrip()
    digest.update(text.encode())  # trailing blank lines don't count, so an addendum can start
    for path in sorted(folder.glob("results*.jsonl")):
        digest.update(path.name.encode() + b"\n" + path.read_bytes())
    return "sha256:" + digest.hexdigest()


def authors(root: Path = ROOT) -> list[dict]:
    """Every paper's authors: the people of the root ``CITATION.cff``, then the group.

    The maintainer decided on 2026-11-19 to be an author of every paper (#5260). The
    agents' part goes in each paper's text, not in this list.

    Args:
        root: The repository's root, which holds the root ``CITATION.cff``.
    """
    people = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))["authors"]
    return [*people, GROUP]


def citation(folder: Path, date: str, version: str, digest: str | None = None) -> dict:
    """The paper's ``CITATION.cff``, as a mapping."""
    cff = {
        "cff-version": "1.2.0",
        "message": "If you use this paper, please cite it as below.",
        "type": "article",
        "title": title((folder / "README.md").read_text(encoding="utf-8")),
        "authors": authors(),
        "date-released": date,
        "version": version,
        "url": f"{SITE}{folder.name}/index.html",
        "repository-code": REPO,
    }
    if digest:
        cff["identifiers"] = [{"type": "other", "value": digest,
                               "description": "SHA-256 of the published text above its "
                               "addenda, and of its results files"}]  # fmt: skip
    return cff


def write_citation(folder: Path, cff: dict) -> None:
    """Write ``CITATION.cff`` in ``folder``."""
    text = yaml.safe_dump(cff, sort_keys=False, allow_unicode=True, width=1000)
    (folder / "CITATION.cff").write_text(text, encoding="utf-8")


def problems(folder: Path, date: str) -> list[str]:
    """What is wrong with one paper's status and citation file, if anything."""
    text = (folder / "README.md").read_text(encoding="utf-8")
    status, path = paper_status(text), folder / "CITATION.cff"
    if status is None:
        return ["no status line under the abstract (**Status:** Preprint. Work left: #N …)"]
    if status.preprint and not status.issues:
        return ["a preprint must name the issues that hold it open"]
    if not path.exists():
        return ["no CITATION.cff (scripts/papers.py new)"]
    cff = yaml.safe_load(path.read_text(encoding="utf-8"))
    want = citation(folder, date, PREPRINT if status.preprint else PUBLISHED)
    out = [f"CITATION.cff {key} is {cff.get(key)!r}, not {want[key]!r}"
           for key in ("cff-version", "type", "title", "authors", "date-released", "version", "url")
           if cff.get(key) != want[key]]  # fmt: skip
    if not status.preprint:
        stored = [i.get("value") for i in cff.get("identifiers", []) if i.get("type") == "other"]
        if stored != [frozen_hash(folder)]:
            out.append("the published text or results changed: add an addendum instead")
    url = alphaxiv_url(folder)
    if url and status.preprint:
        out.append("a preprint has an alphaXiv URL: only a published paper goes there")
    if url and not ALPHAXIV_URL.fullmatch(url):
        out.append(f"the alphaXiv URL {url!r} is not https://www.alphaxiv.org/abs/<id>")
    return out


def check(papers: Path = PAPERS) -> list[str]:
    """Every problem of every listed paper, each as ``<slug>: <problem>``."""
    return [f"{slug}: {p}" for date, slug in entries(papers) for p in problems(papers / slug, date)]


def work_left(papers: Path = PAPERS) -> dict[str, tuple[int, ...]]:
    """Each preprint's issues of work left, by slug. A bracketed issue is a citation."""
    out = {}
    for _, slug in entries(papers):
        status = paper_status((papers / slug / "README.md").read_text(encoding="utf-8"))
        if status is not None and status.preprint:
            out[slug] = status.issues
    return out


FIX = "remove it from the status line, or run scripts/papers.py publish {slug} if no work is left"


def named(issues: set[int], papers: Path = PAPERS) -> list[str]:
    """A problem for each status line that names one of ``issues`` as work left."""
    return [f"{slug}: #{n} is closed, or closes in this PR, but the line names it: "
            f"{FIX.format(slug=slug)}"
            for slug, left in work_left(papers).items() for n in left if n in issues]  # fmt: skip


def closed(numbers: set[int]) -> set[int]:
    """Which of ``numbers`` are closed on GitHub."""
    import json
    import subprocess

    out = set()
    for n in sorted(numbers):
        view = subprocess.run(["gh", "issue", "view", str(n), "--repo", REPO, "--json", "state"],
                              capture_output=True, text=True, check=True)  # fmt: skip
        if json.loads(view.stdout)["state"] == "CLOSED":
            out.add(n)
    return out


def new(slug: str, work: str, papers: Path = PAPERS) -> None:
    """Give a paper its preprint status line, under the abstract, and its citation file."""
    folder = papers / slug
    text = (folder / "README.md").read_text(encoding="utf-8")
    if paper_status(text) is None:
        head, _, rest = text.partition("\n\n")  # the title
        abstract, _, body = rest.partition("\n\n")
        line = f"**Status:** Preprint. Work left: {work}"
        (folder / "README.md").write_text(f"{head}\n\n{abstract}\n\n{line}\n\n{body}")
    date = dict((s, d) for d, s in entries(papers))[slug]
    write_citation(folder, citation(folder, date, PREPRINT))


def publish(slug: str, papers: Path = PAPERS, today: str | None = None) -> None:
    """Publish a paper: its status line says so, and its citation file holds the hash."""
    folder = papers / slug
    today = today or datetime.now(UTC).strftime("%Y-%m-%d")
    readme = folder / "README.md"
    line = f"**Status:** Published {today}. Changes since are in the addenda at the end."
    readme.write_text(STATUS.sub(line, readme.read_text(encoding="utf-8"), count=1))
    date = dict((s, d) for d, s in entries(papers))[slug]
    write_citation(folder, citation(folder, date, PUBLISHED, frozen_hash(folder)))


def _published(slug: str, papers: Path) -> Path:
    """The folder of a published paper.

    Args:
        slug: The paper's folder name.
        papers: The folder that holds the papers.

    Returns:
        The paper's folder.

    Raises:
        SystemExit: The paper is a preprint, or has no status line.
    """
    folder = papers / slug
    status = paper_status((folder / "README.md").read_text(encoding="utf-8"))
    if status is None or status.preprint:
        raise SystemExit(f"{slug} is not published: only a published paper goes to alphaXiv")
    return folder


def version(text: str) -> int:
    r"""The alphaXiv version of a published paper: 1, and 1 more for each dated addendum.

    Args:
        text: The paper's README.

    Returns:
        The version number.

    Examples:
        >>> version("# T\n\nText.")
        1
        >>> version("# T\n\n## Addenda\n\n### 2026-12-01\n\nA fix.\n\n### 2027-01-02\n\nMore.")
        3
    """
    addenda = text.split(ADDENDA, 1)[1] if ADDENDA in text else ""
    return 1 + len(re.findall(r"^### \d{4}-\d{2}-\d{2}", addenda, re.M))


def abstract(text: str) -> str:
    r"""The paper's abstract as written: the first paragraph after its title, maths kept.

    Args:
        text: The paper's README.

    Returns:
        The paragraph, on one line.

    Examples:
        >>> abstract("# T\n\nIt fits $R^2$ = 0.5\nwell.\n\n**Status:** x")
        'It fits $R^2$ = 0.5 well.'
    """
    body = text.split("\n\n", 1)[1].lstrip("\n")
    return " ".join(body.split("\n\n", 1)[0].split())


def submission(folder: Path) -> str:
    """The text of ``submission.md``: each field of alphaXiv's form for one paper.

    Args:
        folder: The paper's folder.

    Returns:
        The Markdown text.
    """
    text = (folder / "README.md").read_text(encoding="utf-8")
    paper = _read_paper(folder, REPO)
    names = [a.get("name") or f"{a['given-names']} {a['family-names']}" for a in authors()]
    doi = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))["doi"]
    files = sorted(p.name for p in folder.glob("results*.jsonl"))
    links = [f"{SITE}{folder.name}/index.html", f"{REPO}/tree/main/papers/{folder.name}",
             *(f"{REPO}/blob/main/papers/{folder.name}/{f}" for f in files),
             *sorted(set(DATASET.findall(text))), f"https://doi.org/{doi}"]  # fmt: skip
    lines = [
        f"# alphaXiv AI Track submission: {folder.name}",
        "",
        f"Upload `paper.pdf` with these fields from the project account, at {AI_TRACK}.",
        f"Then record the URL: `uv run python scripts/papers.py alphaxiv {folder.name} <url>`.",
        "",
        f"- **Title:** {paper.title}",
        f"- **Authors, in order:** {'; '.join(names)}",
        f"- **Licence:** {LICENCE}",
        f"- **Categories:** {', '.join(CATEGORIES)}",
        f"- **Track:** {ALPHAXIV}",
        f"- **Version:** {version(text)}",
        "- **File:** `paper.pdf`, beside this file",
        f"- **Disclosure:** {DISCLOSURE}",
        "",
        "## Abstract",
        "",
        abstract(text),
        "",
        "## Links",
        "",
        *(f"- {link}" for link in links),
        "",
    ]
    return "\n".join(lines)


def submit(slug: str, papers: Path = PAPERS, dist: Path = DIST, out: Path = SUBMISSIONS) -> Path:
    """Write the upload package of a published paper: its PDF and ``submission.md``.

    Args:
        slug: The paper's folder name.
        papers: The folder that holds the papers.
        dist: The built site, which holds ``research/<slug>/paper.pdf``.
        out: The folder that gets one package folder for each paper.

    Returns:
        The package's folder.

    Raises:
        SystemExit: The paper is a preprint, or the site has no PDF of it.
    """
    folder = _published(slug, papers)
    pdf = dist / "research" / slug / "paper.pdf"
    if not pdf.is_file():
        raise SystemExit(f"no {pdf}: run make build, which prints each paper's PDF")
    package = out / slug
    package.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, package / "paper.pdf")
    (package / "submission.md").write_text(submission(folder), encoding="utf-8")
    return package


def record(slug: str, url: str, papers: Path = PAPERS) -> None:
    """Record a published paper's alphaXiv URL in its ``CITATION.cff``.

    It adds a ``url`` identifier, or replaces the one there. The hash identifier stays.

    Args:
        slug: The paper's folder name.
        url: The paper's page on alphaXiv.
        papers: The folder that holds the papers.

    Raises:
        SystemExit: The paper is a preprint, or the URL is not an alphaXiv paper page.
    """
    folder = _published(slug, papers)
    if not ALPHAXIV_URL.fullmatch(url):
        raise SystemExit(f"{url!r} is not https://www.alphaxiv.org/abs/<id>")
    path = folder / "CITATION.cff"
    cff = yaml.safe_load(path.read_text(encoding="utf-8"))
    kept = [i for i in cff.get("identifiers", []) if i.get("description") != ALPHAXIV]
    cff["identifiers"] = [*kept, {"type": "url", "value": url, "description": ALPHAXIV}]
    write_citation(folder, cff)


def muse_issue(slug: str, papers: Path = PAPERS, run=None) -> str:
    """Open the issue that asks Muse to post a published paper to alphaXiv.

    Its label ``muse`` tells Muse to act on it, and ``scripts/next_issue.py`` never takes
    it. Its body holds the steps and each field of the form. It isn't opened twice: an open
    issue with the same title stands.

    Args:
        slug: The paper's folder name.
        papers: The folder that holds the papers.
        run: Runs a ``gh`` command and returns its standard output. The default calls
            ``gh`` itself.

    Returns:
        The issue's URL, or the line that says it is open already.
    """
    import json
    import subprocess

    def gh(*args: str, stdin: str | None = None) -> str:
        """Run ``gh`` with ``args``, and return its output.

        Args:
            *args: The arguments after ``gh``.
            stdin: Text to send to the command.

        Returns:
            The standard output.
        """
        done = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True,
                              check=True)  # fmt: skip
        return done.stdout

    run = run or gh
    folder = _published(slug, papers)
    text = (folder / "README.md").read_text(encoding="utf-8")
    title = f"Post {slug} v{version(text)} to the alphaXiv AI Track"
    listed = run("issue", "list", "--repo", REPO, "--state", "open", "--label", MUSE_LABELS[0],
                 "--search", f"{title} in:title", "--json", "title")  # fmt: skip
    if any(i["title"] == title for i in json.loads(listed or "[]")):
        return f"open already: {title}"
    steps = (
        "For Muse (`docs/muse.md`). Post this published paper to the alphaXiv AI Track "
        "(#5259).\n\n"
        "1. Build the package: `make build`, then "
        f"`uv run python scripts/papers.py submit {slug}`. "
        f"It writes `build/alphaxiv/{slug}/paper.pdf` and `submission.md`.\n"
        f"2. Upload `paper.pdf` with the fields below at {AI_TRACK}, from the project account.\n"
        f"3. Record the URL in a PR whose body says `Closes` and this issue's number: "
        f"`uv run python scripts/papers.py alphaxiv {slug} <url>`.\n\n"
    )
    labels = [arg for label in MUSE_LABELS for arg in ("--label", label)]
    return run("issue", "create", "--repo", REPO, "--title", title, *labels, "--body-file", "-",
               stdin=steps + submission(folder)).strip()  # fmt: skip


def _parser() -> argparse.ArgumentParser:
    """The command line: one subcommand for each action.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check_cmd = sub.add_parser("check", help="Fail on a paper with no status, or a changed one.")
    check_cmd.add_argument(
        "--closing",
        default="",
        help="Issues a PR closes, e.g. 4250,5394: fail if a line names one.",
    )
    check_cmd.add_argument("--online", action="store_true",
                           help="Ask GitHub, and fail if a line names a closed issue.")  # fmt: skip
    one = sub.add_parser("new", help="Give a paper its preprint status and citation file.")
    one.add_argument("slug")
    one.add_argument("--work", required=True, help='The issues still open, e.g. "#12 (x)".')
    sub.add_parser("publish", help="Publish a paper with no work left.").add_argument("slug")
    submit_cmd = sub.add_parser("submit", help="Write a published paper's alphaXiv package.")
    submit_cmd.add_argument("slug")
    submit_cmd.add_argument(
        "--issue", action="store_true", help="Open the issue that asks Muse to post it."
    )
    url_cmd = sub.add_parser("alphaxiv", help="Record a published paper's alphaXiv URL.")
    url_cmd.add_argument("slug")
    url_cmd.add_argument("url")
    return parser


def main() -> int:
    """Run the command. Return the exit code."""
    parser = _parser()
    args = parser.parse_args()
    if args.command == "new":
        new(args.slug, args.work)
    elif args.command == "publish":
        publish(args.slug)
    elif args.command == "submit":
        if args.issue:  # Muse builds the package itself, so the issue needs no PDF here
            print(muse_issue(args.slug))
        else:
            print(f"Wrote {submit(args.slug)}: upload it from the project account.")
    elif args.command == "alphaxiv":
        record(args.slug, args.url)
    else:  # check: offline in lint, with --closing in CI, with --online nightly (#5395)
        found = check()
        closing = {int(n.strip().lstrip("#")) for n in args.closing.split(",") if n.strip()}
        if args.online:
            closing |= closed({n for left in work_left().values() for n in left})
        found += named(closing)
        print("\n".join(found) or "Every paper has its status and citation file.")
        return 1 if found else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
