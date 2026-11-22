"""``open-interpretability explain-share``: one PR for an ``explain --local`` run (#5827).

``explain --run --local`` keeps its experiment records, and the explanation record, in
the user's own ``wiki/``. Nothing reaches GitHub. This command shares the run later. It
opens one PR with the explanation record and every experiment record that its run lists as
``local``, with their ``.sig`` files.

The records keep the inputs of a landed run, so CI's ``reproduce`` can check them. The PR
is marked ready and left for ``auto-merge.yml`` or a maintainer. Nothing waits for CI, and
nothing merges here (the maintainer's decision on #5827). A local run claims no unit. So
the PR body names each unit whose folder on ``origin/main`` gained a record after the
local one, under *Landed since*.
"""

from __future__ import annotations

import os
from pathlib import Path

from openinterpretability.core.explanation_records import ExplanationRecord, record_path
from openinterpretability.core.paths import DEFAULT_WIKI_DIR


def local_entries(run) -> list[tuple[str, str, Path]]:
    """A run's local experiments: slug, verdict and record path, from ``run.local``.

    Args:
        run: The record's :class:`~openinterpretability.core.explanation_records.ExplainRun`,
            or None.

    Returns:
        One tuple for each local experiment, in the run's order.

    Examples:
        >>> from openinterpretability.core.explanation_records import ExplainRun
        >>> run = ExplainRun(share_before=0, share_after=0, stopped="done",
        ...                  local=["m/L1/N2-positive VERIFIED wiki/experiments/a.json"])
        >>> local_entries(run)
        [('m/L1/N2-positive', 'VERIFIED', PosixPath('wiki/experiments/a.json'))]
        >>> local_entries(None)
        []
    """
    out = []
    for line in run.local if run is not None else []:
        slug, verdict, path = line.split(" ", 2)
        out.append((slug, verdict, Path(path)))
    return out


def share_files(run) -> list[Path]:
    """The experiment files that the PR adds, relative to the checkout: records and ``.sig``.

    Args:
        run: The explanation record's run, or None.

    Returns:
        Each local experiment's record, and its signature where there is one.

    Raises:
        FileNotFoundError: A record that the run lists is no longer in the checkout.

    Examples:
        >>> import os, tempfile
        >>> from openinterpretability.core.explanation_records import ExplainRun
        >>> here = os.getcwd()
        >>> os.chdir(tempfile.mkdtemp())
        >>> _ = Path("r.json").write_text("{}"), Path("r.json.sig").write_text("sig")
        >>> run = ExplainRun(share_before=0, share_after=0, stopped="done",
        ...                  local=["m/L1/N2-positive VERIFIED r.json"])
        >>> share_files(run)
        [PosixPath('r.json'), PosixPath('r.json.sig')]
        >>> os.chdir(here)
    """
    files = []
    for _, _, path in local_entries(run):
        if not path.is_file():
            raise FileNotFoundError(f"{path}: the run lists it, but it isn't in this checkout")
        sig = path.with_name(path.name + ".sig")
        files += [path, *([sig] if sig.is_file() else [])]
    return files


def shared_run(run):
    """``run`` as the shared record holds it: its local experiments land in the same PR.

    Args:
        run: The explanation record's run.

    Returns:
        The run with each ``local`` entry moved to ``landed``, marked as shared.

    Examples:
        >>> from openinterpretability.core.explanation_records import ExplainRun
        >>> run = ExplainRun(share_before=0, share_after=0, stopped="done",
        ...                  local=["m/L1/N2-positive VERIFIED r.json"])
        >>> moved = shared_run(run)
        >>> moved.landed, moved.local
        (['m/L1/N2-positive VERIFIED shared with this explanation'], [])
    """
    landed = [f"{slug} {verdict} shared with this explanation"
              for slug, verdict, _ in local_entries(run)]  # fmt: skip
    return run.model_copy(update={"landed": [*run.landed, *landed], "local": []})


def _shared(record: ExplanationRecord) -> ExplanationRecord:
    """``record`` as the PR holds it (:func:`shared_run`).

    Args:
        record: The explanation record of a local run.

    Returns:
        The record, with its run's local experiments moved to ``landed``.
    """
    if record.run is None:
        return record
    return record.model_copy(update={"run": shared_run(record.run)})


def landed_since(files: list[Path]) -> list[str]:
    """The units whose folder in this checkout has a record newer than the local one.

    Run it in a checkout of ``origin/main``. Record names are UTC timestamps, so a later
    name is a later record.

    Args:
        files: The local experiment files, relative to the checkout.

    Returns:
        One line for each unit that gained a record, naming it and the newer record.

    Examples:
        >>> import os, tempfile
        >>> here = os.getcwd()
        >>> os.chdir(tempfile.mkdtemp())
        >>> Path("u").mkdir()
        >>> _ = Path("u/20261121T000000Z.json").write_text("{}")
        >>> landed_since([Path("u/20261120T000000Z.json")])
        ['u: 20261121T000000Z.json landed after 20261120T000000Z.json']
        >>> os.chdir(here)
    """
    out = []
    for path in files:
        if path.suffix != ".json" or not path.parent.is_dir():
            continue
        newer = sorted(p.name for p in path.parent.glob("*.json") if p.name > path.name)
        if newer:
            out.append(f"{path.parent.as_posix()}: {', '.join(newer)} landed after {path.name}")
    return out


def share(record_file: Path) -> tuple[str, bool]:
    """Open one PR with ``record_file`` and its run's local experiments, and hand it over.

    Args:
        record_file: The explanation record that ``explain --local`` wrote.

    Returns:
        The PR's URL, and whether it has merged already.
    """
    from openinterpretability.cli_loop import loop_worktree
    from openinterpretability.experiment_loop import land, openinterpretability, pr_url, reset, sh
    from openinterpretability.explain_land import branch_name

    original = ExplanationRecord.model_validate_json(record_file.read_text("utf-8"))
    files, record = share_files(original.run), _shared(original)
    contents = {path: path.read_bytes() for path in files}  # read before leaving the checkout
    os.chdir(loop_worktree()[1])
    reset()  # the PR's base is a fresh origin/main, as for any landing
    since = landed_since(files)  # before the copies, which would hide what main gained
    sh(["git", "switch", "--quiet", "-c", f"{branch_name(record)}-share"])
    for path, data in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    target = record_path(DEFAULT_WIKI_DIR, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    openinterpretability(["wiki"])  # the pages that the records add, as a loop commits them
    openinterpretability(["check", "--base-ref", "origin/main"])
    title = f"Explanation: {record.model_name} · {record.prompt_id} (shared from a local run)"
    sh(["git", "add", str(DEFAULT_WIKI_DIR)])
    sh(["git", "commit", "--quiet", "-m", title])
    sh(["git", "push", "--quiet", "-u", "origin", "HEAD"])
    sh(["gh", "pr", "create", "--title", title, "--body", _share_body(record, files, since)])
    url = pr_url()
    return url, land(url)


def _share_body(record: ExplanationRecord, files: list[Path], since: list[str]) -> str:
    """The PR body: the explanation's, the experiments it shares, and any that landed since.

    Args:
        record: The record as the PR holds it (:func:`_shared`).
        files: The experiment files that the PR adds.
        since: The lines of :func:`landed_since`.

    Returns:
        The body, in Markdown.
    """
    from openinterpretability.explain_land import pr_body

    records = [p for p in files if p.suffix == ".json"]
    lines = [pr_body(record), "", f"Shared from an `explain --local` run (#5827): "
             f"{len(records)} experiment record(s)."]  # fmt: skip
    lines += [f"- `{p.as_posix()}`" for p in records]
    if since:
        lines += ["", "**Landed since.** These units gained a record on main after the local "
                  "one, because a local run claims nothing:"]  # fmt: skip
        lines += [f"- {line}" for line in since]
    return "\n".join(lines)
