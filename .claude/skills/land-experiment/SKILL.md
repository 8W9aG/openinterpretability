---
name: land-experiment
description: Run experiments end to end and land them. This is `choose-next-experiment` plus building the site, waiting for CI to go green and merging the PR when you have write access; without access, leave the PR for a maintainer and move on. Use when the user asks to run experiments *and merge them*, to "land" results, or to experiment continuously without a human merging each PR. For an experiment that should stop at an open PR, use `choose-next-experiment` alone.
---

# Landing an experiment

This skill wraps `choose-next-experiment`, which still decides *what* to run and *how* to post it. Everything there applies unchanged: accept the draw, claim at once, read the history, and post whatever the verdict is. This skill adds what happens after the PR is posted, up to the merge.

## Procedure

1. **Start from a fresh, detached `origin/main`:**
   ```bash
   git fetch origin && git switch --detach origin/main
   ```
   Use this instead of `git pull` on a local `main`: `main` may be checked out in another worktree, and a detached start never reuses an old branch.
2. **Run one experiment** with `choose-next-experiment`, steps 2–7: draw, claim, read the history, run it, then `openinterp wiki` and `make check lint test`. Commit `wiki/` (records, `.sig` files and pages; the generated `README.md` indexes are gitignored) and push.
3. **Build the site:**
   ```bash
   make build
   ```
   `dist/` is gitignored, so don't commit it; the deploy workflow rebuilds it from `main` after the merge. The build is a check: if the new record breaks the site compiler or the Parquet export, find out before the merge, not in the deploy. Fix a failure caused by your record. If the failure is in the builder itself, note it in the PR and don't merge (step 7).
4. **Mark the PR ready:** `gh pr ready`.
5. **Bring the branch up to date with `main`.** Other experiments merge while yours runs:
   ```bash
   git fetch origin && git merge origin/main
   ```
   Experiments only add new files, and the generated indexes aren't committed, so this shouldn't conflict. If it does (a record, a page, code), don't resolve it by hand: run `git merge --abort`, comment on the PR describing the conflict, and go to step 9 without merging. Otherwise run `make check`, then push.
6. **Wait for the checks.** CI runs "Lint and tests", "Experiment records" and "Reproduce records". Reproduction takes a few minutes per unsigned record and is skipped for signed ones.
   ```bash
   gh pr checks --watch --fail-fast --interval 30
   ```
   Checks can take a minute to register after a push. If `gh` reports no checks, wait and retry; don't treat that as green. Give up after 45 minutes pending: comment on the PR that checks didn't finish, and go to step 9.
   - **A check fails:** read the log (`gh run view <run-id> --log-failed`). Fix it if the failure is yours: lint, a committed `README.md` index, or a record CI can't reproduce because of something you changed. Then push and wait again. After two failed fix attempts, or if the failure isn't caused by this PR (flaky runner, broken `main`), comment with what you found and go to step 9.
7. **Decide whether you may merge.** Merge only when all of these hold:
   - every check passed on the current head commit;
   - every changed file is under `wiki/`: run `gh pr diff --name-only` and check each path. A PR that touches code, workflows, skills or `.github/allowed_signers` always waits for a human;
   - you have write access: `gh repo view --json viewerPermission -q .viewerPermission` prints `ADMIN`, `MAINTAIN` or `WRITE`;
   - the PR isn't draft, and `gh pr view --json mergeable -q .mergeable` prints `MERGEABLE`.

   If any of these fails, leave the PR ready for review, say in a comment which condition failed, and go to step 9.
8. **Merge:**
   ```bash
   gh pr view --json headRefOid -q .headRefOid     # the head commit; paste it below
   gh pr merge --merge --match-head-commit <that SHA>
   git push origin --delete "experiment/$KEY"
   ```
   `--match-head-commit` refuses the merge if someone pushed after your checks ran. Don't pass `--admin` or use any other way around a failing or pending check, even though `main` isn't branch-protected. Delete the remote branch yourself rather than with `gh pr merge --delete-branch`, which tries to switch the local checkout to `main` and fails if `main` is checked out in another worktree. If the merge is refused, go back to step 5 once. If it's refused again, comment and move on.
9. **Continue.** If you were asked for one experiment (e.g. by `scripts/experiment-loop.sh`), stop. Make sure the working tree is clean, and end with the result line the caller asked for. Otherwise go back to step 1. An unmerged PR stays open for a maintainer: don't wait on it, and don't close it unless the experiment itself was abandoned.

## Rules

- A green PR with a REJECTED or NEEDS_REVISION verdict merges exactly like a VERIFIED one. Merging records the experiment; it doesn't endorse the hypothesis.
- Never edit another experiment's record or page to make a merge or a check pass.
- Never merge a PR you didn't open in this session. Other agents' PRs are theirs to land.
- If `gh` isn't authenticated or has no access to the repo, you can't claim either. Stop and say so; `choose-next-experiment` needs `gh` too.
