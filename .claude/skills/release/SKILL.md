---
name: release
description: Cut a package release of openinterp. Refresh the sister-neuron rows (`openinterp sisters`) for every pole VERIFIED since the last run. Bump `openinterp.__version__`, date the CHANGELOG's Unreleased section, open the release PR and tag the merged commit. Use when the user asks to release, cut a version, bump the version or ship what's on main.
---

# Releasing openinterp

A release does two jobs. It refreshes the analyses that are too slow to run per experiment, which today is the sister search. Then it versions what's on `main`. Do them in that order, so that the release ships the refreshed rows.

The CHANGELOG's header explains the two version numbers. The **package version** (`openinterp/__init__.py`) changes every release. The **protocol version** (`PROTOCOL_VERSION` in `openinterp/core/schemas.py`) changes only in the PR that changes the decision rule, never here. A release only reports it.

## Procedure

1. **Start from a fresh `origin/main` on a release branch.** Work in the main checkout, not a worktree. Pick the next version: bump the patch (`0.0.11` → `0.0.12`) unless the user names one.
   ```bash
   git fetch origin --tags && git switch -c release/<version> origin/main
   ```
2. **Check the corpus.** The sister rows record the corpus's SHA-256, and a run must use the same corpus, or the new rows aren't comparable with the old ones:
   ```bash
   shasum -a 256 corpus.jsonl   # must be ded48cfc… (docs/experiments.md, "The scan corpus")
   head -1 docs/analyses/sister-neurons.jsonl | grep -o '"corpus_sha256":"[0-9a-f]*"'
   ```
   If `corpus.jsonl` is missing, rebuild it with the README's `openinterp corpus` command. If the two hashes differ, stop and ask the user: a new corpus means rerunning every row, not resuming.
3. **Refresh the sister rows.** Run this on a machine with a GPU (MPS or CUDA). On CPU the passes against Qwen2.5-1.5B take hours. Run it in the background: with every pair to search it takes about an hour, but a run with nothing new finishes in seconds.
   ```bash
   uv run openinterp sisters --device auto
   ```
   It searches every ordered pair of the configured models (`[tool.openinterp] models`) plus `SIBLINGS`. It skips every pole that already has a row for that model. So it only searches the poles VERIFIED since the last run, and any pair that a newly configured model adds. Each pair prints `N VERIFIED poles of A against B`, then its progress (`forward:` and `backward:` positions). A pair with nothing new prints nothing. If no accelerator is available, skip this step and say so in the PR body. The next release picks the poles up.
4. **Record what changed.** If the run wrote rows (`git diff --stat docs/analyses/sister-neurons.jsonl`), count them:
   ```bash
   uv run python - <<'EOF'
   import subprocess
   from openinterp.sisters import SisterRow
   old = subprocess.run(["git", "show", "HEAD:docs/analyses/sister-neurons.jsonl"], capture_output=True, text=True).stdout
   seen = {(r.slug, r.other) for r in map(SisterRow.model_validate_json, old.splitlines())}
   rows = [SisterRow.model_validate_json(l) for l in open("docs/analyses/sister-neurons.jsonl")]
   new = [r for r in rows if (r.slug, r.other) not in seen]
   print(len(new), "new rows,", sum(r.sister is not None for r in new), "new sisters")
   for r in new:
       if r.sister: print(" ", r.slug, "->", r.sister.slug(r.other), round(r.sister.pearson, 2))
   EOF
   ```
   Add a bullet under `### Changed` in the CHANGELOG's Unreleased section. Give the count of new rows and new sisters, by pair. Name the strongest few, since each is now a `sister` draw. Don't rewrite `docs/analyses/sister-neurons.md`: it reports the first run, and a dated note of later runs belongs in the CHANGELOG.
5. **Check:** `make check-all lint test`. A failure here comes from `main`, not from the release. Stop and report it rather than fixing it in the release PR.
   Then check memory: `uv run python scripts/memory_benchmark.py` (about 7 minutes on CPU). Run it in the background, and keep the machine otherwise idle. It scans and reviews GPT-2 and Qwen2.5-0.5B and compares each peak with `scripts/memory_baseline.json`. If a peak is more than 25% above the baseline, it exits 1. Stop and report the regression, as for a failed test. If it says a peak is lower, run it again with `--update` and commit the new baseline in the release PR. On a platform other than the baseline's it only reports. Don't overwrite the baseline from there.
6. **Bump and date.** In `openinterp/__init__.py`, set `__version__ = "<version>"`. In `CHANGELOG.md`, put a dated heading under `## [Unreleased]`, so every entry that was there now sits under the new version:
   ```markdown
   ## [Unreleased]

   ## [<version>] - <YYYY-MM-DD>

   Ships with protocol version **<PROTOCOL_VERSION>**.
   ```
   If the protocol version changed since the last tag (`git diff v<previous> -- openinterp/core/schemas.py | grep PROTOCOL_VERSION`), say so in that line: "Ships with protocol version **0.9** (was 0.8)." The section's entries explain why.
7. **Commit and push.** Check the branch first (`git branch --show-current` prints `release/<version>`):
   ```bash
   git add CHANGELOG.md openinterp/__init__.py docs/analyses/sister-neurons.jsonl
   git commit -m "Release <version> (protocol v<PROTOCOL_VERSION>)"
   git push -u origin release/<version>
   ```
8. **Open the PR** against `main`:
   ```bash
   gh pr create --title "Release <version> (protocol v<PROTOCOL_VERSION>)" --body "..."
   ```
   The body says what the release bumps and whether the protocol version changed. It also gives the sister refresh (rows and sisters added, or why it was skipped). A release touches code, so it always waits for a human to merge it (`land-experiment`, step 7).
9. **Tag the merged commit.** Do this after the human merges the PR, not before. A squash merge makes a new commit on `main`, so a tag on the release branch's commit is not on `main`:
   ```bash
   git fetch origin
   git tag -a v<version> "$(gh pr view <number> --json mergeCommit --jq .mergeCommit.oid)" -m "openinterp <version> (protocol v<PROTOCOL_VERSION>)"
   git push origin v<version>
   git merge-base --is-ancestor v<version> origin/main && echo on-main
   ```
