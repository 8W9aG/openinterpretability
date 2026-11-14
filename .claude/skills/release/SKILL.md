---
name: release
description: Cut a package release of openinterp. First resolve the open Dependabot and code-scanning alerts, and propose new issues and papers for the literature review. Refresh the sister-neuron rows (`openinterp sisters`) for every pole VERIFIED since the last run. Bump `openinterp.__version__`, date the CHANGELOG's Unreleased section, open the release PR, tag the merged commit and publish the GitHub release that Zenodo archives. Use when the user asks to release, cut a version, bump the version or ship what's on main.
---

# Releasing openinterp

A release does two jobs. It refreshes the analyses that are too slow to run per experiment, which today is the sister search. Then it versions what's on `main`. Do them in that order, so that the release ships the refreshed rows.

The CHANGELOG's header explains the two version numbers. The **package version** (`openinterp/__init__.py`) changes every release. The **protocol version** (`PROTOCOL_VERSION` in `openinterp/core/schemas.py`) changes only in the PR that changes the decision rule, never here. A release only reports it.

## When to release

Wait at least 24 hours after the last release was published. Check before anything else:

```bash
gh release view --json tagName,publishedAt --jq '"\(.tagName) \(.publishedAt)"'
date -u +%Y-%m-%dT%H:%M:%SZ
```

If less than 24 hours have passed, stop and say when the next release may go out. Until then, keep working issues from the open milestone (`work-an-issue`). When the milestone has no eligible issue left, add more. Choose eligible issues from Linear by priority, across research, engineering, the site and docs. Give each one the milestone and the `release: <version>` label.

## Before the release

Do these four steps first, in order. Each fix goes in its own PR, which merges before the release branch is cut.

1. **Resolve every open Dependabot alert.**
   ```bash
   gh api "repos/8W9aG/openinterpretability/dependabot/alerts?state=open" --jq '.[] | "\(.number) \(.security_advisory.severity) \(.dependency.package.name) -> \(.security_vulnerability.first_patched_version.identifier)"'
   gh pr list --author app/dependabot
   ```
   Merge Dependabot's own PRs when CI is green. For an alert with no PR, upgrade the package (`uv lock --upgrade-package <name>`), run `make lint test`, and open one PR for all of them. GitHub closes an alert when `main` no longer has the vulnerable version. If no fixed version exists, or the package is never reached at run time, don't dismiss the alert yourself. List it in the release PR with the reason, and the maintainer decides.
2. **Resolve every open code-scanning (CodeQL) alert.**
   ```bash
   gh api "repos/8W9aG/openinterpretability/code-scanning/alerts?state=open" --jq '.[] | "\(.number) \(.rule.severity) \(.rule.id) \(.most_recent_instance.location.path):\(.most_recent_instance.location.start_line)"'
   ```
   Fix each one in the code, with a test where the fix changes behaviour. GitHub closes an alert when the next scan of `main` no longer finds it. A false positive is listed in the release PR with the reason, for the maintainer to dismiss.
3. **Propose new work.** Read the *Use in this repo* section of each page in `docs/literature/`. Also read the *Limits* and *What this means* sections of the analyses and papers that changed since the last tag (`git diff --stat v<previous> -- docs papers`). Then do a short web search for new papers and posts on native-neuron interpretability, explanation scoring and automated interpretability. Before opening anything, search the open issues and `docs/literature/` for it. Open each new idea as an issue with an `area:` label and a priority label, and give no milestone. Open each new paper as an issue titled `Review <title>`, with the label `area: literature`. List them all in the release PR.
4. **Release the version**, with the procedure below.

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
   Write the bullet in a fragment, `changelog.d/sisters-<version>.changed.md` (`changelog.d/README.md`). Give the count of new rows and new sisters, by pair. Name the strongest few, since each is now a `sister` draw. Don't rewrite `docs/analyses/sister-neurons.md`: it reports the first run, and a dated note of later runs belongs in the CHANGELOG.
   Then refresh the registry coverage, which the home page shows. Run it in the background, one model at a time, on an idle machine:
   ```bash
   for m in $(uv run python -c 'from openinterp.planner import configured_models; print(*configured_models())'); do
     uv run openinterp coverage --model "$m"
   done
   ```
   Each run appends its rows to `docs/analyses/registry-coverage.jsonl` with the day's date, and the site shows each model's newest run. Commit the file in the release PR, and write the new shares in a fragment, `changelog.d/coverage-<version>.changed.md`.
5. **Check:** `make check-all lint test`. A failure here comes from `main`, not from the release. Stop and report it rather than fixing it in the release PR.
   Then check memory: `uv run python scripts/memory_benchmark.py` (about 7 minutes on CPU). Run it in the background, and keep the machine otherwise idle. It scans and reviews GPT-2 and Qwen2.5-0.5B and compares each peak with `scripts/memory_baseline.json`. If a peak is more than 25% above the baseline, it exits 1. Stop and report the regression, as for a failed test. If it says a peak is lower, run it again with `--update` and commit the new baseline in the release PR. On a platform other than the baseline's it only reports. Don't overwrite the baseline from there.
6. **Bump and date.** In `openinterp/__init__.py`, set `__version__ = "<version>"`. Then gather the changelog fragments into the version's section. The script writes it under `## [Unreleased]`, which stays empty, and removes the fragments:
   ```bash
   python3 scripts/changelog.py collect <version> <YYYY-MM-DD> "Ships with protocol version **<PROTOCOL_VERSION>**."
   ```
   If the protocol version changed since the last tag (`git diff v<previous> -- openinterp/core/schemas.py | grep PROTOCOL_VERSION`), say so in that line: "Ships with protocol version **0.9** (was 0.8)." The section's entries explain why.
7. **Commit and push.** Check the branch first (`git branch --show-current` prints `release/<version>`):
   ```bash
   git add CHANGELOG.md changelog.d openinterp/__init__.py docs/analyses/sister-neurons.jsonl docs/analyses/registry-coverage.jsonl
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
10. **Publish the GitHub release** from the tag. Zenodo archives each GitHub release (not each tag) and gives it its own DOI, so a citation can name this exact version (`CITATION.cff`). Use the version's CHANGELOG section as the notes:
    ```bash
    notes=$(mktemp)
    awk '/^## \[<version>\]/{f=1; next} /^## \[/{f=0} f' CHANGELOG.md > "$notes"
    gh release create v<version> --verify-tag --title "openinterp <version> (protocol v<PROTOCOL_VERSION>)" --notes-file "$notes"
    ```
    Then check that Zenodo made the DOI: the release's Zenodo record appears under the repository's concept DOI after a few minutes. If it doesn't, the Zenodo GitHub integration is off. Tell the user, since only the repository owner can turn it on at zenodo.org.
11. **Close the milestone.** Move any open issues in `v<version>` to the next milestone, then close it:
    ```bash
    gh issue list --milestone v<version> --state open --json number --jq '.[].number' | xargs -I{} gh issue edit {} --milestone v<next>
    gh api -X PATCH repos/8W9aG/openinterpretability/milestones/<number> -f state=closed
    ```
