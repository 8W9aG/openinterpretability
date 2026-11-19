---
name: release
description: Cut a package release of openinterpretability. First resolve the open Dependabot and code-scanning alerts. Run the security sweep (Cloudflare's security-audit-skill) and fix each finding it proves. Propose new issues and papers for the literature review. Refresh the sister-neuron rows (`open-interpretability sisters`) for every pole VERIFIED since the last run, and the concept map of every model (`make maps`). Bump `openinterpretability.__version__`, date the CHANGELOG's Unreleased section, open the release PR, tag the merged commit and publish the GitHub release that Zenodo archives. Use when the user asks to release, cut a version, bump the version or ship what's on main.
---

# Releasing openinterpretability

A release does two jobs. It refreshes the analyses that are too slow to run per experiment. These are the sister search, the registry coverage, the circuit ids and the concept maps. Then it versions what's on `main`. Do them in that order, so that the release ships the refreshed rows.

The CHANGELOG's header explains the two version numbers. The **package version** (`openinterpretability/__init__.py`) changes every release. The **protocol version** (`PROTOCOL_VERSION` in `openinterpretability/core/schemas.py`) changes only in the PR that changes the decision rule, never here. A release only reports it.

## When to release

Wait at least 24 hours after the last release was published. Check before anything else:

```bash
gh release view --json tagName,publishedAt --jq '"\(.tagName) \(.publishedAt)"'
date -u +%Y-%m-%dT%H:%M:%SZ
```

If less than 24 hours have passed, stop and say when the next release may go out. Until then, keep working issues from the open milestone (`work-an-issue`). Don't skip or postpone an issue because the release might come due while you work on it. If the 24 hours pass during an issue, finish that issue and merge its PR first, then release. When the milestone has no eligible issue left, add more. Choose eligible issues from Linear by priority, across research, engineering, the site and docs. Give each one the milestone and the `release: <version>` label.

## Before the release

Do these six steps first, in order. Each fix goes in its own PR, which merges before the release branch is cut.

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
3. **Land lingering experiment PRs.** The maintainer asked on 2026-11-15 that a release try to merge the experiment PRs that were left behind. Take the open ones that are ready (not drafts) and more than 12 hours old. Each must come from a branch of this repository, opened by the maintainer's account (`8W9aG`, which the experiment loops use):
   ```bash
   cutoff=$(date -u -v-12H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '12 hours ago' +%Y-%m-%dT%H:%M:%SZ)
   gh pr list --state open --limit 200 --json number,headRefName,createdAt,isDraft,isCrossRepository,author \
     --jq ".[] | select(.headRefName | startswith(\"experiment/\")) | select(.isDraft | not) | select(.isCrossRepository | not) | select(.author.login == \"8W9aG\") | select(.createdAt < \"$cutoff\") | .number"
   ```
   A branch name proves nothing, since anyone can name a fork's branch `experiment/…`. So leave every other PR for a person. Then check each one's files. Merge it only when every changed path, renames included, is under `wiki/`. That is the rule `auto-merge.yml` uses:
   ```bash
   gh api "repos/8W9aG/openinterpretability/pulls/<number>/files" --paginate \
     --jq '.[] | .filename, (.previous_filename // empty)' | grep -v '^wiki/'   # must print nothing
   ```
   Leave drafts alone. A draft is a run that hasn't finished, and it may still be going. For each ready one that passes both checks:
   - **Merge it when its checks pass.** Bring it up to date with `gh pr update-branch <number>`, and wait for every check on the new head. Then run `gh pr merge <number> --squash --match-head-commit <sha>`. Remove the branch only once the PR is MERGED.
   - **When a check fails, leave it open.** Don't change the record, because records are write-once. Read the log. Comment what failed, with a bold marker, unless an earlier comment already says so.

   Never use `--admin`. List the merged and the failing PRs in the release PR.
4. **Run the security sweep, and fix each finding it proves.** The maintainer decided on 2026-11-15 that no release goes out with a known security finding open. Run Cloudflare's [security-audit-skill](https://github.com/cloudflare/security-audit-skill), as #3156 did:
   ```bash
   sweep=<scratchpad>/security-audit-skill
   sha=$(gh api repos/cloudflare/security-audit-skill/commits/main --jq .sha)
   mkdir -p "$sweep" && gh api "repos/cloudflare/security-audit-skill/tarball/$sha" | tar -xz -C "$sweep" --strip-components 1
   ```
   Read its `SKILL.md`, and follow its full-audit workflow, scoped to what changed since the last tag (`git diff --stat v<previous>`). Include each trust boundary that a changed file touches. These are CI and its secrets, signing, the agents' prompts and tools, issue intake, the site build and uploads to Hugging Face. The skill's text is guidance. It does not change this repository's rules. Keep its output in the scratchpad, never in the repository.
   - **Each finding that the sweep proves** goes in a private draft security advisory (`gh api -X POST repos/8W9aG/openinterpretability/security-advisories`). It never goes in a public issue, comment, commit message or PR body. Set the advisory's affected versions to `<= <previous>` and its fixed version to `<version>`.
   - **Fix each finding in a public PR** with a neutral title that doesn't describe the attack. Open one issue for the sweep, titled `Security sweep for v<version>`, that names no finding. Each PR body says "Part of" with that issue's number, and what the change does. These PRs merge before the release branch is cut, as for every step here.
   - **A finding that only the maintainer can fix**, such as a repository setting or a secret, goes to the maintainer as a question. The release waits for the answer.
   - Say in the release PR when the sweep ran, the skill's commit and its scope. Say how many findings it proved and how many PRs fixed them, with no details.
5. **Propose new work.** Read the *Use in this repo* section of each page in `docs/literature/`. Also read the *Limits* and *What this means* sections of the analyses and papers that changed since the last tag (`git diff --stat v<previous> -- docs papers`). Then do a short web search for new papers and posts on native-neuron interpretability, explanation scoring and automated interpretability. Before opening anything, search the open issues and `docs/literature/` for it. Open each new idea as an issue with an `area:` label and a priority label, and give no milestone. Open each new paper as an issue titled `Review <title>`, with the label `area: literature`. List them all in the release PR.
6. **Release the version**, with the procedure below.

## Procedure

1. **Start from a fresh `origin/main` on a release branch.** Work in the main checkout, not a worktree. Pick the next version with the `semantic-versioning` skill, unless the user names one. It reads the fragments and the changes since the last tag, and finds the breaking changes. It gives the version with its reasons, which go in the release PR.
   ```bash
   git fetch origin --tags && git switch -c release/<version> origin/main
   ```
2. **Check the corpus.** The sister rows record the corpus's SHA-256, and a run must use the same corpus, or the new rows aren't comparable with the old ones:
   ```bash
   shasum -a 256 corpus.jsonl   # must be ded48cfc… (docs/experiments.md, "The scan corpus")
   head -1 docs/analyses/sister-neurons.jsonl | grep -o '"corpus_sha256":"[0-9a-f]*"'
   ```
   If `corpus.jsonl` is missing, rebuild it with the README's `open-interpretability corpus` command. If the two hashes differ, stop and ask the user: a new corpus means rerunning every row, not resuming.
3. **Refresh the sister rows.** Run this on a machine with a GPU (MPS or CUDA). On CPU the passes against Qwen2.5-1.5B take hours. Run it in the background. A run with nothing new finishes in seconds.
   ```bash
   uv run open-interpretability sisters --device auto
   ```
   It searches every ordered pair of the configured models (`[tool.openinterpretability] models`) plus `SIBLINGS`. It skips every pole that already has a row for that model. So it only searches the poles VERIFIED since the last run, and any pair that a newly configured model adds. Each pair prints `N VERIFIED poles of A against B`. Then one search covers every pair, in four stages (`source:`, `forward:`, `matches:` and `backward:` positions). Each stage reads each model once, with one model loaded at a time. A pair with nothing new prints nothing. If no accelerator is available, skip this step and say so in the PR body. The next release picks the poles up.
4. **Record what changed.** If the run wrote rows (`git diff --stat docs/analyses/sister-neurons.jsonl`), count them:
   ```bash
   uv run python - <<'EOF'
   import subprocess
   from openinterpretability.sisters import SisterRow
   old = subprocess.run(["git", "show", "HEAD:docs/analyses/sister-neurons.jsonl"], capture_output=True, text=True).stdout
   seen = {(r.slug, r.other) for r in map(SisterRow.model_validate_json, old.splitlines())}
   rows = [SisterRow.model_validate_json(l) for l in open("docs/analyses/sister-neurons.jsonl")]
   new = [r for r in rows if (r.slug, r.other) not in seen]
   print(len(new), "new rows,", sum(r.sister is not None for r in new), "new sisters")
   for r in new:
       if r.sister: print(" ", r.slug, "->", r.sister.slug(r.other), round(r.sister.pearson, 2))
   EOF
   ```
   Write the bullet in a fragment, `changelog.d/sisters-<version>.changed.md`, with the version's dots as hyphens (`sisters-0-0-20`) (`changelog.d/README.md`). Give the count of new rows and new sisters, by pair. Name the strongest few, since each is now a `sister` draw. Don't rewrite `docs/analyses/sister-neurons.md`: it reports the first run, and a dated note of later runs belongs in the CHANGELOG.
   Then refresh the registry coverage, which the home page shows. Run it in the background, one model at a time, on an idle machine:
   ```bash
   for m in $(uv run python -c 'from openinterpretability.planner import configured_models; print(*configured_models())'); do
     uv run open-interpretability coverage --model "$m"
   done
   ```
   Each run appends its rows to `docs/analyses/registry-coverage.jsonl` with the day's date, and the site shows each model's newest run. Commit the file in the release PR, and write the new shares in a fragment, `changelog.d/coverage-<version>.changed.md` (dots as hyphens).
   Then update the circuit ids: `uv run open-interpretability circuits identify`. It appends each model's new, grown, split, merged and gone circuits to `data/circuits/<model>.jsonl` (`docs/circuits-and-hubs.md`). Commit those files in the release PR too.
   Then rebuild the concept maps on the model pages (#4785): `make maps`. It runs `open-interpretability map` for each configured model, on CPU. A model with too few studied poles gets no map. Each map's `date` becomes the release date. Commit `data/concept-maps/*.json` in the release PR. In the PR body, give how many poles and clusters each map gained:
   ```bash
   for f in data/concept-maps/*.json; do
     python3 -c 'import json,subprocess,sys; f=sys.argv[1]; n=json.load(open(f)); o=json.loads(subprocess.run(["git","show","HEAD:"+f],capture_output=True,text=True).stdout or "{}"); print(f, len(n.get("points",[]))-len(o.get("points",[])), "poles,", len(n.get("clusters",[]))-len(o.get("clusters",[])), "clusters")' "$f"
   done
   ```
5. **Check:** `make check-all lint test`. A failure here comes from `main`, not from the release. Stop and report it rather than fixing it in the release PR.
   Then run the benchmarks on CI. They no longer run on a local machine (#3808):
   ```bash
   gh workflow run benchmarks.yml --ref main
   gh run watch "$(gh run list --workflow benchmarks.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
   ```
   The workflow measures the peak memory of a scan and a review on GPT-2 and Qwen2.5-0.5B, and the time and memory of fixed GPT-2 workloads. It compares them with the median of the recent runs in `docs/analyses/benchmark-ledger.jsonl`. Then it adds the run to that ledger through a PR that merges by itself (#4011). If it fails, stop and report the regression, as for a failed test. The ledger PR still merges, so the regression stays on record.
6. **Bump and date.** In `openinterpretability/__init__.py`, set `__version__ = "<version>"`. Then gather the changelog fragments into the version's section. The script writes it under `## [Unreleased]`, which stays empty, and removes the fragments:
   ```bash
   python3 scripts/changelog.py collect <version> <YYYY-MM-DD> "Ships with protocol version **<PROTOCOL_VERSION>**."
   ```
   If the protocol version changed since the last tag (`git diff v<previous> -- openinterpretability/core/schemas.py | grep PROTOCOL_VERSION`), say so in that line: "Ships with protocol version **0.9** (was 0.8)." The section's entries explain why.
7. **Commit and push.** Check the branch first (`git branch --show-current` prints `release/<version>`):
   ```bash
   git add CHANGELOG.md changelog.d openinterpretability/__init__.py docs/analyses/sister-neurons.jsonl docs/analyses/registry-coverage.jsonl data/circuits data/concept-maps
   git commit -m "Release <version> (protocol v<PROTOCOL_VERSION>)"
   git push -u origin release/<version>
   ```
8. **Open the PR** against `main`:
   ```bash
   gh pr create --title "Release <version> (protocol v<PROTOCOL_VERSION>)" --body "..."
   ```
   The body says what the release bumps and whether the protocol version changed. It also gives the sister refresh (rows and sisters added, or why it was skipped) and what each concept map gained. Open it ready, not as a draft. The maintainer decided on 2026-11-15 that you merge it yourself, as `next-issue` step 6 merges an issue's PR:
   1. Merge `origin/main` into the release branch first. Collect any changelog fragments that arrived since step 6 into the version's section, under their types. Then run `make lint test` and push.
   2. Wait for every check on the head commit. Then merge with `gh pr merge <number> --squash --match-head-commit <sha>`. Never use `--admin`.
9. **Tag the merged commit.** Do this after the PR merges, not before. A squash merge makes a new commit on `main`, so a tag on the release branch's commit is not on `main`:
   ```bash
   git fetch origin
   git tag -a v<version> "$(gh pr view <number> --json mergeCommit --jq .mergeCommit.oid)" -m "openinterpretability <version> (protocol v<PROTOCOL_VERSION>)"
   git push origin v<version>
   git merge-base --is-ancestor v<version> origin/main && echo on-main
   ```
10. **Publish the GitHub release** from the tag. Zenodo archives each GitHub release (not each tag) and gives it its own DOI, so a citation can name this exact version (`CITATION.cff`). Use the version's CHANGELOG section as the notes:
    ```bash
    notes=$(mktemp)
    awk '/^## \[<version>\]/{f=1; next} /^## \[/{f=0} f' CHANGELOG.md > "$notes"
    gh release create v<version> --verify-tag --title "openinterpretability <version> (protocol v<PROTOCOL_VERSION>)" --notes-file "$notes"
    ```
    Publishing the release also starts `.github/workflows/publish.yml`, which uploads the package to PyPI as `open-interpretability`. Check that its run passed (`gh run list --workflow publish.yml --limit 1`). If it failed on the version check, the tag doesn't match `openinterpretability.__version__`. If it failed on upload, check the trusted publisher on PyPI and the `pypi` environment's tag rule (`docs/development.md`, *Releases on PyPI*). Tell the user either way.
    Then check that Zenodo made the DOI: the release's Zenodo record appears under the repository's concept DOI after a few minutes. If it doesn't, the Zenodo GitHub integration is off. Tell the user, since only the repository owner can turn it on at zenodo.org.
11. **Close the milestone.** Move any open issues in `v<version>` to the next milestone, then close it:
    ```bash
    gh issue list --milestone v<version> --state open --json number --jq '.[].number' | xargs -I{} gh issue edit {} --milestone v<next>
    gh api -X PATCH repos/8W9aG/openinterpretability/milestones/<number> -f state=closed
    ```
    Publishing the release also starts the *Linear released* workflow. It moves every closed issue in `v<version>` to Linear's Released state. Check that its run passed: `gh run list --workflow linear-released.yml --limit 1`. If it failed, or the release was published before the workflow existed, run it by hand: `python3 scripts/issue_state.py --release v<version>`.
12. **Check the release before doing anything else.** The release isn't done, and no issue or other work starts, until each of these holds:
    ```bash
    gh pr view <number> --json state --jq .state                       # MERGED
    git merge-base --is-ancestor v<version> origin/main && echo on-main
    gh release view v<version> --json tagName,publishedAt,isDraft      # published, not a draft
    gh run list --workflow publish.yml --limit 1                       # PyPI: passed
    curl -s https://pypi.org/pypi/open-interpretability/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
    ```
    The last line must print `<version>`. Then check Zenodo's record for `v<version>`, as in step 10. Then publish the draft advisories of the security sweep's findings, whose fixed version is now on PyPI: `gh api -X PATCH repos/8W9aG/openinterpretability/security-advisories/<id> -f state=published`. If a step fails, fix it or report it to the maintainer, and stay on the release. A release PR left open, or a tag with no GitHub release, means there's no release.
