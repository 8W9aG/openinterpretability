# This project's release

The `release` skill reads this file. It is this project's part of the skill (#6630).

This release refreshes the analyses that are too slow to run for each experiment. These are the sister search, the registry coverage, the circuit ids and the concept maps.

The CHANGELOG's header explains the two version numbers. The **package version** changes every release. The **protocol version** (`PROTOCOL_VERSION` in `openinterpretability/core/schemas.py`) changes only in the PR that changes the decision rule, never here. A release only reports it.

## Decisions

The maintainer decided these rules of the generic skill for this project:

- 2026-11-15: no release goes out with a known security finding open (the security sweep), and the agent merges the release PR itself.
- 2026-11-19: each release runs a fresh CodeQL scan, not only the daily one.

## Topics

The search for new work (*Before the release*, step 4) covers native-neuron interpretability, explanation scoring and automated interpretability. The workstreams are research, engineering, the site and docs.

## Trust boundaries

The security sweep also covers signing and uploads to Hugging Face.

## Before the release

Do these after the generic steps 1 to 4, in this order.

1. **Land lingering experiment PRs.** The maintainer asked on 2026-11-15 that a release try to merge the experiment PRs that were left behind. Take the open ones that are ready (not drafts) and more than 12 hours old. Each must come from a branch of this repository, opened by the maintainer's account (`8W9aG`, which the experiment loops use):
   ```bash
   cutoff=$(date -u -v-12H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '12 hours ago' +%Y-%m-%dT%H:%M:%SZ)
   gh pr list --state open --limit 200 --json number,headRefName,createdAt,isDraft,isCrossRepository,author \
     --jq ".[] | select(.headRefName | startswith(\"experiment/\")) | select(.isDraft | not) | select(.isCrossRepository | not) | select(.author.login == \"8W9aG\") | select(.createdAt < \"$cutoff\") | .number"
   ```
   A branch name proves nothing, since anyone can name a fork's branch `experiment/…`. So leave every other PR for a person. Then check each one's files. Merge it only when every changed path, renames included, is under `wiki/`. That is the rule `auto-merge.yml` uses:
   ```bash
   gh api "repos/$REPO/pulls/<number>/files" --paginate \
     --jq '.[] | .filename, (.previous_filename // empty)' | grep -v '^wiki/'   # must print nothing
   ```
   Leave drafts alone. A draft is a run that hasn't finished, and it may still be going. For each ready one that passes both checks:
   - **Merge it when its checks pass.** Bring it up to date with `gh pr update-branch <number>`, and wait for every check on the new head. Then run `gh pr merge <number> --squash --match-head-commit <sha>`. Remove the branch only once the PR is MERGED.
   - **When a check fails, leave it open.** Don't change the record, because records are write-once. Read the log. Comment what failed, with a bold marker, unless an earlier comment already says so.

   Never use `--admin`. List the merged and the failing PRs in the release PR.
2. **Open an alphaXiv issue for Muse.** The maintainer decided on 2026-11-19 that each published paper goes to the alphaXiv AI Track (#5259). On 2026-11-22 the maintainer gave the upload to Muse (`docs/muse.md`). Find the papers published since the last tag, and the published papers that got a new addendum:
   ```bash
   git diff --name-only v<previous> -- 'papers/*/README.md'
   ```
   For each changed paper whose status line says *Published*, open its issue:
   ```bash
   uv run python scripts/papers.py submit <slug> --issue
   ```
   The issue has the label `muse`, which tells Muse to act on it. Only the maintainer's own issues, or issues with the maintainer's thumbs up, may have that label (`docs/muse.md`). Its body gives the steps and each field of alphaXiv's form. An addendum is a new alphaXiv version, so it gets a new issue. The command doesn't open an issue twice. Never upload a paper yourself, and never work a `muse` issue: `next-issue` skips them. List the issues in the release PR.

## Refresh

1. **Check the corpus.** The sister rows record the corpus's SHA-256, and a run must use the same corpus, or the new rows aren't comparable with the old ones:
   ```bash
   shasum -a 256 corpus.jsonl   # must be ded48cfc… (docs/experiments.md, "The scan corpus")
   head -1 docs/analyses/sister-neurons.jsonl | grep -o '"corpus_sha256":"[0-9a-f]*"'
   ```
   If `corpus.jsonl` is missing, rebuild it with the README's `open-interpretability corpus` command. If the two hashes differ, stop and ask the user: a new corpus means rerunning every row, not resuming.
2. **Refresh the sister rows.** Run this on a machine with a GPU (MPS or CUDA). On CPU the passes against Qwen2.5-1.5B take hours. Run it in the background. A run with nothing new finishes in seconds.
   ```bash
   uv run open-interpretability sisters --device auto
   ```
   It searches every ordered pair of the configured models (`[tool.openinterpretability] models`) plus `SIBLINGS`. It skips every pole that already has a row for that model. So it only searches the poles VERIFIED since the last run, and any pair that a newly configured model adds. Each pair prints `N VERIFIED poles of A against B`. Then one search covers every pair, in four stages (`source:`, `forward:`, `matches:` and `backward:` positions). Each stage reads each model once, with one model loaded at a time. A pair with nothing new prints nothing. If no accelerator is available, skip this step and say so in the PR body. The next release picks the poles up.
3. **Record what changed.** If the run wrote rows (`git diff --stat docs/analyses/sister-neurons.jsonl`), count them:
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
   Then update the shared concept map (#4785, #4586): `make maps`. It runs `open-interpretability map` once, on CPU, for every configured model. Old poles keep their positions and regions, and the new ones are placed among their nearest neighbours. Don't pass `--refit` in a release: a refit moves every point, so it needs its own PR. Commit `data/concept-maps/shared.json` in the release PR. In the PR body, give how many poles it gained, and each model's region coverage:
   ```bash
   python3 -c 'import json,subprocess; f="data/concept-maps/shared.json"; n=json.load(open(f)); o=json.loads(subprocess.run(["git","show","HEAD:"+f],capture_output=True,text=True).stdout or "{}"); print(n["poles"]-o.get("poles",0), "new poles"); [print(m, v["poles"], v["coverage"]) for m,v in n["coverage"]["models"].items()]'
   ```

The release commits these files: `docs/analyses/sister-neurons.jsonl`, `docs/analyses/registry-coverage.jsonl`, `data/circuits` and `data/concept-maps`. The PR body gives the sister refresh (rows and sisters added, or why it was skipped) and what each concept map gained.

## Check

Run `make check-all` too. Then run the benchmarks on CI. They no longer run on a local machine (#3808):

```bash
gh workflow run benchmarks.yml --ref main
gh run watch "$(gh run list --workflow benchmarks.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

The workflow measures the peak memory of a scan and a review on GPT-2 and Qwen2.5-0.5B, and the time and memory of fixed GPT-2 workloads. It compares them with the median of the recent runs in `docs/analyses/benchmark-ledger.jsonl`. Then it adds the run to that ledger through a PR that merges by itself (#4011). If it fails, stop and report the regression, as for a failed test. The ledger PR still merges, so the regression stays on record.

## The release's line and title

The release's line is `Ships with protocol version **<PROTOCOL_VERSION>**.` If the protocol version changed since the last tag (`git diff v<previous> -- openinterpretability/core/schemas.py | grep PROTOCOL_VERSION`), say so in that line: "Ships with protocol version **0.9** (was 0.8)." The section's entries explain why. The PR body also says whether the protocol version changed.

The title's suffix is ` (protocol v<PROTOCOL_VERSION>)`, as in `Release 0.6.1 (protocol v0.9)`.
