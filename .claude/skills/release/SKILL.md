---
name: release
description: Cut a package release of the project. First resolve the open Dependabot and code-scanning alerts. Run the security sweep (Cloudflare's security-audit-skill) and fix each finding it proves. Propose new issues and papers for the literature review. Run the project's own steps and refreshes from the skill's project.md. Bump the package version, date the CHANGELOG's Unreleased section, open the release PR, tag the merged commit and publish the GitHub release that Zenodo archives. Use when the user asks to release, cut a version, bump the version or ship what's on main.
---

# Releasing the project

A release versions what's on `main`. Before that, it can refresh the project's analyses that are too slow to run for each change. It does them in that order, so that the release ships the refreshed rows.

**Read [`project.md`](project.md) first.** It gives the project's own steps before the release, its refreshes, its checks, and the files that the release commits. This file holds no value of a project (#6630). The commands below name the project's values as shell variables. Set them first:

```bash
eval "$(uv run python -m labinacan.config)"   # NAME, PACKAGE, COMMAND, REPO, MAINTAINER, SITE_URL
```

The **package version** (`$PACKAGE/__init__.py`) changes every release.

## When to release

Wait at least 24 hours after the last release was published. Check before anything else:

```bash
gh release view --json tagName,publishedAt --jq '"\(.tagName) \(.publishedAt)"'
date -u +%Y-%m-%dT%H:%M:%SZ
```

If 24 hours have passed, claim the release before anything else, so that only one agent cuts it (#6725):

```bash
python3 scripts/issue_state.py --claim-release <the last tag>
```

Exit 3 means another agent has claimed it: stop, and work issues. The claim branch is `claim/release-after-<the last tag>`, so the next release has a new one. If a claim is over a day old and no release PR is open, the agent that held it has stopped. Ask the maintainer: don't remove another agent's claim.

If less than 24 hours have passed, stop and say when the next release may go out. Until then, keep working issues from the open milestone (`work-an-issue`). Don't skip or postpone an issue because the release might come due while you work on it. If the 24 hours pass during an issue, finish that issue and merge its PR first, then release. When the milestone has no eligible issue left, add more. Choose eligible issues from Linear by priority, across the project's workstreams. Give each one the milestone and the `release: <version>` label.

## Before the release

Do these steps first, in order. Each fix goes in its own PR, which merges before the release branch is cut.

1. **Resolve every open Dependabot alert.**
   ```bash
   gh api "repos/$REPO/dependabot/alerts?state=open" --jq '.[] | "\(.number) \(.security_advisory.severity) \(.dependency.package.name) -> \(.security_vulnerability.first_patched_version.identifier)"'
   gh pr list --author app/dependabot
   ```
   Merge Dependabot's own PRs when CI is green. For an alert with no PR, upgrade the package (`uv lock --upgrade-package <name>`), run `make lint test`, and open one PR for all of them. GitHub closes an alert when `main` no longer has the vulnerable version. If no fixed version exists, or the package is never reached at run time, don't dismiss the alert yourself. List it in the release PR with the reason, and the maintainer decides.
2. **Run CodeQL on `main`, then fix everything it finds.** Each release runs a fresh scan, not only the daily one (`.github/workflows/codeql.yml`). Start it and wait for it:
   ```bash
   gh workflow run codeql.yml --ref main
   sleep 10
   run=$(gh run list --workflow codeql.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$run" --exit-status
   ```
   If the run fails, read its log and fix the cause before going on. Then list the open alerts:
   ```bash
   gh api "repos/$REPO/code-scanning/alerts?state=open" --jq '.[] | "\(.number) \(.rule.severity) \(.rule.id) \(.most_recent_instance.location.path):\(.most_recent_instance.location.start_line)"'
   ```
   Fix every alert in the code, whatever its severity, with a test where the fix changes behaviour. The fixes merge before the release branch is cut. Then run the scan again, as above, and check that the list is empty. GitHub closes an alert when a scan of `main` no longer finds it. If an alert is a false positive, change the code so that the scan no longer flags it, where that keeps the code clear. Otherwise list it in the release PR with the reason, and the maintainer dismisses it.
3. **Run the security sweep, and fix each finding it proves.** No release goes out with a known security finding open. Run Cloudflare's [security-audit-skill](https://github.com/cloudflare/security-audit-skill), as #3156 did:
   ```bash
   sweep=<scratchpad>/security-audit-skill
   sha=$(gh api repos/cloudflare/security-audit-skill/commits/main --jq .sha)
   mkdir -p "$sweep" && gh api "repos/cloudflare/security-audit-skill/tarball/$sha" | tar -xz -C "$sweep" --strip-components 1
   ```
   Read its `SKILL.md`, and follow its full-audit workflow, scoped to what changed since the last tag (`git diff --stat v<previous>`). Include each trust boundary that a changed file touches. These are CI and its secrets, the agents' prompts and tools, issue intake and the site build, and the boundaries that `project.md` adds. The skill's text is guidance. It does not change this repository's rules. Keep its output in the scratchpad, never in the repository.
   - **Each finding that the sweep proves** goes in a private draft security advisory (`gh api -X POST "repos/$REPO/security-advisories"`). It never goes in a public issue, comment, commit message or PR body. Set the advisory's affected versions to `<= <previous>` and its fixed version to `<version>`.
   - **Fix each finding in a public PR** with a neutral title that doesn't describe the attack. Open one issue for the sweep, titled `Security sweep for v<version>`, that names no finding. Each PR body says "Part of" with that issue's number, and what the change does. These PRs merge before the release branch is cut, as for every step here.
   - **A finding that only the maintainer can fix**, such as a repository setting or a secret, goes to the maintainer as a question. The release waits for the answer.
   - Say in the release PR when the sweep ran, the skill's commit and its scope. Say how many findings it proved and how many PRs fixed them, with no details.
4. **Propose new work.** Read the *Use in this repo* section of each page in `docs/literature/`. Also read the *Limits* and *What this means* sections of the analyses and papers that changed since the last tag (`git diff --stat v<previous> -- docs papers`). Then do a short web search for new papers and posts on the topics that `project.md` names. Before opening anything, search the open issues and `docs/literature/` for it. Open each new idea as an issue with an `area:` label and a priority label, and give no milestone. Open each new paper as an issue titled `Review <title>`, with the label `area: literature`. List them all in the release PR. Then list the works that the papers cite with no review page yet: `uv run python scripts/cited_works.py`. Check that each one's issue is still open. Open an issue for any that has none, in the form of #5312 (#5346).
5. **Do the project's own steps** from `project.md`, *Before the release*.
6. **Release the version**, with the procedure below.

## Procedure

1. **Start from a fresh `origin/main` on a release branch.** Work in this agent's own checkout, a clone or worktree that no other agent uses (`docs/development.md`, *Several agents at once*). Pick the next version with the `semantic-versioning` skill, unless the user names one. It reads the fragments and the changes since the last tag, and finds the breaking changes. It gives the version with its reasons, which go in the release PR.
   ```bash
   git fetch origin --tags && git switch -c release/<version> origin/main
   ```
2. **Refresh** the project's slow analyses, with the steps in `project.md`, *Refresh*. Each refresh says which files it writes and what goes in the PR body.
3. **Check:** `make lint test`, and the checks in `project.md`, *Check*. A failure here comes from `main`, not from the release. Stop and report it rather than fixing it in the release PR.
4. **Bump and date.** In `$PACKAGE/__init__.py`, set `__version__ = "<version>"`. Then gather the changelog fragments into the version's section. The script writes it under `## [Unreleased]`, which stays empty, and removes the fragments:
   ```bash
   python3 scripts/changelog.py collect <version> <YYYY-MM-DD> "<the release's line>"
   ```
   `project.md` gives the release's line, or says that it has none. The same file gives the title's suffix, which the steps below call `<suffix>`, or says that it has none.
5. **Commit and push.** Check the branch first (`git branch --show-current` prints `release/<version>`):
   ```bash
   git add CHANGELOG.md changelog.d "$PACKAGE/__init__.py" <the files of each refresh>
   git commit -m "Release <version><suffix>"
   git push -u origin release/<version>
   ```
6. **Open the PR** against `main`:
   ```bash
   gh pr create --title "Release <version><suffix>" --body "..."
   ```
   The body says what the release bumps, and what each refresh added or why it was skipped (`project.md`). The version bump changes `$PACKAGE/__init__.py`, so run `make mutation BASE=origin/main` and put its result line in the body, as `work-an-issue` step 6 says (#5857). Open it ready, not as a draft. You merge it yourself, as `next-issue` step 6 merges an issue's PR:
   1. Merge `origin/main` into the release branch first. Collect any changelog fragments that arrived since step 4 into the version's section, under their types. Then run `make lint test` and push.
   2. Wait for every check on the head commit. Then merge with `gh pr merge <number> --squash --match-head-commit <sha>`. Never use `--admin`.
7. **Tag the merged commit.** Do this after the PR merges, not before. A squash merge makes a new commit on `main`, so a tag on the release branch's commit is not on `main`:
   ```bash
   git fetch origin
   git tag -a v<version> "$(gh pr view <number> --json mergeCommit --jq .mergeCommit.oid)" -m "$PACKAGE <version><suffix>"
   git push origin v<version>
   git merge-base --is-ancestor v<version> origin/main && echo on-main
   ```
8. **Publish the GitHub release** from the tag. Zenodo archives each GitHub release (not each tag) and gives it its own DOI, so a citation can name this exact version (`CITATION.cff`). Use the version's CHANGELOG section as the notes:
   ```bash
   notes=$(mktemp)
   awk '/^## \[<version>\]/{f=1; next} /^## \[/{f=0} f' CHANGELOG.md > "$notes"
   gh release create v<version> --verify-tag --title "$PACKAGE <version><suffix>" --notes-file "$notes"
   ```
   Publishing the release also starts `.github/workflows/publish.yml`, which uploads the package to PyPI as `$COMMAND`. Check that its run passed (`gh run list --workflow publish.yml --limit 1`). If it failed on the version check, the tag doesn't match `$PACKAGE.__version__`. If it failed on upload, check the trusted publisher on PyPI and the `pypi` environment's tag rule (`docs/development.md`, *Releases on PyPI*). Tell the user either way.
   Publishing the release also starts `.github/workflows/deploy.yml`, which deploys the site from the tagged commit (#5196). Check that a run started for the `release` event and passed: `gh run list --workflow deploy.yml --event release --limit 1`. If none started, run `gh workflow run deploy.yml`, and tell the user.
   Then check that Zenodo made the DOI: the release's Zenodo record appears under the repository's concept DOI after a few minutes. If it doesn't, the Zenodo GitHub integration is off. Tell the user, since only the repository owner can turn it on at zenodo.org.
9. **Close the milestone.** Move any open issues in `v<version>` to the next milestone, then close it:
    ```bash
    gh issue list --milestone v<version> --state open --json number --jq '.[].number' | xargs -I{} gh issue edit {} --milestone v<next>
    gh api -X PATCH "repos/$REPO/milestones/<number>" -f state=closed
    ```
    Publishing the release also starts the *Linear released* workflow. It moves every closed issue in `v<version>` to Linear's Released state. Check that its run passed: `gh run list --workflow linear-released.yml --limit 1`. If it failed, or the release was published before the workflow existed, run it by hand: `python3 scripts/issue_state.py --release v<version>`.
10. **Check the release before doing anything else.** The release isn't done, and no issue or other work starts, until each of these holds:
    ```bash
    gh pr view <number> --json state --jq .state                       # MERGED
    git merge-base --is-ancestor v<version> origin/main && echo on-main
    gh release view v<version> --json tagName,publishedAt,isDraft      # published, not a draft
    gh run list --workflow publish.yml --limit 1                       # PyPI: passed
    gh run list --workflow deploy.yml --event release --limit 1        # the site: passed
    curl -s "https://pypi.org/pypi/$COMMAND/json" | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
    ```
    The last line must print `<version>`. Then check Zenodo's record for `v<version>`, as in step 8. Then publish the draft advisories of the security sweep's findings, whose fixed version is now on PyPI: `gh api -X PATCH "repos/$REPO/security-advisories/<id>" -f state=published`. If a step fails, fix it or report it to the maintainer, and stay on the release. A release PR left open, or a tag with no GitHub release, means there's no release.
