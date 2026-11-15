# Security audit, November 2026

This page gives the scope and the outcome of the security audit of issue #3156. It gives no details of how each finding worked. Those details are in private GitHub security advisories, which become public once version 0.0.21 is on PyPI.

## Method

- **Tool.** Cloudflare's [security-audit-skill](https://github.com/cloudflare/security-audit-skill), at commit `c1c8a8c`, used read-only.
- **Scope.** The maintainer chose a scoped audit (option 1b on #3156). The audit split the repository into seven units, one for each trust boundary:
  1. CI on pull requests, and the secrets its jobs hold.
  2. Auto-merge and the signing of records.
  3. Releases, PyPI publishing and the scheduled workflows.
  4. The agents' prompts and tools.
  5. How agents take in issues and comments.
  6. The site build and its deploy.
  7. Data handling: the Hugging Face datasets, `exchange/` and the removal of personal data.
- **Steps.** Agents found candidate findings in each unit. Then a separate agent tried to confirm or refute each one from the source.
- **Date.** 2026-11-14 to 2026-11-15.

## Outcome

| Severity | Findings | Fixed |
|---|---|---|
| High | 4 | 4 |
| Medium | 3 | 3 |
| Low | 10 | 10 |
| **All** | **17** | **17** |

The verifiers confirmed 15 findings. Two more were left as "needs validation": the verifier could not settle them from the source alone. Both are fixed too.

Each fix is a public pull request. Its title says what the change does, not what the finding was:

- High: #3917, #3919, #3920, #3921.
- Medium and low: #3927, #3928, #3929, #3930, #3933, #3934, #3939, #3940, #3942, #3944, #3947.

## What changed, by area

- **CI and auto-merge.** Code from a pull request runs with no write token. A maintainer's pull request auto-merges only when the maintainer made each push. Someone else's pull request auto-merges only when it adds new files under `wiki/`. Every action in a workflow is pinned to a commit.
- **Records.** Signatures are write-once, like records. Continuous integration (CI) loads only model weights from a commit on the model's `main` branch. An erratum can't set aside a new record.
- **Agents.** Text that agents read from the wiki is fenced. Only the maintainer's comments count as answers. An approval covers an issue as it was when the maintainer approved it. The reviewer's isolation hook ignores letter case.
- **Data.** Evidence blobs reach the public dataset only after a merge. `exchange.py` stages only real names, and never the benchmark's private files. The site removes personal data from erratum notes and paper files too.
- **Releases.** PyPI uploads use trusted publishing from a protected environment, with no stored token.

## What this does not show

- The audit read the source. It did not attack a running system. It did not look at the settings of GitHub, PyPI, Hugging Face or Cloudflare beyond what the source shows.
- The scope was the repository as it was on 2026-11-14. Each release now runs the same audit on what changed since the last release (`.claude/skills/release/SKILL.md`, step 4).
