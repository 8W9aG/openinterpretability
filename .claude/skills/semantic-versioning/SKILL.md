---
name: semantic-versioning
description: Decide the next package version of openinterp under Semantic Versioning. Read the waiting changelog fragments and the changes since the last tag. Find the breaking changes (record schema, CLI, Python API, protocol), and choose a major, minor or patch bump with its reasons. Use in the release skill before the version is bumped, or when the user asks what the next version should be.
---

# Semantic versioning

The package version (`openinterp.__version__`) follows [Semantic Versioning](https://semver.org/). The `release` skill runs this skill before it bumps the version. The result is one version and the reasons for it, which go in the release PR.

## The rule

A change is **breaking** when it breaks one of these for a user of the package:

- **The record schema.** A field is removed or renamed, its type or meaning changes, or old records stop validating (`openinterp/core/schemas.py`).
- **The CLI.** A command or option is removed or renamed, or its default changes what it writes (`open-interpretability --help`).
- **The Python API.** A public name in `openinterp` is removed or renamed, or its signature changes in an incompatible way.
- **The protocol.** `PROTOCOL_VERSION` changes (`docs/PROTOCOL.md` §6), because the same evidence can then get another verdict.

Then:

| Version now | Breaking change | New feature, nothing breaking | Anything else |
| --- | --- | --- | --- |
| 0.x | minor (0.0.21 → 0.1.0) | patch | patch |
| 1.0 or later | major | minor | patch |

## Procedure

1. **Ask the script.** It reads the fragments in `changelog.d/` and the protocol version since the last tag:
   ```bash
   python3 scripts/changelog.py next
   ```
   It counts a change as breaking when there is a `removed` fragment, a bullet that starts with `**Breaking.**`, or a change to `PROTOCOL_VERSION`.
2. **Look for breaking changes that nobody marked.** A fragment's author can forget the marker, so check the changes themselves:
   ```bash
   last=$(git describe --tags --abbrev=0)
   git diff "$last" -- openinterp/core/schemas.py     # removed or renamed fields, new required fields
   git diff "$last" -- openinterp/cli.py openinterp/cli_*.py | grep -E '^[-+].*(@click\.(option|command|argument)|def )'
   git log --oneline "$last"..HEAD
   ```
   Check each removed or renamed field, option, command and public function. A new optional field or option is not breaking. If you find a breaking change, add `**Breaking.**` to the start of its fragment's bullet in the release branch, so that the CHANGELOG says so.
3. **Check the protocol.** If a change since the last tag changes the decision rule but `PROTOCOL_VERSION` did not change, stop and ask the maintainer. A protocol change needs its own PR with the bump (`docs/PROTOCOL.md` §6), never a quiet release.
4. **Decide.** Run `python3 scripts/changelog.py next` again after any fragment you marked. Use the version it prints, unless the maintainer named one. Give the version and every reason in the release PR body, and name each breaking change.

## Notes

- **A 0.x release can break things.** Semantic Versioning allows that, but say so: the minor bump tells users to read the CHANGELOG before they upgrade.
- **Version 1.0 is the maintainer's decision.** Never propose it on your own. It promises a stable record schema, CLI and API.
- **Records are write-once.** A schema change that stops old records from validating is breaking, even if no code outside this repository uses the field.
