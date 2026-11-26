---
name: semantic-versioning
description: Decide the next package version of the project under Semantic Versioning. Read the waiting changelog fragments and the changes since the last tag. Find the breaking changes on the surfaces that the project lists, such as its data schema, CLI, Python API or protocol. Then choose a major, minor or patch bump with its reasons. Use in the release skill before the version is bumped, or when the user asks what the next version should be.
---

# Semantic versioning

The project's package version follows [Semantic Versioning](https://semver.org/). The `release` skill runs this skill before it bumps the version. The result is one version and the reasons for it, which go in the release PR.

**Read [`project.md`](project.md) first.** It lists the project's public surfaces, the commands that show how each one changed since the last tag, and any check of the project's own. This file holds the rule and the steps, and no value of a project (#6630).

## The rule

A change is **breaking** when it breaks one of the project's surfaces (`project.md`) for a user of the package. Examples are a removed or renamed field, command, option or public function, a changed meaning, or old data that stops validating.

Then:

| Version now | Breaking change | New feature, nothing breaking | Anything else |
| --- | --- | --- | --- |
| 0.x | minor (0.0.21 → 0.1.0) | patch | patch |
| 1.0 or later | major | minor | patch |

## Procedure

1. **Ask the script.** It reads the fragments in `changelog.d/` since the last tag:
   ```bash
   python3 scripts/changelog.py next
   ```
   It counts a change as breaking when there is a `removed` fragment, or a bullet that starts with `**Breaking.**`. The project can add a signal of its own, which `project.md` names.
2. **Look for breaking changes that nobody marked.** A fragment's author can forget the marker, so check the changes themselves, with the commands in `project.md` and:
   ```bash
   last=$(git describe --tags --abbrev=0)
   git log --oneline "$last"..HEAD
   ```
   Check each removed or renamed surface. A new optional field or option is not breaking. If you find a breaking change, add `**Breaking.**` to the start of its fragment's bullet in the release branch, so that the CHANGELOG says so.
3. **Run the project's own checks** from `project.md`. One may tell you to stop and ask the maintainer.
4. **Decide.** Run `python3 scripts/changelog.py next` again after any fragment you marked. Use the version it prints, unless the maintainer named one. Give the version and every reason in the release PR body, and name each breaking change.

## Notes

- **A 0.x release can break things.** Semantic Versioning allows that, but say so: the minor bump tells users to read the CHANGELOG before they upgrade.
- **Version 1.0 is the maintainer's decision.** Never propose it on your own. It promises stable surfaces.
