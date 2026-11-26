# This project's surfaces

The `semantic-versioning` skill reads this file. It is this project's part of the skill (#6630).

The package version is `openinterpretability.__version__`.

## The surfaces

A change is **breaking** when it breaks one of these for a user of the package:

- **The record schema.** A field is removed or renamed, its type or meaning changes, or old records stop validating (`openinterpretability/core/schemas.py`).
- **The CLI.** A command or option is removed or renamed, or its default changes what it writes (`open-interpretability --help`).
- **The Python API.** A public name in `openinterpretability` is removed or renamed, or its signature changes in an incompatible way.
- **The protocol.** `PROTOCOL_VERSION` changes (`docs/PROTOCOL.md` §6), because the same evidence can then get another verdict. `scripts/changelog.py next` also counts this change as breaking.

## How each one changed

```bash
last=$(git describe --tags --abbrev=0)
git diff "$last" -- openinterpretability/core/schemas.py     # removed or renamed fields, new required fields
git diff "$last" -- openinterpretability/cli.py openinterpretability/cli_*.py | grep -E '^[-+].*(@click\.(option|command|argument)|def )'
```

## This project's own check

**The protocol.** If a change since the last tag changes the decision rule but `PROTOCOL_VERSION` did not change, stop and ask the maintainer. A protocol change needs its own PR with the bump (`docs/PROTOCOL.md` §6), never a quiet release.

**Records are write-once.** A schema change that stops old records from validating is breaking, even if no code outside this repository uses the field.
