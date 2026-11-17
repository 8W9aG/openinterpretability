# Changelog fragments

Each change adds one file here, not an entry in `CHANGELOG.md`. Two pull requests never edit the same file, so they never conflict.

- Name: `<issue>.<type>.md`, for example `3490.changed.md`. Use a short slug when no issue fits. The type is `added`, `changed`, `deprecated`, `removed`, `fixed` or `security`.
- Content: one or more bullets in the form `CHANGELOG.md` uses. Start with a bold summary, then the files in brackets and the issue number.
- **Breaking changes.** Start the bullet with `**Breaking.**` when the change breaks the record schema, the CLI or the Python API. A `removed` fragment and a change to `PROTOCOL_VERSION` count as breaking without the marker. `python3 scripts/changelog.py next` reads these to propose the next version.
- At each release, `python3 scripts/changelog.py collect <version> <date> "<intro>"` writes the version's section from these files and deletes them (the `release` skill).

`make lint` runs `python3 scripts/changelog.py check`. It fails on a badly named or empty fragment, and on any entry in `CHANGELOG.md`'s Unreleased section.
