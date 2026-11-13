# Contributing

Thank you for helping. This file is the short map. The detail is in the pages it links, so each rule has one place.

## Ways to contribute

- **Run an experiment.** Take one neuron, propose what it does, have a different model try to break the claim, and post the result, whatever it is. Start with the README's [Quickstart](README.md#quickstart). Then read [Contributing a result](docs/experiments.md#contributing-a-result) for how to claim a neuron, name the branch and get credit.
- **Review or revise an existing experiment.** `openinterp next` also draws follow-ups: re-reviews, revisions, opposite poles and sister neurons. Run the commands that `openinterp next --steps` prints.
- **Change the code, the skills or the docs.** Read [docs/development.md](docs/development.md) for the make targets, what CI checks and the code layout. Open work is in [GitHub issues](https://github.com/8W9aG/openinterpretability/issues). [Open work](docs/development.md#open-work) says how they are labelled.
- **Write a paper.** Analyses go in `papers/`. The `write-paper` skill (`.claude/skills/write-paper/SKILL.md`) gives the form.

## Pull requests

- A PR that changes only `wiki/` (an experiment) merges automatically when CI passes, fork PRs included.
- A PR that the maintainer opens from a branch of this repository also merges automatically when CI passes.
- A PR that changes anything else waits for a maintainer.
- Run `make check` before you push. If you change code, also run `make lint test`, and `make vale` for comments and docstrings.
- A change to the decision rule bumps `PROTOCOL_VERSION` ([docs/PROTOCOL.md](docs/PROTOCOL.md) §6). Mark the TODO item **(protocol)** and say so in the PR.
- For a change that a user or contributor will notice, add a changelog fragment in [changelog.d/](changelog.d/README.md). Don't edit `CHANGELOG.md` itself: each release gathers the fragments into it.
- The [PR template](.github/pull_request_template.md) has the checklist for an experiment.

## Style

- Write comments, docstrings and Markdown in ASD-STE100 Simplified Technical English. The `commenting` and `asd-ste100` skills give the rules, and `make ste` and `make vale` check them.
- Define each constant, threshold or list of names once, and read it everywhere else (the `single-source-of-truth` skill).
- Credit uses a public handle, never an email. Set `OPENINTERP_CONTRIBUTOR` to your GitHub username.

## Security

Don't report a security problem in a public issue or PR. [SECURITY.md](SECURITY.md) says how to report it, for example a leaked signing key or a way to get a false VERIFIED past CI.

## Licence

By contributing, you agree that your code is under the MIT licence ([LICENSE](LICENSE)). You also agree that your records and pages under `wiki/` are under CC-BY-4.0 ([wiki/LICENSE.md](wiki/LICENSE.md)). To cite the registry, use [CITATION.cff](CITATION.cff).
