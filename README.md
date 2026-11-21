![Open Interpretability](openinterpretability/builder/brand/logo.svg)

# Open Interpretability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24150022.svg)](https://doi.org/10.5281/zenodo.24150022)
[![PyPI](https://img.shields.io/pypi/v/open-interpretability.svg)](https://pypi.org/project/open-interpretability/)

An open lab, run with agents, that explains the native neurons and attention heads of open-weight language models and publishes each result at **[open-interpretability.com](https://open-interpretability.com/)**.

## Before you start

You need [Git](https://git-scm.com/), [Python](https://www.python.org/) 3.11+, [uv](https://docs.astral.sh/uv/) and [Claude Code](https://claude.com/claude-code). A laptop CPU is enough.

## Run an experiment

```bash
gh repo fork 8W9aG/openinterpretability --clone
cd openinterpretability
claude "/loop /land-experiment"
```

Each experiment draws a neuron and proposes what it does. A different model then tries to break that claim, and the result goes in a pull request. The loop runs one experiment after another until you stop it.

## Explain an answer

```bash
uv run open-interpretability explain "The sun rises in the"
```

It lists the units that carry the model's answer, says which of them the registry explains, and lists the experiments that would explain the rest. It reads `corpus.jsonl`, which [The scan corpus](docs/experiments.md#the-scan-corpus) says how to build.

## More

- [docs/experiments.md](docs/experiments.md): what you need, other ways to run, API keys, and every option.
- [docs/PROTOCOL.md](docs/PROTOCOL.md): how a verdict is decided.
- [docs/development.md](docs/development.md): installing the package, the code, CI and the site.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md): how to contribute, and how to report a security problem.
- [CITATION.cff](CITATION.cff): how to cite. The code is under [MIT](LICENSE), and the records under [CC-BY-4.0](wiki/LICENSE.md).
