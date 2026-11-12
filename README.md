# Open Interpretability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24150022.svg)](https://doi.org/10.5281/zenodo.24150022)

`openinterp` is a decentralized, agent-audited mechanistic interpretability registry and testing protocol for open-weight foundation models (Llama, Qwen, Mistral, …).

The registry is published at **[open-interpretability.com](https://open-interpretability.com/)**: every experiment, unit, model and paper, with search and the Parquet export.

Rather than cataloging the latents of learned SAE dictionaries, this project audits **native model weights directly**. Each entry in the registry is one *pole* (positive or negative half) of one MLP neuron. It starts from a working hypothesis, not an established finding. Goodfire's essay [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (Cammarata, 2026) argues that most apparent polysemanticity in native neurons comes from solvable tooling problems rather than superposition. Its technical companion paper has not been published, so the registry tests that claim one pole at a time rather than assuming it. A pole that resists a clean explanation is a result too.

## Quickstart

An experiment takes one neuron, proposes what it does, has a *different* model try to break that claim, and posts the result, whatever it is. A hypothesis that fails is a result too. The shortest path is to let Claude Code drive the whole loop. It needs no API keys: Claude Code's own subagents write the proposal and the review.

### What you need

| Tool | What it is for | Needed |
| --- | --- | --- |
| [Git](https://git-scm.com/) and a GitHub account | Fork the repo, and push your experiment as a pull request | Always |
| [Python](https://www.python.org/) 3.11+ | Runs `openinterp` and the model | Always |
| [uv](https://docs.astral.sh/uv/) | Installs the dependencies into `.venv` and runs the commands (`uv run …`) | Always |
| [GitHub CLI](https://cli.github.com/) (`gh`) | Opens the draft PR that claims a neuron. The planner also reads open PRs with it, to skip neurons that others have claimed | To contribute. Without it, the planner can't see claims, and you open the PR on github.com |
| [Claude Code](https://claude.com/claude-code) | Drives the experiment, and its subagents answer as proposer, reviewer and judge | Path A. Path B with no API keys too |
| An LLM API key (any [LiteLLM](https://docs.litellm.ai/) provider) | Proposer and reviewer through an API instead of Claude Code | Path B only |

Node.js is not needed: no step of the build uses it, and Claude Code's native installer doesn't need it either (only its npm install does). [Vale](https://vale.sh) is needed only for `make vale` when you change code.

A laptop CPU is enough. The default models are small (Qwen2.5-0.5B is about 1 GB, downloaded on first use). An 8 GB machine runs GPT-2 and Qwen2.5-0.5B, and Qwen2.5-1.5B needs 16 GB ([Memory](docs/experiments.md#memory)). The first scan also downloads about 1.5 GB of labelling models.

### 1. Set up (once)

```bash
gh repo fork 8W9aG/openinterpretability --clone   # or fork on GitHub and git clone your fork
cd openinterpretability
make install                                      # creates .venv with everything
uv run openinterp corpus --dataset pile-10k --first --ascii \
  --revision 127bfedcd5047750df5ccf3a12979a47bfa0bafa --out corpus.jsonl   # the registry's scan corpus
export OPENINTERP_CONTRIBUTOR=<your GitHub username>                       # to be credited (never an email)
```

### 2A. Let an agent run it (Claude Code)

Start Claude Code in the repo (`claude`) and ask:

```text
Run the next experiment and open a PR with the result.
```

The repo's skills load automatically. The agent draws an experiment with `openinterp next`, claims it with a draft PR and scans the neuron. Then one subagent proposes a hypothesis, another reviews it, and the agent posts the result. A scan takes about 10–15 minutes on CPU. Review the PR before marking it ready.

### 2B. Run it yourself

With API keys set for the proposer and the reviewer (e.g. `export ANTHROPIC_API_KEY=...` for the default proposer, `anthropic/claude-opus-5-5`, and `OPENAI_API_KEY` for the `openai/gpt-5` reviewer below):

```bash
uv run openinterp next --steps            # draws an experiment and prints every step as exact commands
uv run openinterp audit --model Qwen/Qwen2.5-0.5B --random --corpus corpus.jsonl
uv run openinterp verify --audit-file submissions/<…>.json --agent-model openai/gpt-5
```

`audit` scans the neuron and asks the proposer for a hypothesis. `verify` has the reviewer write probes, measures them on the model, and writes the experiment under `wiki/experiments/`.

### 3. Contribute it

Claim the neuron with a draft PR before the scan, so others don't draw it too (`next --steps` prints the commands). When it finishes, run `make check`, commit `wiki/` and mark the PR ready. A PR that changes only `wiki/` merges automatically when CI passes.

### More detail

- [docs/experiments.md](docs/experiments.md): other LLMs, answer files without API keys, the scan corpus and caches, and loops and parallel sessions. Also attention heads and dictionary latents, contributing and credit, the skills, definitions and the models.
- [docs/benchmark.md](docs/benchmark.md): the public benchmark for explanation tools, and how to score an explainer on it.
- [docs/development.md](docs/development.md): make targets, what CI checks on a pull request, record signing, the code layout, and the static site and deploy.
- [docs/PROTOCOL.md](docs/PROTOCOL.md): how verdicts are decided.
- [docs/moe.md](docs/moe.md): how neurons in mixture-of-experts models will be audited. It is a decision, not built yet.
- [CONTRIBUTING.md](CONTRIBUTING.md): the ways to contribute, and the rules for pull requests and style.
- [SECURITY.md](SECURITY.md): how to report a leaked signing key, a way to get a false VERIFIED past CI, or another security problem. Report it privately, not in an issue.

## Protocol

Every claim is empirical and falsifiable:

1. **Propose.** A submitting agent scans a corpus for the neuron pole's top-activating snippets. It then writes a hypothesis, says whether the pole is *retrospective* (reacts to tokens it has already read) or *lookahead* (prepares an upcoming token). It also supplies minimal-pair counterfactuals with predicted activation deltas.
2. **Review.** A reviewer agent (ideally, but not necessarily, a different model from the proposer) writes 10–20 adversarial probes. Most are hard negatives (near-misses the hypothesis says should *not* fire), plus some positive holdouts. Every probe and counterfactual is then run through the real model. The verdict comes from those measurements and not from the reviewer's opinion, so anyone with the weights can reproduce it:
   - balanced accuracy on hard negatives vs. positive holdouts, each gated separately, with near-duplicate probes counted once
   - confound control: the hypothesis must match or beat token-identity, position and never-active baselines
   - counterfactual Δ reproduction within a tolerance relative to the prediction
   - lookahead consistency (the claimed mechanism compared with KL(clean ‖ pole-ablated) on next-token distributions)
   - covariance: the most correlated same-layer neurons, and joint ablation with them to expose redundancy (recorded, not gated)
   - causal sufficiency (clean → corrupted patching) and necessity (corrupted → clean resample ablation), recorded but not yet gated. Downstream MLP outputs are frozen to mitigate self-repair (the hydra effect). Each score is ranked against same-layer neurons that change as much between the pair
3. **Publish.** Every experiment is compiled into a fully static site plus a Parquet registry, whatever its verdict. Null results are data too. Within each neuron, experiments are ordered by how *conclusive* their evidence is, in either direction (see `openinterp/builder/significance.py`). The output is hosted on Cloudflare R2 (zero egress) at [open-interpretability.com](https://open-interpretability.com/) and searched client-side with Pagefind and DuckDB-Wasm.

The full experimental protocol (roles, lifecycle, decision rule and reproducibility requirements) is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Licence

The code is under the MIT licence (`LICENSE`). The data (the records and pages under `wiki/` and the site built from them) is under CC-BY-4.0 (`wiki/LICENSE.md`). Corpus excerpts inside records keep their corpus's own terms, and model outputs are also subject to the model's terms.

## Citing

To cite the registry or the package, use [CITATION.cff](CITATION.cff). GitHub shows it as "Cite this repository". The concept DOI, [10.5281/zenodo.24150022](https://doi.org/10.5281/zenodo.24150022), names every version and resolves to the latest. Each release is archived on Zenodo with its own DOI, so a citation can name an exact version.

## Status

Early scaffold. `make sample-run` builds the site from six real experiments in `wiki/experiments/` (listed in the `Makefile`), imported with `openinterp wiki --import-dir`.
