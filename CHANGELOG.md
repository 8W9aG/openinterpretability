# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

The project has two version numbers:

- **Package version** (`openinterp.__version__`): the code. Bump it in `openinterp/__init__.py`, which `pyproject.toml` reads.
- **Protocol version** (`openinterp.core.schemas.PROTOCOL_VERSION`): the review decision rule. It is stored in every verdict and documented in `docs/PROTOCOL.md` §6. A protocol change always needs a package release too, but not the other way round.

## [Unreleased]

## [0.0.3] - 2026-10-18

Ships with protocol version **0.4**.

### Added

- `scripts/experiment-loop.sh` (`make loop`): runs experiments back to back in fresh headless Claude Code sessions, in a separate worktree reset to `origin/main` each time, until `--max` or until `claude` fails (for example at the usage limit). `--proposer` and `--reviewer` choose the LLMs; `--driver` chooses the Claude model driving the session. Logs go to `logs/loop/` (git-ignored).
- `NeuronAuditSubmission.model_mlp_width` (neurons per MLP layer), recorded by the proposer and exported to Parquet. It is optional, so existing records still validate.
- `--agent-model external/<name>`: any LLM, agent or person can act as proposer or reviewer by answering `exchange/<Agent>.prompt.md` (which includes the JSON schema) with `exchange/<Agent>.response.json`. No API keys are needed.

### Changed

- **Experiments, not neurons, are the unit of storage.** `audits/` is retired: each experiment's write-once `wiki/experiments/<slug>/<id>.json` is the only source of truth, so a neuron can have any number of experiments and parallel PRs on one neuron no longer collide. `verify` and `reevaluate` add a new experiment and never overwrite one (`--out` writes an optional extra copy). `openinterp check` drops the one-record-per-neuron rule, checks each record's path and page instead, and rejects records left in `audits/`. The planner derives each neuron's state from all of its experiments. `openinterp wiki --audits-dir` becomes `--import-dir`, and `check`/`next` lose `--audits-dir`.
- **Protocol 0.4:** `independent_review` is recorded but no longer gated by default (`ReviewPolicy.require_independent_review = False`).
- Site coverage now counts neurons rather than layers: model cards and pages show neurons studied and neurons with a verified pole, each out of the model's total neurons (layers × MLP width). Before, one experiment anywhere in a layer counted the whole layer as covered. Records without a width fall back to `KNOWN_MLP_WIDTHS` in `builder/tree.py`; when the width is unknown, the total shows as "?".

## [0.0.2] - 2026-10-18

Ships with protocol version **0.3**.

### Added

- README "Get started" guide for contributors: setup, API keys for an independent proposer and reviewer, building a Pile scan corpus, running an experiment with Claude Code or by hand, and claiming and posting it as a PR. `submissions/` and `corpus.jsonl` are now git-ignored.
- **Protocol v0.3: covariance and confounds.**
  - Near-duplicate probes cluster into single observations (`core/stats.py`). The probe-count gates, balanced accuracy and significance use these effective counts.
  - The scan (`harness/scan.py`) gathers in one pass the top examples, the most correlated same-layer neurons (`harness/covariance.py`) and a confound profile: token-identity and position R², frequency correlation and trigger tokens (`harness/confounds.py`).
  - New gate `beats_confound_baseline`: the hypothesis must match or beat token-identity, position and never-active baselines on the reviewer's probes.
  - Joint ablation of the pole with its correlated partners, reporting a redundancy ratio (recorded, not gated).
  - Re-measurements of an earlier probe set are marked as not independent, rank after the original, and are excluded from the most-conclusive list.
  - The home page estimates expected false discoveries among VERIFIED claims.
  - Neuron pages summarise both poles together.
  - The proposer sees the confound profile and correlated neurons.
- **Jacobian lens** (`harness/jlens.py`; Gurnee et al., 2026, arXiv:2607.15495).
  - Reads each pole's output direction through both the context-averaged Jacobian lens and the logit lens, and records `submission.vocabulary_lens` (both token lists, overlap, cos(w, J·w), kurtosis).
  - Estimates J·w from one perturbed forward pass per prompt and sign. Tests check it against exact autograd on Llama-style and GPT-2 models.
  - On Qwen2.5-0.5B the two lenses share 0/10 tokens up to the middle layers and converge only near the output.
  - Shown on the site and wiki, exported to Parquet, fed to the proposer; the vocabulary skill now defaults to it.
- **Rank-based and regime-aware correlation.**
  - Frequency correlation is Spearman's ρ (tie-aware, exact).
  - Partner neurons are found per *pole* by co-activation lift (`PartnerSearch`): a fixed-seed reservoir sample plus the target's top tokens, with chunked Spearman and Pearson kept for reference.
  - Partners are ranked and selected by the 95% Wilson lower bound of the lift, which the new `min_partner_lift` and `min_partner_coactive` policy fields govern.
  - Global Pearson missed real partners: on L11/N4255 every |r| was below 0.07, while co-activation lift was at least 10.
- Every experiment is published, null and superseded results included. The site follows Model → Layer → Neuron → Experiments, with model and layer index pages, and each neuron page lists all experiments on both poles.
- Experiments are ranked by *conclusiveness*: the posterior probability that balanced accuracy exceeds 0.6, folded to |2P − 1|, so strong evidence ranks high in either direction (`builder/significance.py`). The home page features the most conclusive results in both directions.
- A write-once experiment archive: `wiki/experiments/**/<id>.json` stores each experiment's full record next to its page. `openinterp check` validates it and rejects edits to it.

### Changed

- `openinterp build` reads the wiki archive (`--wiki-dir`) instead of `audits/`, and `--include-unverified` is gone because everything is published.
- `registry.parquet` has one row per experiment, with `experiment_id`, `is_latest`, `p_useful`, `conclusiveness` and `evidence_direction`.
- Neuron pages moved from `neurons/<slug>.html` to `models/<model>/L<layer>/N<idx>.html`.

## [0.0.1] - 2026-10-18

First release. Ships with protocol version **0.2**.

### Added

**Harness and protocol**

- Strict Pydantic v2 schemas for neuron identifiers, counterfactual pairs, lookahead profiles, causal metrics, submissions, verdicts and audit records.
- `NeuronHookRunner`: hooks on MLP down-projection inputs for Llama/Qwen/Mistral/Gemma, GPT-2, GPT-NeoX, GPT-J, OPT and Phi, with positive/negative pole isolation and corpus scanning.
- Activation patching: denoising, noising and resample ablation per pole, with optional freezing of downstream MLP outputs to mitigate self-repair.
- Attribution patching for whole layers in one backward pass.
- Lookahead profiling (pole-ablation KL and concentration) and vocabulary projection of neuron output directions.
- `ProposerAgent` and `ReviewerAgent` using LiteLLM and instructor (default model `anthropic/claude-opus-5-5`). Verdicts are computed deterministically by `evaluate_submission`.
- Protocol v0.2 (`docs/PROTOCOL.md`): a gated decision rule using balanced accuracy over hard negatives and positive holdouts, a counterfactual tolerance relative to the prediction, minimum probe counts and an independent-review gate. It replaced v0.1's weighted mean and absolute Δ tolerance of 0.35 before release, after v0.1 verified an over-broad hypothesis.
- Experiment provenance on submissions (model revision, corpus hash, selection method, seed) and the review policy stored on verdicts.
- CLI commands `audit` (including `--random --seed`), `verify`/`review` (which also accepts an existing audit record), `reevaluate`, `build`, `deploy`, `wiki`, `check` and `next`.

**Registry, wiki and site**

- Experiment wiki (`wiki/`): a write-once Markdown page per completed experiment, with per-neuron history pages and a home index. `verify` and `reevaluate` write pages automatically, `reevaluate` saves the superseded verdict before overwriting a record, and `openinterp wiki` backfills existing records.
- `openinterp next` and the `choose-next-experiment` skill: a randomised next-experiment planner (ε-greedy exploration of random neurons, Boltzmann sampling over follow-ups, decay for repeats, recency and open-PR claims). It prints a seed for reproducibility.
- Static site builder (Jinja2 + Tailwind) with per-layer coverage, leaderboards, Pagefind search and a DuckDB-Wasm SQL console. The registry is exported as `registry.parquet` plus content-addressed snapshots.
- The build is deterministic. There are no build timestamps, orderings are total, and stale pages are removed, so unchanged records rebuild byte-for-byte.
- Cloudflare R2 delta sync with per-file cache policies:
  - content-addressed files upload first and HTML last;
  - `pagefind/` only syncs when a source file changed;
  - `data/snapshots/` is never deleted;
  - an optional Cloudflare cache purge covers only the changed URLs.

**Project infrastructure**

- GitHub Actions CI: lint, a 500-line limit on source and documentation files, tests, and `openinterp check` (audit/wiki pairing, stale indexes and write-once pages) on every PR.
- Hourly deploy workflow that rebuilds from `main` and delta-syncs to R2. It is skipped until the `R2_BUCKET` variable is set.
- A PR template with an experiment checklist.
- Ten Claude Code research skills in `.claude/skills/`, each grounded in the interpretability literature.
- First real audit: Qwen2.5-0.5B L11/N4255+, currently NEEDS_REVISION.
