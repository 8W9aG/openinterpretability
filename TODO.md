# TODO

Open work, roughly in priority order. Move items to `CHANGELOG.md` under `[Unreleased]` when they land. Items marked **(protocol)** change the decision rule and must bump `PROTOCOL_VERSION` (`docs/PROTOCOL.md` §6).

## Protocol and science

- [ ] Rescan L11/N4255+ so it gets a confound profile and correlated neurons (protocol v0.3); its current submission predates them and fails `beats_confound_baseline`.
- [ ] Re-audit Qwen2.5-0.5B L11/N4255+ with ≥ 512-token windows, a narrowed hypothesis ("*including* / *e.g.* after a category noun") and an independent reviewer model.
- [ ] Scan both poles by default and show them together on neuron pages.
- [ ] Positive control: audit known induction heads or other well-documented units and confirm the protocol VERIFIES them.
- [ ] **(protocol)** Decide whether steering predictions and simulation scores should become gates. Both need a pre-registered prediction format first.
- [ ] Report registry-wide statistics (share VERIFIED, share dead) over `selection = "random"` records only.
- [ ] Record dead neurons found by random selection instead of discarding them.

## Pipeline

- [ ] `openinterp audit`: configurable window length and stride, and Hugging Face dataset input (`--dataset NeelNanda/pile-10k`).
- [ ] Faster scans: MPS/CUDA batching and caching activations to disk for re-use across neurons in a layer.
- [ ] A `supersedes` field on submissions to link revisions to the record they replace.
- [ ] Record the human contributor behind each experiment, not just the agent model: an optional `contributor` (a public handle, e.g. GitHub username, never an email) on the submission and on the verdict, so proposer and reviewer can differ. Fill it from a `--contributor` flag or `OPENINTERP_CONTRIBUTOR`, falling back to nothing rather than reading git config silently. Export it to Parquet and add a contributor leaderboard next to the home page's "Submitting agents" table (experiments, VERIFIED, well-probed rejections, reviews done), so people get credit as well as models.
- [ ] A scheduled hourly build and deploy (CI cron), as described in the vision.
- [ ] Review [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for ideas or reuse: its standardised hook points across many architectures (compare with `openinterp/harness/hooks.py`), activation caching (for faster scans), patching utilities (for the causal metric), attention-head hooks (for `ComponentIdentifier`) and weight processing such as LayerNorm folding (which changes what a neuron's weights look like). Decide whether to adopt it as a dependency, borrow ideas, or keep the current Hugging Face hooks.
- [ ] Branding for the generated site and repo: a logo and wordmark to replace the `⊕` placeholder in `base.html`, a favicon, a colour palette and type defined once in the Tailwind config (currently ad hoc emerald/slate), Open Graph/social preview images for shared links, and the same mark on the README and GitHub social preview.
- [ ] An About page in the generated site (`openinterp/builder/templates/about.html`, rendered by `compiler.py` and linked from the `base.html` header): the project's premise, how the proposer/reviewer protocol decides a verdict (linking `docs/PROTOCOL.md` and the current `PROTOCOL_VERSION`), what VERIFIED does and doesn't mean, how to contribute, and the licence.
- [ ] Google Colab guide (a `docs/colab.md` page plus a ready-to-run notebook with an "Open in Colab" badge in the README): install from git (or PyPI once published) without clobbering Colab's CUDA torch, pick a GPU runtime and `--device cuda`, store API keys and `HF_TOKEN` in Colab secrets (or use the two-step answer files with no keys), build the scan corpus, note which models fit on a free T4 and when to use `--dtype bfloat16`/`float16`, and save results by pushing a branch or copying `wiki/` to Drive before the runtime resets.
- [ ] Publish `openinterp` to PyPI so `pip install openinterp` (or `uvx openinterp`) runs the whole pipeline without cloning the repo. Check the name is free on PyPI; confirm the wheel ships the templates and `data/sample_corpus.txt`; move `pagefind[bin]` out of the dev group (or into an extra) so `openinterp build` works from an install; let users choose a CPU/CUDA torch; make commands that assume the repo layout (`submissions/`, `wiki/`, `exchange/`) take paths or work from any directory; and add a release workflow that builds and publishes on a version tag via PyPI trusted publishing.

## Merge verification

At the moment CI's "Experiment records" job (`openinterp check`) only confirms that a PR adds a valid JSON record and its wiki page. Nothing in CI re-runs the experiment, so a record with edited activations or a hand-written verdict would pass. `openinterp reproduce --base-ref origin/main` now does that check (docs/PROTOCOL.md §5), and CI's "Reproduce records" job runs it on every pull request, on GitHub-hosted CPU runners with cached weights and a model allow-list. That is enough while volume is low. The items below are for when it isn't: larger models, many PRs, or runner minutes adding up.

- [ ] Remote worker: move "Reproduce records" to a GPU (or large CPU) machine that pulls jobs from a queue, keeps a warm cache of model weights, and runs one job at a time per device. No secrets or repo write access on the worker; it only needs to read the record and report a result.
- [ ] Queue and cost controls: a submission from CI only enqueues a job (record path, commit SHA, record hash). Deduplicate by record hash, cap jobs per PR and per author per day, set a per-job timeout and a monthly budget, and refuse jobs once the budget is spent rather than scaling up.
- [ ] Treat records as untrusted input: run only allow-listed models (no arbitrary Hugging Face repos, never `trust_remote_code`), bound probe count and length, and run each job in a sandbox with no network beyond the model cache.
- [ ] Report back to GitHub as a commit status or check run on the PR's head SHA ("Reproduction: passed / failed / queued"), with a link to the log, and make it a required check for `main` once it is reliable.
- [ ] Re-queue automatically when a PR's record changes, and let maintainers re-run or skip a job by hand (e.g. a label) for records whose model the worker can't load yet.

## Visualisations

The site currently shows one chart (the activation histogram on experiment pages); everything else is tables and token chips. Keep new charts as static SVG rendered at build time, readable in light and dark mode, with the numbers available as a table or in the Parquet export. Some need extra data stored on records; check that before building them.

- [ ] Token heatmaps of the top-activating examples: the text with each token shaded by pole strength, the peak token marked, hover for the exact value (the standard Neuroscope-style view).
- [ ] Probe results chart on each experiment: every positive, hard-negative and holdout probe as a dot on one activation axis, with the reviewer's threshold drawn as a line, so you can see at a glance why it passed or failed.
- [ ] Causal effects chart: ablation/patching scores per probe pair as bars, next to the baseline they must beat (`beats_confound_baseline`).
- [ ] Vocabulary projection bars: the top and bottom tokens the neuron promotes and suppresses (Jacobian and logit lens), replacing the current token lists.
- [ ] Steering samples: the same prompt continued with the neuron clamped high, at baseline and ablated, side by side.
- [ ] Layer map on model pages: a layer × neuron grid coloured by status (VERIFIED, rejected, dead, unexplored), so coverage and where the interpretable neurons sit are visible.
- [ ] Registry dashboard on the home page: experiments and VERIFIED share over time, verdicts per model and per layer, and per-agent acceptance rates.
- [ ] Later: circuit diagrams once attention heads land, and a cross-model graph linking matched neurons for universality.

## Scope expansion

- [ ] Generalise `NeuronIdentifier` to `ComponentIdentifier` (MLP neuron pole, attention head) and add head hooks. See `attention-head-audit`.
- [ ] A transcoder/SAE baseline arm to test the native-neuron premise on the same layers.
- [ ] Cross-model universality links: match verified neurons across sibling models (Qwen2.5-0.5B ↔ 1.5B).

## Datasets

Scan corpora beyond `NeelNanda/pile-10k`. Record the dataset ID and revision on every submission so scans are reproducible, and check each licence before redistributing excerpts in the wiki.

- [ ] `HuggingFaceFW/fineweb-edu` (sample-10BT): clean modern web text, closer to the pretraining mix of recent models than the Pile.
- [ ] `allenai/c4` (en, validation split): a standard web baseline used by much of the SAE literature.
- [ ] `monology/pile-uncopyrighted`: a larger Pile sample when 10k documents are too few to find rare-concept activations.
- [ ] `wikitext` (`wikitext-103-raw-v1`): long, clean articles for ≥ 512-token windows and entity/fact neurons.
- [ ] `roneneldan/TinyStories`: simple English for a low-confound control corpus and small models.
- [ ] Code: `bigcode/the-stack-smol` or `codeparrot/github-code-clean`, for syntax, bracket and indentation neurons.
- [ ] Multilingual: `wikimedia/wikipedia` (non-English configs) or `uonlp/CulturaX`, for language-identity and script neurons.
- [ ] Maths: `open-web-math/open-web-math` and `openai/gsm8k`, for number, operator and LaTeX neurons.
- [ ] Chat/instruction text: `HuggingFaceH4/ultrachat_200k` or `lmsys/lmsys-chat-1m` (gated), formatted with the model's chat template, for instruct models.
- [ ] Labelled concept sets for `sparse-probing-search` and positive holdouts: `conll2003` (NER), `universal_dependencies` (POS/syntax), `papluca/language-identification`, `ag_news` (topic).
- [ ] A mixed default corpus: a fixed, versioned blend of the above, so random-selection statistics aren't biased toward one domain.

## Models

The hooks already resolve Llama/Qwen2/Mistral/Gemma/Phi, GPT-2, GPT-J, GPT-NeoX and OPT layouts (`openinterp/harness/hooks.py`). Each new model needs a smoke test, a note on its MLP activation (gated SwiGLU/GeGLU vs plain GELU, which changes what a "negative pole" means) and a check that the final-norm path resolves.

- [ ] Qwen2.5 family: 1.5B, 3B and the `-Instruct` variants, for base ↔ instruct and size-scaling comparisons with the default model.
- [ ] Qwen3 (0.6B, 1.7B, 4B): the successor architecture; check `q_norm`/`k_norm` don't break hook resolution.
- [ ] Llama 3.2 (1B, 3B) and Llama 3.1 8B (gated licence).
- [ ] Gemma 2 (2B, 9B) and Gemma 3 (1B, 4B): GeGLU MLPs, and Gemma Scope / Gemma Scope 2 SAEs and transcoders for the baseline arm.
- [ ] Pythia suite (70M–1.4B, with training checkpoints and the `-deduped` variants): the standard for tracking a neuron across training.
- [ ] GPT-2 small/medium: the best-documented model in the circuits literature; good for the positive-control task.
- [ ] SmolLM2 (135M, 360M, 1.7B): fully open data, cheap enough for CI smoke tests.
- [ ] OLMo 2 (1B, 7B): open data and checkpoints, so activating examples can be traced to training documents.
- [ ] Phi-3.5-mini / Phi-4-mini: a fused `gate_up_proj`, so check the `mlp.down_proj` input still exposes one value per neuron.
- [ ] Mistral 7B v0.3: a common 7B baseline (needs GPU or quantised loading).
- [ ] Stanford `stanford-crfm` GPT-2 seeds (alias/battlestar/…): same architecture, different seeds, for a universality baseline.
- [ ] Mixture-of-experts (Qwen1.5-MoE-A2.7B, OLMoE-1B-7B): needs a decision on how to identify an expert neuron (`expert` index in the identifier) and how to handle routing in patching. Defer until `ComponentIdentifier` lands.

## Housekeeping

- [ ] Fix the scheduled deploy (`.github/workflows/deploy.yml`): the hourly run doesn't seem to deploy. First suspect: the job-level `if: vars.R2_BUCKET != ''` is evaluated before the `production` environment is applied, so if `R2_BUCKET` is an environment variable rather than a repository variable the job is silently skipped, not failed. Check the Actions run history for skipped runs, and confirm the cron is firing at all (schedules only run from the default branch and are disabled after 60 days without activity).
- [ ] Replace the synthetic `examples/audits/` fixtures with real records once some are VERIFIED.
- [ ] Soften the README's "remarkably clean" premise until Goodfire's technical companion paper is published.
- [ ] Mark the CI jobs "Lint and tests", "Experiment records" and "Reproduce records" as required in the GitHub branch protection rules for `main`.
- [ ] Add the R2 secrets and variables to GitHub (see README, "Hourly deploy"), then run the deploy workflow once by hand.
- [ ] GitHub disables scheduled workflows after 60 days without repository activity; keep an eye on the hourly deploy.

## Later stage

Research directions to pick up once the registry covers several models well.

- [ ] Investigate using the Platonic Representation Hypothesis (PRH) once multiple models are well understood.
- [ ] Investigate predicting an unknown neuron's interpretability based on previous audits.
- [ ] Investigate simplifying neural pathways.
- [ ] Investigate inserting memories or behaviours.
- [ ] Investigate removing memories or behaviours.
- [ ] Investigate complete explainability of each token.
- [ ] Investigate model concept merging or diffing.
