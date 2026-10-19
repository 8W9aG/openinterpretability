# TODO

Open work, roughly in priority order. Move items to `CHANGELOG.md` under `[Unreleased]` when they land. Items marked **(protocol)** change the decision rule and must bump `PROTOCOL_VERSION` (`docs/PROTOCOL.md` §6).

## Protocol and science

- [ ] **(protocol)** Replace the top-1-logit causal metric with one that can detect single-neuron effects: logit difference on the claimed tokens, or the projection onto the neuron's output direction at a downstream reader. See `activation-patching-practice`.
- [ ] Rescan L11/N4255+ so it gets a confound profile and correlated neurons (protocol v0.3); its current submission predates them and fails `beats_confound_baseline`.
- [ ] Re-audit Qwen2.5-0.5B L11/N4255+ with ≥ 512-token windows, a narrowed hypothesis ("*including* / *e.g.* after a category noun") and an independent reviewer model.
- [ ] Scan both poles by default and show them together on neuron pages.
- [ ] Positive control: audit known induction heads or other well-documented units and confirm the protocol VERIFIES them.
- [ ] **(protocol)** Decide whether steering predictions and simulation scores should become gates. Both need a pre-registered prediction format first.
- [ ] Report registry-wide statistics (share VERIFIED, share dead) over `selection = "random"` records only.
- [ ] Record dead neurons found by random selection instead of discarding them.

## Pipeline

- [ ] Claude Code agent flow ("option 2"): `.claude/agents/neuron-proposer.md` and `neuron-reviewer.md`, plus CLI commands that emit scan evidence and accept hypotheses and probe sets as JSON, so agents need no API key.
- [ ] `openinterp audit`: configurable window length and stride, and Hugging Face dataset input (`--dataset NeelNanda/pile-10k`).
- [ ] Faster scans: MPS/CUDA batching and caching activations to disk for re-use across neurons in a layer.
- [ ] A `supersedes` field on submissions to link revisions to the record they replace.
- [ ] A scheduled hourly build and deploy (CI cron), as described in the vision.

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

- [ ] Replace the synthetic `examples/audits/` fixtures with real records once some are VERIFIED.
- [ ] Soften the README's "remarkably clean" premise until Goodfire's technical companion paper is published.
- [ ] Mark the CI jobs "Lint and tests" and "Experiment records" as required in the GitHub branch protection rules for `main`.
- [ ] Optional CI job: re-run `openinterp reevaluate` on changed records to confirm their verdicts reproduce (needs model downloads and caching).
- [ ] Add the R2 secrets and variables to GitHub (see README, "Hourly deploy"), then run the deploy workflow once by hand.
- [ ] GitHub disables scheduled workflows after 60 days without repository activity; keep an eye on the hourly deploy.
