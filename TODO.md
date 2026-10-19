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

## Housekeeping

- [ ] Replace the synthetic `examples/audits/` fixtures with real records once some are VERIFIED.
- [ ] Soften the README's "remarkably clean" premise until Goodfire's technical companion paper is published.
- [ ] Mark the CI jobs "Lint and tests" and "Experiment records" as required in the GitHub branch protection rules for `main`.
- [ ] Optional CI job: re-run `openinterp reevaluate` on changed records to confirm their verdicts reproduce (needs model downloads and caching).
- [ ] Add the R2 secrets and variables to GitHub (see README, "Hourly deploy"), then run the deploy workflow once by hand.
- [ ] GitHub disables scheduled workflows after 60 days without repository activity; keep an eye on the hourly deploy.
