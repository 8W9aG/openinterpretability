# Open Interpretability

`openinterp` is a decentralized, agent-audited mechanistic interpretability registry and testing protocol for open-weight foundation models (Llama, Qwen, Mistral, …).

Rather than cataloging the latents of learned SAE dictionaries, this project audits **native model weights directly**: each entry in the registry is one *pole* (positive or negative half) of one MLP neuron. It builds on recent findings that native representations are remarkably clean when evaluated properly (e.g. Goodfire / Cammarata, 2026; Bereska & Gavves, 2024).

## Protocol

Every claim is empirical and falsifiable:

1. **Propose.** A submitting agent scans a corpus for the neuron pole's top-activating snippets. It then writes a hypothesis, says whether the pole is *retrospective* (reacts to tokens it has already read) or *lookahead* (prepares an upcoming token), and supplies minimal-pair counterfactuals with predicted activation deltas.
2. **Review.** A reviewer agent, which must run on a different model from the proposer, writes 10–20 adversarial probes: mostly hard negatives (near-misses the hypothesis says should *not* fire), plus some positive holdouts. Every probe and counterfactual is then run through the real model. The verdict comes from those measurements and not from the reviewer's opinion, so anyone with the weights can reproduce it:
   - balanced accuracy on hard negatives vs. positive holdouts, each gated separately
   - counterfactual Δ reproduction within a tolerance relative to the prediction
   - lookahead consistency (the claimed mechanism compared with KL(clean ‖ pole-ablated) on next-token distributions)
   - causal sufficiency (clean → corrupted patching) and necessity (corrupted → clean resample ablation), with downstream MLP outputs frozen to mitigate self-repair (the hydra effect); recorded but not yet gated
3. **Publish.** `VERIFIED` records are compiled into a fully static site plus a Parquet registry. The output is hosted on Cloudflare R2 (zero egress) and searched client-side with Pagefind and DuckDB-Wasm.

The full experimental protocol (roles, lifecycle, decision rule and reproducibility requirements) is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Quickstart

```bash
make install        # uv sync
make test           # pytest (CPU only, no downloads)
make sample-run     # build dist/ from synthetic example records
make serve          # http://localhost:8000 (DuckDB-Wasm and Pagefind need HTTP)
```

Audit a real neuron. This needs an LLM API key for whichever [LiteLLM](https://docs.litellm.ai/) provider you choose:

```bash
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY, etc.
openinterp audit  --model Qwen/Qwen2.5-0.5B --layer 12 --neuron 3071 --pole positive \
                  --corpus my_corpus.txt --agent-model anthropic/claude-opus-5-5
openinterp verify --audit-file submissions/Qwen--Qwen2.5-0.5B/L12/N3071-positive.json \
                  --agent-model openai/gpt-5
openinterp build  --audits-dir audits --output-dir dist
openinterp deploy --dist-dir dist --bucket my-r2-bucket [--delete] [--dry-run]
```

The default agent model is `$OPENINTERP_AGENT_MODEL`, falling back to `anthropic/claude-opus-5-5`. `review` is an alias for `verify`.

## Experiment wiki

Every completed experiment gets a Markdown page in [`wiki/`](wiki/README.md), whatever its verdict. `openinterp verify` and `openinterp reevaluate` write it automatically, and `openinterp wiki` backfills pages for existing records. Each page records the hypothesis, top examples, gate results, probes, counterfactuals and next steps based on the gates that failed. A page is written once and never regenerated, so its Discussion section is safe to edit. Each neuron also gets a history page, and `wiki/README.md` indexes every experiment.

## Pull requests

[CI](.github/workflows/ci.yml) runs two required jobs on every PR:
- **Lint and tests:** ruff, a 500-line limit on source and documentation files (tests, configuration and generated data are exempt), and pytest, with CPU-only torch (`scripts/ci-install.sh`).
- **Experiment records:** `openinterp check`. Every audit record in `audits/` must validate, have a verdict, be the only record for its neuron and have a wiki page for that verdict. Every wiki experiment page must have an audit record. The wiki indexes must be current, and existing experiment pages may only change in their Discussion section.

Run `make check` locally before pushing. The [PR template](.github/pull_request_template.md) has a checklist for posting experiments.

## Claude Code skills

`.claude/skills/` holds research skills that Claude Code loads automatically in this repo. Each one condenses a set of papers into a procedure that uses the `openinterp` harness:

| Skill | Use it to | Based on |
| --- | --- | --- |
| `choose-next-experiment` | Pick the next experiment with `openinterp next`: randomised, explore vs exploit, herd-resistant | ε-greedy and Boltzmann sampling over follow-ups |
| `native-neuron-audit` | Avoid misreading a neuron (context, data, poles, SwiGLU regimes) | Goodfire, *Already Clean* (2026) |
| `vocab-projection-lookahead` | Decide retrospective vs lookahead | Geva et al. 2021/2022; logit lens; tuned lens; Voita et al. 2023 |
| `neuron-family-triage` | Pre-classify neurons from weight statistics | Gurnee et al. 2024; Stolfo et al. 2024; Voita et al. 2023 |
| `explanation-simulation-scoring` | Score explanations by predicted activations | Bills et al. 2023; Huang et al. 2023 |
| `activation-patching-practice` | Choose causal metrics and corruptions; handle self-repair | Meng et al. 2022; Zhang & Nanda 2023; Heimersheim & Nanda 2024; McGrath et al. 2023; Rushing & Nanda 2024; AtP* |
| `neuron-steering-tests` | Test a hypothesis by clamping the neuron and generating | Turner et al. 2023; Goodfire 2026 |
| `sparse-probing-search` | Find neurons for a labelled concept | Gurnee et al. 2023; Bolukbasi et al. 2021 |
| `interpretability-illusions-review` | Write probes as the reviewer; spot illusions | Bolukbasi et al. 2021; Huang et al. 2023; Makelov et al. 2023; Sun et al. 2024 |
| `attention-head-audit` | Extend the registry to attention heads and transcoder baselines | Olsson et al. 2022; Wang et al. 2022; Conmy et al. 2023; Dunefsky et al. 2024; Paulo et al. 2025 |

## Layout

| Path | Purpose |
| --- | --- |
| `openinterp/core/schemas.py` | Strict Pydantic v2 models: `NeuronIdentifier`, `CounterfactualPair`, `LookaheadProfile`, `CausalMetrics`, `NeuronAuditSubmission`, `VerificationVerdict`, `AuditRecord` |
| `openinterp/harness/hooks.py` | `NeuronHookRunner`: forward hooks on the MLP down-projection input (the neuron basis) for Llama/Qwen/Mistral/GPT-2/NeoX/OPT/Phi; pole isolation; corpus scanning |
| `openinterp/harness/patcher.py` | Denoising/noising activation patching and resample ablation per neuron pole, optionally freezing downstream MLPs |
| `openinterp/harness/attribution.py` | Attribution patching (Δactivation × gradient) for whole layers in one backward pass |
| `openinterp/harness/lookahead.py` | Pole-ablation KL profile and unembedding projection of the neuron's output direction |
| `openinterp/agents/` | `ProposerAgent` and `ReviewerAgent` (LiteLLM + instructor structured outputs); deterministic `evaluate_submission` |
| `openinterp/builder/` | Jinja2 + Tailwind static site, `registry.parquet` export, DuckDB-computed leaderboards |
| `openinterp/sync/r2.py` | MD5/ETag delta sync to R2 with cache headers |

### Definitions

- **Neuron activation:** the input to the MLP's down/output projection. For gated MLPs this is `act(gate(x)) * up(x)`.
- **Pole:** the positive pole is `clamp(a, min=0)` and the negative pole is `clamp(a, max=0)`. Pole *strength* is always reported as a non-negative magnitude.
- **Counterfactual Δ:** `(peak strength on base − peak strength on counterfactual) / max activation`.

## Static site & data

```
dist/
├── index.html                       # stats, per-layer coverage, leaderboard, SQL console
├── neurons/<model>/L<layer>/N<idx>-<pole>.html   # data-pagefind-body
├── data/registry.parquet            # latest, max-age=3600
├── data/snapshots/registry-<sha>.parquet         # content-addressed, immutable
├── data/manifest.json
└── pagefind/                        # written by `python -m pagefind --site dist` (make search-index)
```

Cache policy: content-addressed files (`data/snapshots/*` and Pagefind shards) are served with `max-age=31536000, immutable`. Everything else, including HTML and `registry.parquet`, gets `max-age=3600`. Deploys upload assets before HTML, so a page never references an object that has not arrived yet.

R2 credentials are read from `CF_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.

### Hourly deploy

[deploy.yml](.github/workflows/deploy.yml) rebuilds `dist/` from `main` every hour (and on manual dispatch), runs `openinterp check`, and runs `openinterp deploy --delete`. **An unchanged registry deploys as a no-op**, which takes four measures:

- **Deterministic build.** There are no build timestamps. The footer date and `manifest.json`'s `updated_at` come from the latest verdict, and every ordering is total. Unchanged records rebuild byte-for-byte, which a test enforces.
- **Content comparison.** Only files whose MD5 differs from the remote ETag are uploaded.
- **Derived index.** Pagefind's output is not byte-stable between runs because it orders filters from a hash map. `pagefind/` is therefore only synced when an HTML or data file actually changed.
- **Targeted purges.** When something does change, only the changed short-cache URLs are purged from Cloudflare's cache. Content-addressed files get new names, and `data/snapshots/` is never deleted.

Configure the workflow in the repo's GitHub settings:

| Name | Kind | Purpose |
| --- | --- | --- |
| `CF_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | secrets | R2 S3 API access |
| `R2_BUCKET` | variable | Bucket name |
| `CF_ZONE_ID`, `CF_API_TOKEN` | secrets (optional) | Cache purge for a custom domain; the token needs *Zone → Cache Purge* |
| `OPENINTERP_SITE_URL` | variable (optional) | Public URL of the bucket root, used to build the purge URLs |

## Status

Early scaffold. The records in `examples/audits/` are **synthetic fixtures** for exercising the builder. They are not real audit results.
