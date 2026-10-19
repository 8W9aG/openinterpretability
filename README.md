# Open Interpretability

`openinterp` is a decentralized, agent-audited mechanistic interpretability registry and testing protocol for open-weight foundation models (Llama, Qwen, Mistral, …).

Rather than cataloging the latents of learned SAE dictionaries, this project audits **native model weights directly**: each entry in the registry is one *pole* (positive or negative half) of one MLP neuron. It builds on recent findings that native representations are remarkably clean when evaluated properly (e.g. Goodfire / Cammarata, 2026; Bereska & Gavves, 2024).

## Get started: run an experiment and contribute it

An experiment takes one neuron, proposes what it does, has a *different* model try to break that claim, and posts the result, whatever it is. A hypothesis that fails is a result too. You can let an agent drive the whole loop in Claude Code (path A) or run each step yourself (path B).

### 1. Set up (once)

You need Git, a GitHub account, Python 3.11+ and [uv](https://docs.astral.sh/uv/). The [GitHub CLI](https://cli.github.com/) (`gh`) is optional but recommended: it lets the planner see which neurons other contributors have already claimed. A laptop CPU is enough for the default model, Qwen2.5-0.5B (about 1 GB, downloaded on first use).

```bash
gh repo fork 8W9aG/openinterpretability --clone   # or fork on GitHub and git clone your fork
cd openinterpretability
make install                                      # creates .venv with everything
```

**An LLM.** Any LLM can write the proposals and reviews. `--agent-model` takes either:

- a [LiteLLM](https://docs.litellm.ai/) model string, with that provider's API key set, e.g. `export ANTHROPIC_API_KEY=...` for `anthropic/claude-opus-5-5` (the default);
- `external/<name>` (e.g. `external/claude-opus-5-5`), which needs no keys. The command writes `exchange/<Agent>.prompt.md`, including the JSON schema of the answer, and waits for `exchange/<Agent>.response.json`. A coding agent, a chat window or a person can write the answer. Name the model that actually answered, because that name is recorded.
- answer files, which need no keys and no waiting process. `audit --evidence-out` and `review --emit-prompt` write the prompt and exit; `audit --evidence … --hypothesis …` and `review --probes …` take the answer as JSON. In Claude Code, the `neuron-proposer` and `neuron-reviewer` subagents (`.claude/agents/`) write those answers:

  ```bash
  uv run openinterp audit --model Qwen/Qwen2.5-0.5B --random --corpus corpus.jsonl --evidence-out exchange/ev.json
  # neuron-proposer answers exchange/ev.prompt.md -> exchange/ev.hypothesis.json
  uv run openinterp audit --evidence exchange/ev.json --hypothesis exchange/ev.hypothesis.json \
    --agent-model external/claude-opus-5-5 --out submissions/ev.json
  uv run openinterp review --audit-file submissions/ev.json --emit-prompt exchange/review.prompt.md
  # neuron-reviewer answers exchange/review.prompt.md -> exchange/review.probes.json
  uv run openinterp review --audit-file submissions/ev.json --probes exchange/review.probes.json \
    --agent-model external/claude-sonnet-5-5
  ```

  The reviewer subagent has no tool that runs the model, so it can't see probe activations before its probe set is final. The two subagents run on different models (Opus and Sonnet), so the review counts as independent. Pass each one's reported model ID to `--agent-model`.

The record notes whether the reviewer was a different model from the proposer (`independent_review`). A different model is better, but it isn't required.

**A scan corpus.** The built-in corpus is only a smoke test. This builds a better one from 500 Pile documents (one-off download, git-ignored):

```bash
uv run python - <<'PY'
import json, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
path = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet", repo_type="dataset")
with open("corpus.jsonl", "w") as f:
    for text in pq.read_table(path, columns=["text"]).column("text").to_pylist()[:500]:
        f.write(json.dumps({"text": text[:3000]}) + "\n")
PY
```

### 2A. Let an agent run it (Claude Code)

Start [Claude Code](https://claude.com/claude-code) in the repo with the keys above set in the same shell, and ask:

```text
Run the next experiment and open a PR with the result.
Use corpus.jsonl for the scan, and openai/gpt-5 as the reviewer model.
```

The repo's skills load automatically, and `choose-next-experiment` defines the workflow. The agent:
1. picks an experiment with `openinterp next`, which is randomised so contributors don't collide;
2. claims it with a draft PR;
3. reads the neuron's history in `wiki/`;
4. runs the audit and the independent review;
5. posts the result.

The other skills (`native-neuron-audit`, `vocab-projection-lookahead`, `interpretability-illusions-review`, …) supply the methodology. Review the PR before marking it ready.

**In a loop.** `scripts/experiment-loop.sh` runs experiments back to back, each in a fresh headless Claude Code session (`claude -p`), until it reaches `--max` or `claude` exits with an error (for example at the usage limit). Claude drives the workflow; you choose the LLMs that propose and review:

```bash
scripts/experiment-loop.sh --proposer anthropic/claude-opus-5-5 --reviewer openai/gpt-5 --max 10
make loop LOOP_ARGS="--reviewer openai/gpt-5"   # the same, via make
```

It works in a separate worktree (`../<repo>-loop`) reset to `origin/main` before each experiment, so your checkout is untouched, and writes one log per experiment to `logs/loop/`. Each experiment opens its own PR, so review them as they arrive. `--driver` picks the Claude model that drives the session, `--epsilon` how often `openinterp next` explores new neurons, and a proposer or reviewer of `external/<name>` makes the driving session answer that role itself. The session can edit files and run `uv run`, `make`, `git` and `gh pr` without asking; see `ALLOWED_TOOLS` in the script. Run `scripts/experiment-loop.sh --help` for every option.

### 2B. Run it yourself

```bash
uv run openinterp next                    # what to do next, e.g. "explore" a random neuron
uv run openinterp audit --model Qwen/Qwen2.5-0.5B --random --corpus corpus.jsonl
#   → scans the neuron (~5–10 min on CPU), asks the proposer for a hypothesis,
#     writes submissions/<model>/L<layer>/N<idx>-<pole>.json
uv run openinterp verify --audit-file submissions/<…>.json --agent-model openai/gpt-5
#   → probes, measured on the model; writes a new experiment (record + page) under wiki/experiments/
```

For follow-ups that `next` suggests (revise, re-review, opposite pole), run the commands it prints. Add `--corpus corpus.jsonl` to any `audit`.

### 3. Contribute it

The planner reads open PRs to avoid duplicate work, so claim the neuron early. The branch and PR title use the neuron's slug with `/` replaced by `-`, e.g. `Qwen--Qwen2.5-0.5B-L11-N4255-positive`:

```bash
git switch -c experiment/Qwen--Qwen2.5-0.5B-L11-N4255-positive
gh pr create --draft --title "experiment: Qwen--Qwen2.5-0.5B-L11-N4255-positive (explore)"
# …run the experiment, then:
make check lint test                      # the same checks CI runs on your PR
git add wiki && git commit -m "Experiment: L11/N4255+ (NEEDS_REVISION)" && git push
gh pr ready
```

Commit `wiki/`. Each experiment adds its own record (`<id>.json`) and page (`<id>.md`) and never edits another's, so any number of experiments on one neuron can be in flight at once. Never edit generated pages except their **Discussion** section. CI rejects a PR whose record and page don't match, and once merged the hourly deploy publishes the result. Read [docs/PROTOCOL.md](docs/PROTOCOL.md) for how verdicts are decided.

## Protocol

Every claim is empirical and falsifiable:

1. **Propose.** A submitting agent scans a corpus for the neuron pole's top-activating snippets. It then writes a hypothesis, says whether the pole is *retrospective* (reacts to tokens it has already read) or *lookahead* (prepares an upcoming token), and supplies minimal-pair counterfactuals with predicted activation deltas.
2. **Review.** A reviewer agent (ideally, but not necessarily, a different model from the proposer) writes 10–20 adversarial probes: mostly hard negatives (near-misses the hypothesis says should *not* fire), plus some positive holdouts. Every probe and counterfactual is then run through the real model. The verdict comes from those measurements and not from the reviewer's opinion, so anyone with the weights can reproduce it:
   - balanced accuracy on hard negatives vs. positive holdouts, each gated separately, with near-duplicate probes counted once
   - confound control: the hypothesis must match or beat token-identity, position and never-active baselines
   - counterfactual Δ reproduction within a tolerance relative to the prediction
   - lookahead consistency (the claimed mechanism compared with KL(clean ‖ pole-ablated) on next-token distributions)
   - covariance: the most correlated same-layer neurons, and joint ablation with them to expose redundancy (recorded, not gated)
   - causal sufficiency (clean → corrupted patching) and necessity (corrupted → clean resample ablation), with downstream MLP outputs frozen to mitigate self-repair (the hydra effect); recorded but not yet gated
3. **Publish.** Every experiment is compiled into a fully static site plus a Parquet registry, whatever its verdict. Null results are data too. Within each neuron, experiments are ordered by how *conclusive* their evidence is, in either direction (see `openinterp/builder/significance.py`). The output is hosted on Cloudflare R2 (zero egress) and searched client-side with Pagefind and DuckDB-Wasm.

The full experimental protocol (roles, lifecycle, decision rule and reproducibility requirements) is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Development

```bash
make install        # uv sync
make test           # pytest (CPU only, no downloads)
make lint           # ruff + the 500-line limit
make check          # registry consistency (what CI's "Experiment records" job runs)
make sample-run     # build dist/ from synthetic example records
make build          # build dist/ from the wiki archive, plus the Pagefind index
openinterp deploy --dist-dir dist --bucket my-r2-bucket [--delete] [--dry-run]
```

`dist/index.html` opens directly from disk; search and the SQL console need HTTP (`make serve`). The default agent model is `$OPENINTERP_AGENT_MODEL`, falling back to `anthropic/claude-opus-5-5`, and `review` is an alias for `verify`.

## Experiment wiki

Every completed experiment gets a Markdown page and an archived copy of its full record (`<id>.json`) in [`wiki/`](wiki/README.md), whatever its verdict. `openinterp verify` and `openinterp reevaluate` write it automatically, and `openinterp wiki` backfills pages for existing records. Each page records the hypothesis, top examples, gate results, probes, counterfactuals and next steps based on the gates that failed. A page is written once and never regenerated, so its Discussion section is safe to edit. Each neuron also gets a history page, and `wiki/README.md` indexes every experiment.

## Pull requests

[CI](.github/workflows/ci.yml) runs two required jobs on every PR:
- **Lint and tests:** ruff, a 500-line limit on source and documentation files (tests, configuration and generated data are exempt), and pytest, with CPU-only torch (`scripts/ci-install.sh`).
- **Experiment records:** `openinterp check`. Every experiment record (`wiki/experiments/<slug>/<id>.json`) must validate, have a verdict, sit at the path its neuron and id dictate, and have a page whose front matter matches it; every page must have a record. The wiki indexes must be current, existing records are write-once, and existing pages may only change in their Discussion section. The retired `audits/` directory must be empty.

Run `make check` locally before pushing. The [PR template](.github/pull_request_template.md) has a checklist for posting experiments.

## Claude Code skills

`.claude/skills/` holds research skills that Claude Code loads automatically in this repo. Each one condenses a set of papers into a procedure that uses the `openinterp` harness:

| Skill | Use it to | Based on |
| --- | --- | --- |
| `choose-next-experiment` | Pick the next experiment with `openinterp next`: randomised, explore vs exploit, herd-resistant | ε-greedy and Boltzmann sampling over follow-ups |
| `native-neuron-audit` | Avoid misreading a neuron (context, data, poles, SwiGLU regimes) | Goodfire, *Already Clean* (2026) |
| `vocab-projection-lookahead` | Decide retrospective vs lookahead | Gurnee et al. 2026 (Jacobian lens); Geva et al. 2021/2022; logit lens; Voita et al. 2023 |
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
| `openinterp/agents/` | `ProposerAgent` and `ReviewerAgent` (structured outputs from LiteLLM or any external LLM via prompt files); deterministic `evaluate_submission` |
| `openinterp/builder/` | Jinja2 + Tailwind static site, `registry.parquet` export, DuckDB-computed leaderboards |
| `openinterp/sync/r2.py` | MD5/ETag delta sync to R2 with cache headers |

### Definitions

- **Neuron activation:** the input to the MLP's down/output projection. For gated MLPs this is `act(gate(x)) * up(x)`.
- **Pole:** the positive pole is `clamp(a, min=0)` and the negative pole is `clamp(a, max=0)`. Pole *strength* is always reported as a non-negative magnitude.
- **Counterfactual Δ:** `(peak strength on base − peak strength on counterfactual) / max activation`.

## Static site & data

The site is built from the wiki's experiment archive (`wiki/experiments/**/<id>.json`). That archive holds every experiment, including null and superseded results.

```
dist/
├── index.html                                   # totals, models, most conclusive results, verified, SQL console
├── models/<model>/index.html                    # layer coverage
├── models/<model>/L<layer>/index.html           # neurons studied in the layer
├── models/<model>/L<layer>/N<idx>.html          # every experiment (both poles), most conclusive first
├── data/registry.parquet            # one row per experiment; latest, max-age=3600
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

Early scaffold. The records in `examples/audits/` (imported with `openinterp wiki --import-dir`) are **synthetic fixtures** for exercising the builder. They are not real audit results.
