# Open Interpretability

`openinterp` is a decentralized, agent-audited mechanistic interpretability registry and testing protocol for open-weight foundation models (Llama, Qwen, Mistral, …).

Rather than cataloging the latents of learned SAE dictionaries, this project audits **native model weights directly**: each entry in the registry is one *pole* (positive or negative half) of one MLP neuron. It starts from a working hypothesis, not an established finding: Goodfire's essay [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (Cammarata, 2026) argues that most apparent polysemanticity in native neurons comes from fixable tooling problems rather than superposition. Its technical companion paper has not been published, so the registry tests that claim one pole at a time rather than assuming it, and a pole that resists a clean explanation is a result too.

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
  uv run openinterp score --audit-file submissions/ev.json --corpus corpus.jsonl --emit-task exchange/score.task.json
  # one explanation-judge answers exchange/score.detection.prompt.md, another exchange/score.fuzzing.prompt.md
  uv run openinterp score --task exchange/score.task.json --judge-model external/claude-haiku-5-5
  uv run openinterp review --audit-file submissions/ev.json --probes exchange/review.probes.json \
    --agent-model external/claude-sonnet-5-5 --explanation-scores exchange/score.scores.json
  ```

  Every review also records explanation scores. `openinterp score` scores the explanation on held-out text the proposer never saw: Delphi's detection, fuzzing and embedding scores, with the `explanation-judge` subagent answering its prompts. `review --explanation-scores` records the result, and `review` refuses `--probes` without it. A one-step review (no `--probes`) scores the explanation itself on `--corpus`. The scores are reported, not gated (docs/PROTOCOL.md §4).

  To record what each subagent cost, run `uv run openinterp usage <agent ID> --answer <its answer file>` after it finishes, before the command that reads the answer. It reads the subagent's Claude Code transcript and writes `<answer>.usage.json` next to the answer, and `audit`, `review` and `score` then record its tokens and list-price cost (`openinterp/core/pricing.py`) instead of only the prompt's length. `openinterp costs --wiki-dir wiki` reports cost per experiment by role and model.

  The reviewer subagent has no tool that runs the model, so it can't see probe activations before its probe set is final. The two subagents run on different models (Opus and Sonnet), so the review counts as independent. Pass each one's reported model ID to `--agent-model`.

The record notes whether the reviewer was a different model from the proposer (`independent_review`). A different model is better, but it isn't required.

**A scan corpus.** `audit` scans `./corpus.jsonl` unless given `--corpus`, and stops if it isn't there. Almost every record in the registry scanned `corpus.jsonl`, the first 500 documents of [pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k), each cut to 3,000 characters (SHA-256 `ded48cfc…`, git-ignored). Rebuild it byte for byte, so your scans are comparable with the registry's:

```bash
uv run openinterp corpus --dataset pile-10k --first --ascii --revision 127bfedcd5047750df5ccf3a12979a47bfa0bafa --out corpus.jsonl
```

Its sidecar, `corpus.source.json`, is committed, and `openinterp build` uses it to show the source of records that scanned this corpus before the sidecar existed, marked as inferred. The dataset card gives the licence as `bigscience-bloom-rail-1.0`, a model licence, and each Pile component keeps its own terms, so the corpus itself isn't published. For another corpus, `openinterp corpus` samples documents from a Hugging Face dataset, reading only the Parquet row groups it needs. Without `--dataset` it samples 500 documents from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)'s 10B-token sample in about a minute:

```bash
uv run openinterp corpus --out fineweb.jsonl           # --dataset pile-10k, pile-uncopyrighted, c4, wikitext, tinystories, github-code or wikipedia-multilingual for another corpus, --seed N for another sample, --detect-lang to label each document's language
```

It also writes `<name>.source.json` (here `fineweb.source.json`), which pins the dataset commit, files, seed and licence. `openinterp audit --corpus fineweb.jsonl` records it in the submission, and warns about a corpus without one, and experiment pages show the licence next to the excerpts. `audit` scans a seeded sample of up to 200,000 tokens of the corpus, the same documents every time (`--max-tokens`, `--sample-seed`; `--max-tokens 0` scans all of it), and records the sample in provenance. Any Hugging Face Parquet dataset works: `--dataset <org/name> --files '<glob>'`. For long documents, raise `--max-chars` and scan them as overlapping windows with `openinterp audit --window-tokens 512`, which are recorded in provenance.

For labelled text rather than a scan corpus, `openinterp concepts` writes human annotations (CoNLL-2003 entities, Universal Dependencies parts of speech and relations, 20-language identification, AG News topics) as the character spans `openinterp probe-compare` reads; run it without `--label` to list a dataset's labels.

A scan's forward pass captures the whole layer, so `audit` caches every neuron's activations in that layer (in `~/.cache/openinterp/scans`; `$OPENINTERP_SCAN_CACHE` moves it, `off` disables it). A later audit of any neuron in the layer, with the same model, corpus options and dtype, replays the cache instead of running the model and gets the same numbers. The cache is capped at 40 GB (`$OPENINTERP_SCAN_CACHE_GB`); a 200,000-token Qwen2.5-0.5B layer in float32 takes about 4 GB, and the least recently used layers go first. `openinterp scan-cache` lists or clears it. For a neuron's per-token activations over the scan, e.g. every occurrence of one token, use `openinterp activations --model … --layer … --neuron … --corpus corpus.jsonl` (`--token`, `--min-strength`, `--top 0` for all, `--format jsonl`) rather than writing a scan loop.

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

It works in a separate worktree (`../<repo>-loop`) reset to `origin/main` before each experiment, so your checkout is untouched, and writes one log per experiment to `logs/loop/`. Each session follows `land-experiment`: it posts the PR, waits for CI to go green and merges it if the PR only touches `wiki/` and your `gh` account has write access. Otherwise it leaves the PR open for review and moves on. `--driver` picks the Claude model that drives the session (default Haiku, which landed every experiment in two pilots at about a tenth of Sonnet's cost; `docs/analyses/driver-model-pilot.md`), `--epsilon` how often `openinterp next` explores new neurons, and a proposer or reviewer of `external/<name>` makes the driving session answer that role itself. The session can edit files and run `uv run`, `make`, `git`, `gh pr`, `gh run` and `gh repo view` without asking; see `ALLOWED_TOOLS` in the script. After each experiment it prints what that experiment cost and a running total: wall time per `openinterp` command and stage, LLM tokens and cost, and the driving session's turns, tokens and cost. Every `openinterp` command appends its measurements to the ledger in `$OPENINTERP_COSTS_LOG` when that is set, and `openinterp costs <ledger> [--session <claude json>]` summarises one. Records store their own experiment's costs in `provenance.costs`. Run `scripts/experiment-loop.sh --help` for every option.

**Several sessions at once.** Give every session that runs experiments its own working directory, whether it's a loop or an interactive one. Git keeps one current branch per directory, so sessions that share a checkout switch branches under each other, and one session's commits end up on another's claim branch and PR. A `git worktree` per session is enough. Worktrees share the git objects and the Hugging Face model cache, but `corpus.jsonl` is untracked, so copy it in (its sidecar is tracked):

```bash
git worktree add --detach ../openinterpretability-2 origin/main
cp corpus.jsonl ../openinterpretability-2/
cd ../openinterpretability-2 && uv sync && claude
```

Two `experiment-loop.sh` runs need different `--worktree` directories, since both default to `../<repo>-loop`. A branch can be checked out in only one worktree, which is why the workflow starts each experiment from a detached `origin/main`. Each scan holds about 10 GB of RAM on CPU, so memory, not checkouts, limits how many sessions run in parallel.

### 2B. Run it yourself

```bash
uv run openinterp next                    # what to do next, e.g. "explore" a random neuron
                                          #   (--steps prints every step as exact commands)
uv run openinterp audit --model Qwen/Qwen2.5-0.5B --random --corpus corpus.jsonl
#   → scans the neuron (~5–10 min on CPU), asks the proposer for a hypothesis,
#     writes submissions/<model>/L<layer>/N<idx>-<pole>.json
uv run openinterp verify --audit-file submissions/<…>.json --agent-model openai/gpt-5
#   → probes, measured on the model; writes a new experiment (record + page) under wiki/experiments/
uv run openinterp show wiki/experiments/<model>/L<layer>/N<idx>-<pole>   # compact summary
```

Attention heads go through the same commands: `--layer 5 --head 1` audits one head (`--unit head --random` samples one), and every later step reads the head from the submission. A head's strength at a token is the distance between its write to the residual stream and its mean write over the scan, `|W_O (z − z̄)|` ([docs/PROTOCOL.md](docs/PROTOCOL.md) §1).

Latents of published sparse dictionaries are the learned-basis baseline: `openinterp audit --dictionary gpt2-small-transcoders --random --window-tokens 128` audits a random latent of Dunefsky et al.'s GPT-2 small transcoders under the same protocol. The dictionary names its model, and its weights are pinned in the submission. These transcoders fit only the first 128 positions of a context, so their scans, and the GPT-2 neuron scans they are compared with, use 128-token windows. Feature verdicts are reported apart from native units, so the two shares VERIFIED can be compared. The home page compares them per dictionary, over randomly selected units only (`openinterp/builder/learned_baseline.py`). Presets are in `openinterp/harness/dictionaries.py`: besides the GPT-2 transcoders, Gemma Scope's transcoders and MLP-output SAEs for Gemma 2 2B (`gemma-scope-2b-transcoders`, `gemma-scope-2b-mlp-saes`), circuit-tracer's transcoders for Llama 3.2 1B and Qwen3 0.6B (`llama-3.2-1b-transcoders`, `qwen3-0.6b-transcoders`), Gemma Scope's residual-stream SAEs (`gemma-scope-2b-res-saes`), and Llama Scope's residual-stream SAEs, MLP-output SAEs and transcoders for Llama-3.1-8B (`llama-scope-8b-res-saes`, `llama-scope-8b-mlp-saes`, `llama-scope-8b-transcoders`; scan them with `--dtype bfloat16`, since 8B weights in float32 take 32 GB). No public dictionary covers Qwen2.5-0.5B, so `qwen2.5-0.5b-transcoders` was trained here with `scripts/train_transcoders.py`; its card is in `hf_models/`. `docs/dictionaries.md` explains what each preset adds. A residual-stream latent reads and writes the residual stream after its layer, so comparing it with that layer's MLP neurons changes the site as well as the basis. They fit through position 1024, so their scans need `--window-tokens 1024` or less. Each preset's input site was checked by how much of the MLP output it reconstructs; the comment above each preset gives the numbers.

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

To be credited, set `OPENINTERP_CONTRIBUTOR` to a public handle such as your GitHub username (never an email), or pass `--contributor` to `audit` and `verify`. It is recorded beside the agent model, as the proposer's contributor on the submission and the reviewer's on the verdict, and the site's Agents page lists every contributor's experiments, VERIFIED results, well-probed rejections and reviews. Nothing is recorded without it: git config is never read. The loop script's sessions inherit the variable.

Commit `wiki/`. Each experiment adds its own record (`<id>.json`) and page (`<id>.md`) and never edits another's, so any number of experiments on one neuron can be in flight at once. Never edit generated pages except their **Discussion** section. The indexes (`wiki/README.md` and each neuron's `README.md`) are rebuilt locally by `openinterp wiki` and are gitignored, so experiment PRs never conflict over them. CI rejects a PR whose record and page don't match, and once merged the deploy publishes the result within a minute or so. Read [docs/PROTOCOL.md](docs/PROTOCOL.md) for how verdicts are decided.

## Protocol

Every claim is empirical and falsifiable:

1. **Propose.** A submitting agent scans a corpus for the neuron pole's top-activating snippets. It then writes a hypothesis, says whether the pole is *retrospective* (reacts to tokens it has already read) or *lookahead* (prepares an upcoming token), and supplies minimal-pair counterfactuals with predicted activation deltas.
2. **Review.** A reviewer agent (ideally, but not necessarily, a different model from the proposer) writes 10–20 adversarial probes: mostly hard negatives (near-misses the hypothesis says should *not* fire), plus some positive holdouts. Every probe and counterfactual is then run through the real model. The verdict comes from those measurements and not from the reviewer's opinion, so anyone with the weights can reproduce it:
   - balanced accuracy on hard negatives vs. positive holdouts, each gated separately, with near-duplicate probes counted once
   - confound control: the hypothesis must match or beat token-identity, position and never-active baselines
   - counterfactual Δ reproduction within a tolerance relative to the prediction
   - lookahead consistency (the claimed mechanism compared with KL(clean ‖ pole-ablated) on next-token distributions)
   - covariance: the most correlated same-layer neurons, and joint ablation with them to expose redundancy (recorded, not gated)
   - causal sufficiency (clean → corrupted patching) and necessity (corrupted → clean resample ablation), with downstream MLP outputs frozen to mitigate self-repair (the hydra effect), each ranked against same-layer neurons that change as much between the pair; recorded but not yet gated
3. **Publish.** Every experiment is compiled into a fully static site plus a Parquet registry, whatever its verdict. Null results are data too. Within each neuron, experiments are ordered by how *conclusive* their evidence is, in either direction (see `openinterp/builder/significance.py`). The output is hosted on Cloudflare R2 (zero egress) and searched client-side with Pagefind and DuckDB-Wasm.

The full experimental protocol (roles, lifecycle, decision rule and reproducibility requirements) is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Development

```bash
make install        # uv sync
make test           # pytest (CPU only, no downloads)
make lint           # ruff + the 500-line limit
make check          # registry consistency (what CI's "Experiment records" job runs)
make sample-run     # build dist/ from a few real records in the wiki
make build          # build dist/ from the wiki archive, plus the Pagefind index
openinterp deploy --dist-dir dist --bucket my-r2-bucket [--delete] [--dry-run]
```

`dist/index.html` opens directly from disk; search and the SQL console need HTTP (`make serve`). The default agent model is `$OPENINTERP_AGENT_MODEL`, falling back to `anthropic/claude-opus-5-5`, and `review` is an alias for `verify`. If the agents answering prompt files sit behind a proxy that hides the real model (e.g. Claude Code routed through LiteLLM to Grok), set `OPENINTERP_MODEL_ALIASES` so records name the model that answered: `claude-haiku-5-5=grok-4-fast,claude-opus-5-5=grok-4`, or one bare name for every `external/` model.

## Experiment wiki

Every completed experiment gets a Markdown page and an archived copy of its full record (`<id>.json`) in [`wiki/`](wiki/), whatever its verdict. `openinterp verify` and `openinterp reevaluate` write it automatically, and `openinterp wiki` backfills pages for existing records. Each page records the hypothesis, top examples, gate results, probes, counterfactuals and next steps based on the gates that failed. A page is written once and never regenerated, so its Discussion section is safe to edit. Each neuron also gets a history page (`README.md` in its folder), and `wiki/README.md` indexes every experiment. Both are generated by `openinterp wiki` and not committed, so run it after pulling to browse them locally; the site shows the same history.

## Pull requests

[CI](.github/workflows/ci.yml) runs two required jobs on every PR:
- **Lint and tests:** ruff, a 500-line limit on source and documentation files (tests, configuration and generated data are exempt), and pytest, with CPU-only torch (`scripts/ci-install.sh`).
- **Experiment records:** `openinterp check`. Every experiment record (`wiki/experiments/<slug>/<id>.json`) must validate, have a verdict, sit at the path its neuron and id dictate, name only an earlier experiment on the same unit in `supersedes`, and have a page whose front matter matches it; every page must have a record. Generated indexes (`README.md`) must not be committed, existing records are write-once, and existing pages may only change in their Discussion section. The retired `audits/` directory must be empty.
- **Reproduce records:** `openinterp reproduce --base-ref origin/main`, on pull requests only. It re-runs every record the pull request adds or changes against the model commit it pins, on CPU, and fails if any activation, result, gate or status differs from the record (docs/PROTOCOL.md §5). Pull requests without record changes skip it (a skipped job counts as passing). Reproductions run one at a time across all pull requests, queued in order, and a newer push to a pull request cancels its older run. Only models in the job's `ALLOWED_MODELS` are downloaded, and the weights are cached between runs. Expect a few minutes per record for Qwen2.5-0.5B. Records signed by a maintainer are skipped: run `openinterp sign wiki/experiments/<slug>/<id>.json` and commit the `.sig` file next to the record. The default key is `~/.ssh/openinterp_signing` (or `$OPENINTERP_SIGNING_KEY`, or `--key`): a key used only for signing, so it can be revoked without touching your logins. `verify`, `review` and `reevaluate` sign each new experiment automatically when that key exists and is listed in `.github/allowed_signers`, so your own runs (including `make loop`) skip CI reproduction; set `OPENINTERP_AUTO_SIGN=0` to turn this off. Only keys in `.github/allowed_signers` on `main` count; add yours there first, in its own commit to `main`:

  ```bash
  ssh-keygen -t ed25519 -N "" -C "openinterp record signing" -f ~/.ssh/openinterp_signing
  echo "<github-user> namespaces=\"openinterp-record\" $(cat ~/.ssh/openinterp_signing.pub)" >> .github/allowed_signers
  ```

Run `make check` locally before pushing. The [PR template](.github/pull_request_template.md) has a checklist for posting experiments.

## Claude Code skills

`.claude/skills/` holds research skills that Claude Code loads automatically in this repo. Each one condenses a set of papers into a procedure that uses the `openinterp` harness:

| Skill | Use it to | Based on |
| --- | --- | --- |
| `choose-next-experiment` | Pick the next experiment with `openinterp next`: randomised, explore vs exploit, herd-resistant | ε-greedy and Boltzmann sampling over follow-ups |
| `land-experiment` | Run experiments end to end and merge them: `choose-next-experiment`, then build, wait for green CI and merge (with write access), else move on | — |
| `native-neuron-audit` | Avoid misreading a neuron (context, data, poles, SwiGLU regimes) | Goodfire, *Already Clean* (2026) |
| `vocab-projection-lookahead` | Decide retrospective vs lookahead | Gurnee et al. 2026 (Jacobian lens); Geva et al. 2021/2022; logit lens; Voita et al. 2023 |
| `neuron-family-triage` | Pre-classify neurons from weight statistics | Gurnee et al. 2024; Stolfo et al. 2024; Voita et al. 2023 |
| `explanation-simulation-scoring` | Score explanations by predicted activations | Bills et al. 2023; Huang et al. 2023 |
| `activation-patching-practice` | Choose causal metrics and corruptions; handle self-repair | Meng et al. 2022; Zhang & Nanda 2023; Heimersheim & Nanda 2024; McGrath et al. 2023; Rushing & Nanda 2024; AtP* |
| `neuron-steering-tests` | Test a hypothesis by clamping the neuron and generating | Turner et al. 2023; Goodfire 2026 |
| `sparse-probing-search` | Find neurons for a labelled concept | Gurnee et al. 2023; Bolukbasi et al. 2021 |
| `interpretability-illusions-review` | Write probes as the reviewer; spot illusions | Bolukbasi et al. 2021; Huang et al. 2023; Makelov et al. 2023; Sun et al. 2024 |
| `attention-head-audit` | Extend the registry to attention heads and transcoder baselines | Olsson et al. 2022; Wang et al. 2022; Conmy et al. 2023; Dunefsky et al. 2024; Paulo et al. 2025 |
| `single-source-of-truth` | Define each constant, threshold or list once; pin copies that must exist with a test | — |

## Layout

| Path | Purpose |
| --- | --- |
| `openinterp/core/schemas.py` | Strict Pydantic v2 models: `NeuronIdentifier` / `HeadIdentifier` / `FeatureIdentifier` (in `core/identifiers.py`), `HeadProfile` / `DictionaryProfile` (in `core/profiles.py`), `CounterfactualPair`, `LookaheadProfile`, `CausalMetrics`, `NeuronAuditSubmission`, `VerificationVerdict`, `AuditRecord` |
| `openinterp/harness/hooks.py` | `NeuronHookRunner`: forward hooks on the MLP down-projection input (the neuron basis) for Llama/Qwen/Mistral/GPT-2/NeoX/OPT/Phi; pole isolation; corpus scanning |
| `openinterp/harness/scan.py` | The audit's corpus scan: windows, top examples, partners and confounds in one pass, over chunks of whole-layer activations |
| `openinterp/harness/layer_cache.py` | On-disk cache of those chunks per model, layer and corpus sample, which later scans and `openinterp activations` replay |
| `openinterp/harness/units.py` | What strength and ablation mean for each unit: a neuron pole, an attention head measured from its mean output, or a dictionary latent |
| `openinterp/harness/heads.py` | Head scan (mean output, top examples with peak attention, pattern scores) and head patching |
| `openinterp/harness/dictionaries.py` | Published transcoders/SAEs (presets, pinned weights, how each is read) |
| `openinterp/harness/sites.py` | Where a dictionary reads its input and writes its output: the hooks behind `dictionary_interventions` |
| `openinterp/harness/environment.py` | The device, dtype, model revision and library versions recorded with each run |
| `openinterp/harness/features.py` | Dictionary-latent scan (with the dictionary's fit) and latent patching |
| `openinterp/harness/patcher.py` | Denoising/noising activation patching and resample ablation per neuron pole, optionally freezing downstream MLPs |
| `openinterp/harness/attribution.py` | Attribution patching (Δactivation × gradient) for whole layers in one backward pass |
| `openinterp/harness/lookahead.py` | Pole-ablation KL profile and unembedding projection of the neuron's output direction |
| `openinterp/agents/` | `ProposerAgent` and `ReviewerAgent` (structured outputs from LiteLLM or any external LLM via prompt files); deterministic `evaluate_submission` |
| `openinterp/builder/` | Jinja2 + Tailwind static site, `registry.parquet` export, DuckDB-computed leaderboards |
| `openinterp/sync/r2.py` | MD5/ETag delta sync to R2 with cache headers |

### Definitions

- **Neuron activation:** the input to the MLP's down/output projection. For gated MLPs this is `act(gate(x)) * up(x)`.
- **Pole:** the positive pole is `clamp(a, min=0)` and the negative pole is `clamp(a, max=0)`. Pole *strength* is always reported as a non-negative magnitude.
- **Attention head:** one head's slice of the attention output projection's input. Its strength is `|W_O (z − z̄)|`, with z̄ its mean output over the scan.
- **Counterfactual Δ:** `(peak strength on base − peak strength on counterfactual) / max activation`.

### Models

The models `openinterp next` explores (`[tool.openinterp] models` in `pyproject.toml`). What a pole means depends on the MLP's activation.

| Model | MLP | Negative pole | Scan |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | SwiGLU, `silu(gate(x)) * up(x)` | Unbounded: `up(x)` carries the sign | Whole documents |
| `openai-community/gpt2` | GELU, `gelu_new(c_fc(x))` | GELU's dip, at most 0.170 in every layer and reached at a pre-activation of about −0.75. 73–88% of tokens are below zero, and a strongly negative input reads as inactive | 128-token windows (`[tool.openinterp.window_tokens]`), matching its transcoders. GPT-2 itself reads 1,024 positions |

GPT-2's final norm is `transformer.ln_f` and its MLP norm `ln_2`, which the transcoders' `mlp_in_normalized` site standardises.

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

### Deploy

[deploy.yml](.github/workflows/deploy.yml) rebuilds `dist/` from `main` on every push to `main` (plus every 6 hours as a backstop, and on manual dispatch), runs `openinterp check`, and runs `openinterp deploy --delete`. It deploys on push because GitHub's scheduled workflows are best-effort: an hourly cron alone left the site hours behind. **An unchanged registry deploys as a no-op**, which takes four measures:

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

Early scaffold. `make sample-run` builds the site from six real experiments in `wiki/experiments/` (listed in the `Makefile`), imported with `openinterp wiki --import-dir`.
