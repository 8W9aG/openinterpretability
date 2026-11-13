# Development

How to change the code, what CI checks on a pull request, where the code lives, and how the site is built and deployed. To run an experiment, see the README's [Quickstart](../README.md#quickstart) and [experiments.md](experiments.md).

## Commands

```bash
make install        # uv sync
make test           # pytest (CPU only, no downloads)
make lint           # ruff (incl. docstrings), the 500-line limit, JSON and YAML, docstring coverage, Markdown, Vale, STE
make vale           # comment and docstring prose against our STE rules (needs Vale)
make check          # registry consistency of this branch's changes (what CI's "Experiment records" job runs on a PR)
make check-all      # the same for every record (what it runs on main)
make sample-run     # build dist/ from a few real records in the wiki
make build          # build dist/ from the wiki archive, plus the Pagefind index (about 1 min with caches)
openinterp deploy --dist-dir dist --bucket my-r2-bucket [--delete] [--dry-run]
```

`openinterp build` renders the unit pages in one worker process per CPU (`--jobs`, where the platform can fork). `--jobs 1` renders them in one process. The build keeps the personal-data matches of each text on disk, in `~/.cache/openinterp/pii`. So a build searches only new text. `OPENINTERP_PII_CACHE` moves the cache, and `off` turns it off. The deploy keeps this cache between runs. The significance of each record is kept the same way, in `~/.cache/openinterp/build` (`OPENINTERP_BUILD_CACHE`). The two Parquet exports run in forked processes while the pages render. The build gives the same files on every run, whatever the hash seed.

`make lint` and `make vale` need [Vale](https://vale.sh), which is not a Python package: `brew install vale` on macOS, or a release binary from vale.sh. `dist/index.html` opens directly from disk. Search and the SQL console need HTTP (`make serve`). No step needs Node.js: Pagefind is a locked Python dev dependency.

## Pull requests

[CI](../.github/workflows/ci.yml) runs three required jobs on every PR:

- **Lint and tests:** ruff, a 500-line limit on source and documentation files (tests, configuration, papers and generated data are exempt), and pytest, with CPU-only torch (`scripts/ci-install.sh`). Lint also checks the JSON, JSONL and YAML files (`scripts/check_data_files.py`, with yamllint). A pull request that changes only `wiki/` (an experiment) runs the same code as `main`, so it skips this job (a skipped job counts as passing).
- **Experiment records:** `openinterp check`. On a pull request that changes only `wiki/`, the record and page rules read only the files it adds or changes (`--base-ref`). If it changes anything else, and on `main`, they read every file. Every experiment record (`wiki/experiments/<slug>/<id>.json`) must match the schema, have a verdict and sit at the path its neuron and id dictate. It must name only an earlier experiment on the same unit in `supersedes`, and have a page whose front matter matches it. Every page must have a record. New records must have `model_num_layers` and `model_mlp_width`, and `model_n_heads` on head records only. Generated indexes (`README.md`) must not be committed, existing records are write-once, and existing pages may only change in their Discussion section. The retired `audits/` directory must be empty.
- **Nightly reproduction:** `openinterp reproduce --sample 20`, on `main`. It re-runs a rotating sample of all records, signed ones too, so a library upgrade that shifts activations is noticed. It starts no record after 140 minutes (`--time-budget`), and the records it didn't reach don't fail it. A failure opens an issue (docs/PROTOCOL.md §5).
- **Reproduce records:** `openinterp reproduce --base-ref origin/main`, on pull requests only. It re-runs every record the pull request adds or changes against the model commit it pins, on CPU. It fails if any activation, result, gate or status differs from the record (docs/PROTOCOL.md §5). Pull requests without record changes skip it (a skipped job counts as passing). Reproductions run one at a time across all pull requests, queued in order, and a newer push to a pull request cancels its older run. Only models in the job's `ALLOWED_MODELS` are downloaded, and the weights are cached between runs. Expect a few minutes per record for Qwen2.5-0.5B. A record in bfloat16 or float16 is reproduced in float32, under wider rules (docs/PROTOCOL.md §5). Records signed by a maintainer are skipped: run `openinterp sign wiki/experiments/<slug>/<id>.json` and commit the `.sig` file next to the record. The default key is `~/.ssh/openinterp_signing` (or `$OPENINTERP_SIGNING_KEY`, or `--key`): a key used only for signing, so it can be revoked without touching your logins. `verify`, `review` and `reevaluate` sign each new experiment automatically when that key exists and is listed in `.github/allowed_signers`. So your own runs (including `make loop`) skip CI reproduction. Set `OPENINTERP_AUTO_SIGN=0` to turn this off. Only keys in `.github/allowed_signers` on `main` count. Add yours there first, in its own commit to `main`:

  ```bash
  ssh-keygen -t ed25519 -N "" -C "openinterp record signing" -f ~/.ssh/openinterp_signing
  echo "<github-user> valid-after=\"$(date -u +%Y%m%d)Z\",namespaces=\"openinterp-record\" $(cat ~/.ssh/openinterp_signing.pub)" >> .github/allowed_signers
  ```

A PR that changes only files under `wiki/` is squash-merged automatically once all three pass ([auto-merge.yml](../.github/workflows/auto-merge.yml)), fork PRs included. So is a PR that the maintainer opened from a branch of this repository, whatever it changes. Agents that use the maintainer's GitHub login open PRs as the maintainer, so their PRs merge this way too. Any other PR that changes anything else (code, workflows, `.github/allowed_signers`) waits for a maintainer.

Two more checks run on every pull request but aren't required. *Secret scan* runs gitleaks over the commits the PR adds (`.github/workflows/secret-scan.yml`, rules in `.gitleaks.toml`). Run it by hand to scan the whole history. Workflows pin each action to a full commit SHA, and Dependabot (`.github/dependabot.yml`) opens PRs that update the pins and `uv.lock`.

Run `make check` locally before pushing. The [PR template](../.github/pull_request_template.md) has a checklist for posting experiments.

## Layout

| Path | Purpose |
| --- | --- |
| `openinterp/core/schemas.py` | Strict Pydantic v2 models: `NeuronIdentifier` / `HeadIdentifier` / `FeatureIdentifier` (in `core/identifiers.py`), `HeadProfile` / `DictionaryProfile` (in `core/profiles.py`), `CounterfactualPair`, `LookaheadProfile`, `CausalMetrics`, `NeuronAuditSubmission`, `VerificationVerdict`, `AuditRecord` |
| `openinterp/harness/hooks.py` | `NeuronHookRunner`: forward hooks on the MLP down-projection input (the neuron basis) for Llama/Qwen/Mistral/GPT-2/NeoX/OPT/Phi, pole isolation and corpus scanning |
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
| `openinterp/agents/` | `ProposerAgent` and `ReviewerAgent` (structured outputs from LiteLLM or any external LLM via prompt files), and deterministic `evaluate_submission` |
| `papers/` | Papers, one folder each: the write-up as `README.md`, its results and its media. The site publishes them as its Research section |
| `openinterp/benchmark.py`, `benchmark/` | The explainer benchmark ([benchmark.md](benchmark.md)): freezing, sealing and scoring, and its frozen versions |
| `openinterp/builder/` | Jinja2 + Tailwind static site, `registry.parquet` export, DuckDB-computed leaderboards |
| `openinterp/sync/r2.py` | MD5/ETag delta sync to R2 with cache headers |

## Static site and data

The site, served at [open-interpretability.com](https://open-interpretability.com/), is built from the wiki's experiment archive (`wiki/experiments/**/<id>.json`). That archive holds every experiment, including null and superseded results.

```text
dist/
├── index.html                                   # totals, models, most conclusive results, verified, SQL console
├── models/<model>/index.html                    # layer coverage
├── models/<model>/L<layer>/index.html           # neurons studied in the layer
├── models/<model>/L<layer>/N<idx>.html          # every experiment (both poles), most conclusive first
├── data/registry.parquet            # one row per experiment; latest, max-age=3600
├── data/snapshots/registry-<sha>.parquet         # content-addressed, immutable
├── data/manifest.json
├── data/neuronpedia/v1/<model>/<source>/explanations/batch-0.jsonl.gz  # VERIFIED units, Neuronpedia's format
└── pagefind/                        # written by `python -m pagefind --site dist` (make search-index)
```

`data/neuronpedia/` holds the lead experiment of each VERIFIED unit in Neuronpedia's export layout, so a self-hosted Neuronpedia can import it (`openinterp/builder/neuronpedia_export.py`). A dictionary latent goes to the source Neuronpedia hosts it under, with the same index. A native neuron goes to `<layer>-openinterp-neurons`, with its pole. Only models that Neuronpedia knows are exported, today GPT-2 small. Each row links to its signed record and carries the data licence. Nothing is sent to Neuronpedia.

Cache policy: content-addressed files (`data/snapshots/*` and Pagefind shards) are served with `max-age=31536000, immutable`. Everything else, including HTML and `registry.parquet`, gets `max-age=3600`. Deploys upload assets before HTML, so a page never references an object that has not arrived yet.

R2 credentials are read from `CF_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.

## Deploy

[deploy.yml](../.github/workflows/deploy.yml) rebuilds `dist/` from `main` on every push to `main` (plus every 6 hours as a backstop, and on manual dispatch), runs `openinterp check`, and runs `openinterp deploy --delete`. It deploys on push because GitHub's scheduled workflows are best-effort: an hourly cron alone left the site hours behind. **An unchanged registry deploys as a no-op**, which takes four measures:

- **Deterministic build.** There are no build timestamps. The footer date and `manifest.json`'s `updated_at` come from the latest verdict, and every ordering is total. Unchanged records rebuild byte-for-byte, which a test enforces.
- **Content comparison.** Only files whose MD5 differs from the remote ETag are uploaded.
- **Derived index.** Pagefind's output is not byte-stable between runs because it orders filters from a hash map. `pagefind/` is therefore only synced when an HTML or data file actually changed.
- **Targeted purges.** When something does change, only the changed short-cache URLs are purged from Cloudflare's cache. Content-addressed files get new names, and `data/snapshots/` is never deleted.

Configure the workflow in the repo's GitHub settings:

| Name | Kind | Purpose |
| --- | --- | --- |
| `CF_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | secrets | R2 S3 API access |
| `R2_BUCKET` | variable | Bucket name |
| `CF_ZONE_ID`, `CF_API_TOKEN` | secrets (optional) | Cache purge for a custom domain. The token needs *Zone → Cache Purge* |
| `OPENINTERP_SITE_URL` | variable (optional) | Public URL of the bucket root, used to build the purge URLs. Without it, `openinterp deploy` uses `[project.urls] Homepage` in `pyproject.toml`, which the site's canonical URLs use too |
