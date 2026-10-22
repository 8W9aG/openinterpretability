# TODO

Open work, roughly in priority order. Move items to `CHANGELOG.md` under `[Unreleased]` when they land. Items marked **(protocol)** change the decision rule and must bump `PROTOCOL_VERSION` (`docs/PROTOCOL.md` §6).

## Protocol and science

- [ ] **(protocol)** Decide what a pole's status is when its experiments disagree. Today it is the newest experiment's (`builder/tree.py`), so a weaker or self-reviewed rerun overturns a cross-model VERIFIED. Of the 11 poles with more than one experiment, 5 changed verdict: L16/N2071+ went VERIFIED → NEEDS_REVISION, and L11/N4255+ went NR, NR, V, V, NR. Options: the newest experiment under the current protocol, preferring independent reviews; or a "contested" status when recent experiments disagree. Show the disagreement on neuron pages and report reviewer agreement across the registry.
- [ ] Scan both poles by default and show them together on neuron pages.
- [ ] Positive control: audit known induction heads or other well-documented units and confirm the protocol VERIFIES them. Protocol 0.6 can audit heads; a dry run on GPT-2 small L5H1 (induction score 0.89; proposer and judges Claude Opus/Haiku 5.5, reviewer Sonnet 5.5) was NEEDS_REVISION twice, failing only specificity. Every holdout fired, every counterfactual reproduced, causal scores were 1.0 against the layer's other heads, and the record reproduced. The misses: (1) the head also fires on fuzzy and template repeats ("The dog … the dog", `x = a + b; y = c + d`), which an exact-repeat explanation excludes; (2) the reviewer labelled probes by words, not GPT-2 tokens, so exact repeats like ". My … . My" were marked as hard negatives. Next: a third revision that describes the fuzzy matching, run as a registry experiment, and decide whether reviewer prompts for heads should state the token-level matching rule.
- [ ] **(protocol)** Decide whether steering predictions and simulation scores should become gates. Both need a pre-registered prediction format first.
- [ ] Report registry-wide statistics (share VERIFIED, share dead) over `selection = "random"` records only.
- [ ] Record dead neurons found by random selection instead of discarding them.
- [ ] Null calibration: run the full pipeline on targets that should fail and report the VERIFIED rate as the protocol's false-positive rate. Cases: a hypothesis written for neuron A scored against neuron B, a shuffled-label hypothesis, and "neurons" in a randomly rotated MLP hidden basis. The rotated basis also tests the native-basis premise directly: if rotated units verify about as often as real ones, the neuron basis isn't privileged (Elhage et al., 2021). With many agents, neurons and gates, registry-level claims need this false-positive rate, or a multiple-comparisons correction, before they mean anything.
- [ ] Reviewer-effect crossover: review a fixed set of submissions with 2–3 reviewer models each and compare verdicts. Verdict rates differ a lot by model pair (Opus → Sonnet: 20 of 24 VERIFIED; deepseek → deepseek: 14 of 38), but proposer and reviewer are confounded in the registry, so it can't say whether the reviewer or the hypothesis decides the verdict. Null calibration measures the false-positive rate; this measures how much of the verdict is the reviewer.
- [ ] Ground-truth calibration: audit models whose mechanisms are known by construction, and check that agents recover each neuron's role. Candidates: Tracr-compiled transformers, InterpBench (Gupta et al., 2024), and a small model fine-tuned to plant a known neuron. This complements the induction-head positive control: that control checks the protocol can say yes, while this checks that the agent's explanation is the *right* one.
- [ ] **(protocol)** Hypothesis-conditioned resample ablation, i.e. causal scrubbing for one neuron (Chan et al., 2022). Replace the pole's activation with its value on another token the hypothesis labels the same, and separately with its value on a token labelled differently. If the hypothesis captures what downstream layers use, the first barely changes the loss and the second does. This needs the hypothesis as an executable labeller (the simulator in `explanation-simulation-scoring`). It would also answer §4's open point that `next_token_kl` shows a neuron matters, but not *for the reason the hypothesis gives*.
- [ ] **(protocol)** Output-side explanations: require a hypothesis to say what the pole *does* (its effect on the next tokens or on the text) as well as what activates it, and score that side by intervention. Steer the pole, then check whether a judge given the explanation can tell steered text from unsteered text (Delphi's intervention scoring, Paulo et al., 2025; output-centric descriptions, Gur-Arieh et al., 2025). Today only lookahead claims make an output-side prediction.

## Analyses per experiment

Methods from the wider interpretability literature that the registry doesn't use yet. Each one starts as a recorded number (reported, not gated, as in `docs/PROTOCOL.md` §4 "What is not evidence (yet)") and becomes a gate only through a protocol change.

- [ ] Verbalise the pole's output direction with a Patchscope or SelfIE prompt (Ghandeharioun et al., 2024; Chen et al., 2024). Inject `w_out`, scaled, into an inspection prompt and record the decoded text. Where a trained Activation Oracle exists for the model (Karvonen et al., 2025), record its answer too. Both are blind, example-free descriptions: use them as an extra proposer baseline (does the oracle's description verify as often as the agent's?), and never show them to the reviewer.
- [ ] Wiring: the pole's top upstream and downstream connections. Two sources:
  - *weights*: virtual weights (Elhage et al., 2021) from attention heads' OV circuits and earlier neurons' `w_out` into the pole's `w_in`, and from its `w_out` into later neurons' `w_in` and heads' Q/K/V, with the norm folded in or approximated;
  - *activations*: edge attribution patching on the reviewer's probes (EAP, Syed et al., 2023; EAP-IG, Hanna et al., 2024).

  Link edges to other registry entries. This is the first step toward circuit pages, and it overlaps with `ComponentIdentifier`.
- [ ] Linear-probe comparison for VERIFIED concepts. Using labelled data from `sparse-probing-search`, train logistic and mass-mean probes (Marks & Tegmark, 2023) on the layer's MLP hidden activations and on the residual stream. Report:
  - probe accuracy against the neuron alone;
  - the cosine between the probe direction and the neuron's basis vector;
  - the k-sparse probing curve, i.e. how many neurons the concept needs (Gurnee et al., 2023).

  This shows whether a concept is localised in the neuron or only partly carried by it.
- [ ] Causal-abstraction check (Geiger et al., 2024; `stanfordnlp/pyvene`). For a hypothesis that names a variable ("inside a quotation"), compare the interchange-intervention accuracy of swapping the pole between counterfactual pairs with that of a learned one-dimensional DAS direction for the same variable. The probe comparison above is correlational; this is its causal counterpart. A neuron close to the DAS ceiling is the model's variable; one far below it is a partial view of it.
- [ ] Multi-dimensional features (Engels et al., 2024): check whether the pole and its partner poles together trace a circular or ordinal manifold, such as days of the week, months or small numbers. Fit PCA to their joint activations on the activating set. Allow hypotheses of the form "one coordinate of a k-dimensional feature" so these neurons aren't forced into a single-concept story and rejected.
- [ ] Adversarial input search, the text analogue of feature visualisation. Optimise prompts to maximise, or exactly hit, the pole's activation, using GCG/EPO-style fluent optimisation (Thompson et al., "Fluent dreaming", 2024), and check the hypothesis on the results. This automated reviewer arm hunts for false positives and negatives an LLM reviewer wouldn't think of. Record how many optimised inputs the hypothesis mispredicts.

## Pipeline

- [ ] Recover the scan corpus's provenance. 77 of 79 records scanned a `corpus.jsonl` (SHA-256 `ded48cfc…`) that isn't in git and has no `corpus.source.json`, so their `provenance.corpus_source` is empty. `reproduce` copes because it re-measures stored examples, but nobody else can re-run those scans, histograms or `max_activation`. Find out which `openinterp corpus` call (or older snippet) produced it and publish a sidecar, or publish the corpus itself once its licence is checked. Then make `audit` warn on, or refuse, a corpus with no matching sidecar.
- [ ] Backfill explanation scores. Only 2 of 79 records have them, so detection and fuzzing can't be compared with verdicts across the registry. Add a planner draw that runs `score` and `reevaluate --explanation-scores` on records that lack them, VERIFIED first, then check whether VERIFIED predicts held-out detection. That check is a prerequisite for deciding whether simulation scores should become a gate (Protocol and science).
- [ ] Record the human contributor behind each experiment, not just the agent model: an optional `contributor` (a public handle, e.g. GitHub username, never an email) on the submission and on the verdict, so proposer and reviewer can differ. Fill it from a `--contributor` flag or `OPENINTERP_CONTRIBUTOR`, falling back to nothing rather than reading git config silently. Export it to Parquet and add a contributor leaderboard next to the home page's "Submitting agents" table (experiments, VERIFIED, well-probed rejections, reviews done), so people get credit as well as models.
- [ ] Show the proposer what the registry knows about a pole's covariants. The scan finds co-activating partner poles (`correlated_neurons`) and the proposer prompt lists them, but only as IDs with lift and ρ (`agents/proposer.py`). Look each partner up in the registry and add its latest verdict and hypothesis, plus its top trigger tokens when it hasn't been studied. A VERIFIED partner is a strong hint about what the pole shares with it, or what it must be distinguished from. The site's neuron pages already link studied partners (`builder/compiler.py`).
- [ ] Let `openinterp next` explore attention heads. Explore draws always run `openinterp audit --model M --random`, which samples an MLP neuron pole unless given `--unit head`, and the registry has no head records for follow-ups to start from, so heads are only ever audited by hand. Draw the unit kind (neuron pole or head) as its own step, like the model draw: weighted by coverage, or by a configured share in `[tool.openinterp]`, never by unit count (Qwen2.5-0.5B has 336 heads against about 233k neuron poles, so heads would almost never come up). Wait for the induction-head positive control (Protocol and science) to VERIFY first: until reviewers label head probes by tokens, random head audits would mostly measure that problem.
- [ ] `openinterp loop`: run experiments back to back from the Python CLI, with no Claude Code session driving them, so the loop works wherever `openinterp` is installed (including `uvx openinterp` once it is on PyPI) and `scripts/experiment-loop.sh` can become a thin wrapper or go away. What each iteration does:
  - Reset a detached worktree to `origin/main`.
  - Draw with the planner (`--epsilon`, as in `openinterp next`).
  - Claim the draw the way `choose-next-experiment` does: branch `experiment/<key>[-<kind>]`, an empty "Claim" commit, and a draft PR via `gh`.
  - Run the draw's `audit`/`verify` commands in-process with the `--proposer` and `--reviewer` LiteLLM models. Reject `external/<name>`, since nobody would be there to answer it.
  - Run `openinterp wiki` and `openinterp check`, commit `wiki/` (with any `.sig`), push, and mark the PR ready. On an error, close the PR and give the reason.
  - Land it as the `land-experiment` skill does: wait for green checks, then merge if the PR only touches `wiki/` and the token has write access; otherwise leave it open and move on.

  Loop behaviour and options:
  - No delay between iterations.
  - `--max N`, where 0 means run until stopped.
  - Stop after repeated failures, or after backing off on a provider rate or usage limit.
  - One log per iteration in `logs/loop/`, and the same `RESULT: <slug> <verdict> <PR url>` line the script prints.

  Draws that need judgement, such as `revise` (narrowing a claim, fixing failed gates), must give the proposer the previous record and its failed gates. Enforce a reviewer different from the record's proposer for `independent_review`/`replicate` draws, re-drawing once if not. Keep the model loaded across iterations, so experiments don't each reload weights as the per-session script does. Test it with a temporary git repo, a stubbed `gh` and fake agents.
- [ ] A scheduled hourly build and deploy (CI cron), as described in the vision.
- [ ] Expire stale claims: a scheduled GitHub workflow (daily cron, plus `workflow_dispatch` with a dry-run input) that releases claims abandoned for 3 days. A claim is an `experiment/*` branch whose only commits ahead of `main` are the empty "Claim" commit, so it changes no files. When the branch's last commit and its PR's last update are both more than 3 days old, close the draft PR with a comment saying why and how to reclaim, then delete the branch. Leave alone any branch that has a file change, or a PR that someone has commented on or marked ready. A fork PR's branch can't be deleted from here, so only close its PR. Stale claims otherwise stay in the planner's claim weighting and keep agents off those neurons. Needs `contents: write` and `pull-requests: write`.
- [ ] Make the repo's GitHub Wiki (`8W9aG/openinterpretability.wiki.git`) the only home of the wiki pages: experiment pages, neuron history and the index. The records (`<id>.json` and `.sig`) stay in this repo. Records are data, and they keep going through PRs, `openinterp check`, reproduction and signing. A GitHub Wiki can't do that: it has no PRs, branch protection or required checks, and only collaborators can push to it, so fork contributors couldn't submit. The work:
  - Stop committing pages. `verify`, `review` and `reevaluate` write only the record. Remove the `.md` pages from `wiki/` (the generated `README.md` indexes are already gitignored). Consider renaming `wiki/experiments/` to `records/`, updating the site build, Parquet export, `reproduce`, `next` and the planner to match.
  - Add `openinterp wiki --github-wiki <dir>`, which renders pages from the records into a checkout of the wiki repo. The GitHub Wiki gives every page a flat name, so:
    - give each page a unique one, e.g. `Qwen--Qwen2.5-0.5B-L13-N1706-positive` for neuron history and `…-20261020T224903Z` for an experiment;
    - link records and `docs/` to their files on `main`;
    - generate `Home.md` (the index) and a `_Sidebar.md` by model and layer.
  - Keep pages write-once. The export creates a page for each new record and never overwrites an existing one, so Discussion edits made on the wiki survive. Only the index, sidebar and neuron history pages are regenerated, and neuron history pages get their own Discussion section, which is carried over.
  - Publish on merge: a step in the deploy workflow, on push to `main`, runs the export and pushes to the wiki repo. Check whether `GITHUB_TOKEN` can push there, or a deploy key is needed. The wiki has to be initialised once by hand, by creating its first page.
  - Guard edits:
    - restrict wiki editing to collaborators;
    - add a workflow on the `gollum` event that diffs each edit and reverts any change outside a Discussion section, keeping today's "only Discussion may change" rule;
    - since this check runs after the edit, rely on the records in the repo, not the pages, for anything that must be trusted.
  - Simplify CI: drop the record/page consistency and page-immutability checks from `openinterp check`. Move existing Discussion text from the repo pages into the wiki before deleting them. Update the README, `docs/PROTOCOL.md`, the PR template and `choose-next-experiment`, which tell contributors to commit pages.
- [ ] Review [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for ideas or reuse: its standardised hook points across many architectures (compare with `openinterp/harness/hooks.py`), activation caching (for faster scans), patching utilities (for the causal metric), attention-head hooks (for `ComponentIdentifier`) and weight processing such as LayerNorm folding (which changes what a neuron's weights look like). Decide whether to adopt it as a dependency, borrow ideas, or keep the current Hugging Face hooks.
- [ ] Review the rest of the tooling stack the same way:
  - [nnsight](https://github.com/ndif-team/nnsight) with NDIF remote execution, for models too large for contributors' hardware (Llama 3.1 70B/405B). `reproduce` would need to run against the same backend.
  - [pyvene](https://github.com/stanfordnlp/pyvene), for interchange interventions and DAS.
  - [SAELens](https://github.com/decoderesearch/SAELens) and [circuit-tracer](https://github.com/safety-research/circuit-tracer), for loading published SAEs and transcoders in the baseline arm.
  - [Delphi](https://github.com/EleutherAI/delphi), for explanation scoring.
- [ ] Branding for the generated site and repo: a logo and wordmark to replace the `⊕` placeholder in `base.html`, a favicon, a colour palette and type defined once in the Tailwind config (currently ad hoc emerald/slate), Open Graph/social preview images for shared links, and the same mark on the README and GitHub social preview.
- [ ] An About page in the generated site (`openinterp/builder/templates/about.html`, rendered by `compiler.py` and linked from the `base.html` header): the project's premise, how the proposer/reviewer protocol decides a verdict (linking `docs/PROTOCOL.md` and the current `PROTOCOL_VERSION`), what VERIFIED does and doesn't mean, how to contribute, and the licence.
- [ ] Google Colab guide (a `docs/colab.md` page plus a ready-to-run notebook with an "Open in Colab" badge in the README): install from git (or PyPI once published) without clobbering Colab's CUDA torch, pick a GPU runtime and `--device cuda`, store API keys and `HF_TOKEN` in Colab secrets (or use the two-step answer files with no keys), build the scan corpus, note which models fit on a free T4 and when to use `--dtype bfloat16`/`float16`, and save results by pushing a branch or copying `wiki/` to Drive before the runtime resets.
- [ ] Publish `openinterp` to PyPI so `pip install openinterp` (or `uvx openinterp`) runs the whole pipeline without cloning the repo. Check the name is free on PyPI; confirm the wheel ships the templates and `data/sample_corpus.txt`; move `pagefind[bin]` out of the dev group (or into an extra) so `openinterp build` works from an install; let users choose a CPU/CUDA torch; make commands that assume the repo layout (`submissions/`, `wiki/`, `exchange/`) take paths or work from any directory; and add a release workflow that builds and publishes on a version tag via PyPI trusted publishing.

## Performance

Make each experiment faster and cheaper in tokens without changing what it measures. Baseline from the 2026-10-20 loop (claim commit to result commit in git history):

- **Time:** an explore draw takes 25–40 min, and a replicate that skips the scan about 5 min. The CPU corpus scan is roughly 13–15 min of an explore draw.
- **Prompt sizes:** proposer prompts (`ev.prompt.md`) are 57–62 KB, about 15k tokens. Reviewer prompts are about 30 KB, and the scan evidence (`ev.json`) about 500 KB.

Measure before optimising, and check that any change to what the agents see doesn't move verdict rates.

- [ ] Use the GPU on Macs: `--device auto` picks CUDA or CPU and never MPS (`openinterp/cli.py`), so scans on Apple silicon run on CPU. Add MPS to `auto` (checking float32 results match CPU within tolerance) and batch long documents by token budget rather than a fixed `batch_size`.
- [ ] Load the model once per experiment. `audit` and `verify` run as separate processes, and each loads the model; the scan, lenses, probe measurement and gates could share one load. `openinterp loop` (Pipeline) extends this across experiments. Batch the reviewer's probes and counterfactuals into as few forward passes as their lengths allow.
- [ ] Trim the proposer prompt.
  - Top-activating snippets currently include whole documents: the first example for L13/N1706+ is an entire film review. Show a window of a fixed number of tokens around the peak instead.
  - Drop near-duplicate snippets.
  - Cap the token-strip and lens lists at what the hypothesis actually uses.
  - Do the same pass on the reviewer prompt.
  - A/B the trimmed prompts against the current ones on a fixed set of neurons before switching.
- [ ] Order prompts for provider prompt caching. Put the stable parts (system prompt, methodology, answer schema) first and the neuron's evidence last, and pass cache-control hints through LiteLLM, so repeated experiments reuse the cached prefix.
- [ ] Cut the driving session's tokens. In Claude Code, the driver reads skills, docs and large JSON files on every experiment, and the proposer subagent re-reads the evidence.
  - Make `openinterp next` print the exact commands for the draw, so the driver doesn't need to re-derive them from `SKILL.md` and `docs/PROTOCOL.md`.
  - Add a compact `openinterp show <record|evidence>` summary, so agents never open `ev.json` or a full record.
  - Keep CLI output terse by default (`--quiet` for progress bars and warnings).
  - Measure whether a smaller driver model (`--driver`) completes experiments as reliably, since the CLI does the substantive work.
- [ ] Keep per-experiment checks proportional to the change. `openinterp check` and the wiki index rebuild walk every record, so their cost grows with the registry. Check only the records a branch adds or changes (`--base-ref origin/main`, as `reproduce` does), and keep the full check for CI on `main`.
- [ ] Run experiments in parallel on one machine: several loops in separate worktrees. The planner's claim weighting already keeps them off the same neuron. Share one model-weight cache and the per-layer scan cache, and cap concurrency by RAM/VRAM.

## Merge verification

At the moment CI's "Experiment records" job (`openinterp check`) only confirms that a PR adds a valid JSON record and its wiki page. Nothing in CI re-runs the experiment, so a record with edited activations or a hand-written verdict would pass. `openinterp reproduce --base-ref origin/main` now does that check (docs/PROTOCOL.md §5), and CI's "Reproduce records" job runs it on every pull request, on GitHub-hosted CPU runners with cached weights and a model allow-list. That is enough while volume is low. The items below are for when it isn't: larger models, many PRs, or runner minutes adding up.

- [ ] Queue and cost controls: a submission from CI only enqueues a job (record path, commit SHA, record hash). Deduplicate by record hash, cap jobs per PR and per author per day, set a per-job timeout and a monthly budget, and refuse jobs once the budget is spent rather than scaling up.
- [ ] Treat records as untrusted input: run only allow-listed models (no arbitrary Hugging Face repos, never `trust_remote_code`), bound probe count and length, and run each job in a sandbox with no network beyond the model cache.
- [ ] Report back to GitHub as a commit status or check run on the PR's head SHA ("Reproduction: passed / failed / queued"), with a link to the log, and make it a required check for `main` once it is reliable.
- [ ] Auto-merge experiment-only PRs: when every changed file is under `wiki/` (experiment records, their `.sig` files, pages and generated indexes) and "Experiment records" and "Reproduce records" pass (reproduced, or signed by an allowed signer), merge the PR automatically. Any change outside `wiki/` (code, workflows, `.github/allowed_signers`) keeps the PR manual. Make those checks required first, so GitHub's auto-merge waits for them. Fork PRs need a separate `pull_request_target` workflow that only reads the PR's file list through the API and enables auto-merge, and never checks out or runs the PR's code with a write token. Squash-merge so `main` gets one commit per experiment.
- [ ] Re-queue automatically when a PR's record changes, and let maintainers re-run or skip a job by hand (e.g. a label) for records whose model the worker can't load yet.

## Visualisations

Experiment pages currently show the activation histogram and token heatmaps of the top-activating examples; everything else is tables and token chips. Keep new charts as static SVG rendered at build time, readable in light and dark mode, with the numbers available as a table or in the Parquet export. Some need extra data stored on records; check that before building them.

- [ ] Steering samples: the same prompt continued with the neuron clamped high, at baseline and ablated, side by side.
- [ ] Layer map on model pages: a layer × neuron grid coloured by status (VERIFIED, rejected, dead, unexplored), so coverage and where the interpretable neurons sit are visible.
- [ ] Feasibility study: a concept map on model pages, in the style of "2.5 million cancer research papers, mapped" (embed each item's text → k-NN graph → UMAP → Leiden clusters, labelled by theme). Each point is a neuron, placed by the meaning of the text it fires on, not by its activations or weights. Write up the findings and a recommendation before building anything.
  - **The embedding.** Use a text embedding (sentence-embedding model or embeddings API) of the text each pole fires on: its top-activating contexts, with the peak token marked inside a window of about ±20 tokens. A neuron's vector is the mean embedding of its positive-pole texts minus the mean embedding of its negative-pole texts. Then L2-normalise it and compare with cosine. So a "Paris as a city" neuron lands near other city neurons. Two neurons whose poles swap the same concepts (+city/−code and +code/−city) point opposite ways. Text that both poles share, such as "this is English web text", cancels out. Things to settle:
    - *Weighting.* Use a plain mean, or weight each text by its activation or rank so the strongest examples count most. Means keep the two poles on the same scale even when they keep different numbers of examples.
    - *Near-zero vectors.* When both poles fire on similar text, the difference is mostly noise. Record the pre-normalisation norm as a confidence, and drop or fade neurons below a threshold.
    - *One meaningful pole.* When one pole is scanned but dead or diffuse, the vector is the live pole's concept minus generic text. Compare that with treating the weak pole as missing (zero).
    - *Two unrelated poles.* A neuron with +Paris and −Python code sits on the direction between the two. Measure how often this happens, using studied neurons whose explanations cover both poles, and decide whether those neurons need per-pole points as well.
  - **Missing poles count as zero.** A pole with no scanned texts contributes a zero vector, so the neuron's vector is the mean embedding of the pole it does have, negated if that pole is the negative one. This means the existing records can draw a map now, with no re-scan. There are about 100 of them against 116,736 neurons in Qwen2.5-0.5B (24 × 4,864), and most cover one pole only. Two caveats:
    - In a two-pole neuron, text the poles share cancels out. In a one-pole neuron it doesn't, so one-pole neurons all carry the same generic "web text" component and may clump together, or sit apart from two-pole neurons that have the same concept. Test subtracting the corpus-mean embedding from each pole mean, or measure how much the clumping matters.
    - A neuron known only by its negative pole points away from its concept. That's consistent with the two-pole rule (−city sits opposite +city), but the map should show which neurons are one-pole.

    Growing coverage beyond the records still needs top-k contexts for more neurons. That means one pass over the corpus with hooks on every layer, keeping a top-k heap of (snippet, position) per pole. One pole per neuron is now enough to place it, though both are better. The scan already reads whole layers (`CovarianceAccumulator` takes `layer_acts`), so the hooks exist, and the forward pass costs roughly one scan. Embedding is up to 233k × k short texts; cost and time it for a local model and for an API.
  - **Check it against explanations.** For studied neurons, embed `natural_language_explanation` too and compare it with the context-difference vector, e.g. how often the two are nearest neighbours. Also check that neurons with similar VERIFIED explanations land together, across layers.
  - **Drop neurons with nothing to say first.** Dead neurons (`neuron-family-triage`) are one case. Purely positional or frequency neurons are another: their "concept" is a position, and their contexts would scatter them as noise. Place those in their own labelled regions, or leave them out.
  - **Labelling clusters.** The cancer map names themes. Name clusters from the explanations of their studied members where there are some, otherwise from an LLM summary of the members' positive and negative contexts. That summary must read as unverified, so a cluster name never looks like a VERIFIED claim. Colour studied neurons by status and link them to neuron pages.
  - **Fit with the build.** `umap-learn` (numba, pynndescent, scikit-learn), `leidenalg`/`python-igraph` and any local embedding model are heavy. Put them in an optional dependency group, run the map offline (`openinterp map <model>`), and commit the coordinates, cluster IDs and norms (a few MB per model), so the site build stays light and deterministic. UMAP is only reproducible with `random_state` set, which makes it single-threaded; time that on 117k points. 117k SVG circles would be several MB, which breaks the static-SVG rule above. Render the density as a build-time PNG and draw the studied neurons as an SVG overlay.
  - **Questions to answer.** Does the map show coherent semantic clusters? Do the neurons for one concept at different layers (e.g. every "city" neuron) sit together? Is the map stable across seeds, choices of k and embedding models? Can the planner use it to draw from unexplored or unexplained clusters? Say plainly in the write-up that distances between clusters on a UMAP don't mean much.
- [ ] Registry dashboard on the home page: experiments and VERIFIED share over time, verdicts per model and per layer, and per-agent acceptance rates.
- [ ] Later: circuit diagrams once attention heads land, and a cross-model graph linking matched neurons for universality.

## Scope expansion

- [ ] A transcoder/SAE baseline arm to test the native-neuron premise on the same layers.
- [ ] Cross-reference existing feature dictionaries. For models with public SAEs or transcoders (Gemma Scope, Llama Scope, the circuit-tracer transcoders), list the latents whose decoder aligns with the pole's `w_out` or whose encoder aligns with its `w_in`, and pull their descriptions from the Neuronpedia API. Agreement is supporting evidence; disagreement is something for the reviewer to probe. Store the dictionary ID and latent index, not copies of the descriptions.
- [ ] Parameter-space decomposition as a second baseline: compare verified neurons with APD/SPD components (Braun et al., 2025; Bushnaq et al., 2025) on the small models where those methods run.
- [ ] Cross-model universality links: match verified neurons across sibling models (Qwen2.5-0.5B ↔ 1.5B).

## Datasets

Scan corpora beyond `NeelNanda/pile-10k`. `openinterp corpus` samples any Hugging Face Parquet dataset and records its ID, commit, seed and licence on every submission (`fineweb-edu` is the default preset). Adding a dataset below is mostly a new preset in `openinterp/corpus.py`. A dataset stored as JSON can use the Hub's Parquet conversion, as the `c4` preset does (`revision="refs/convert/parquet"`). Check each licence before redistributing excerpts in the wiki.

- [ ] `wikitext` (`wikitext-103-raw-v1`): long, clean articles for ≥ 512-token windows and entity/fact neurons.
- [ ] `roneneldan/TinyStories`: simple English for a low-confound control corpus and small models.
- [ ] Code: `bigcode/the-stack-smol` or `codeparrot/github-code-clean`, for syntax, bracket and indentation neurons.
- [ ] Multilingual: `wikimedia/wikipedia` (non-English configs) or `uonlp/CulturaX`, for language-identity and script neurons.
- [ ] Maths: `open-web-math/open-web-math` and `openai/gsm8k`, for number, operator and LaTeX neurons.
- [ ] Chat/instruction text: `HuggingFaceH4/ultrachat_200k` or `lmsys/lmsys-chat-1m` (gated), formatted with the model's chat template, for instruct models.
- [ ] Labelled concept sets for `sparse-probing-search` and positive holdouts: `conll2003` (NER), `universal_dependencies` (POS/syntax), `papluca/language-identification`, `ag_news` (topic).
- [ ] A mixed default corpus: a fixed, versioned blend of the above, so random-selection statistics aren't biased toward one domain.

## Models

The hooks already resolve Llama/Qwen2/Mistral/Gemma/Phi, GPT-2, GPT-J, GPT-NeoX and OPT layouts (`openinterp/harness/hooks.py`). Each new model needs a smoke test, a note on its MLP activation (gated SwiGLU/GeGLU vs plain GELU, which changes what a "negative pole" means) and a check that the final-norm path resolves. Then add it to `[tool.openinterp] models` in `pyproject.toml` and to `ALLOWED_MODELS` in `.github/workflows/ci.yml`, and `openinterp next` starts exploring it.

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
- [ ] OpenAI's weight-sparse transformers (Gao et al., 2025, `openai/circuit_sparsity`): trained so that circuits are small and neurons are close to monosemantic by construction. They should have a high VERIFIED rate, which makes them a positive-control model family and a reference point for how clean dense models' native neurons are. Check the architecture against the hook resolver first.
- [ ] Mixture-of-experts (Qwen1.5-MoE-A2.7B, OLMoE-1B-7B): needs a decision on how to identify an expert neuron (`expert` index in the identifier) and how to handle routing in patching. Defer until `ComponentIdentifier` lands.

## Housekeeping

- [ ] Mark the CI jobs "Lint and tests", "Experiment records" and "Reproduce records" as required in the GitHub branch protection rules for `main`.
- [ ] Add the R2 secrets and variables to GitHub (see README, "Deploy"), then run the deploy workflow once by hand.
- [ ] Keep a short "State of the registry" note at the top of this file (or a `make stats` target that prints it): experiments, units, models, VERIFIED share, self-reviewed share and protocol versions in use. As of 2026-10-20: 79 experiments on 67 units, all Qwen2.5-0.5B, 46 VERIFIED, and 8 records still under protocols 0.2–0.4. Several items above depend on these numbers, and nobody should have to rerun ad-hoc scripts to see them.
- [ ] Say in `CHANGELOG.md` that only records reviewed after the explanation-scores change have scores. "Every review records them" reads as if the whole registry has them, but the site shows them on 2 pages.

## Later stage

Research directions to pick up once the registry covers several models well.

- [ ] Investigate using the Platonic Representation Hypothesis (PRH) once multiple models are well understood.
- [ ] Investigate predicting an unknown neuron's interpretability based on previous audits.
- [ ] Investigate simplifying neural pathways.
- [ ] Investigate inserting memories or behaviours.
- [ ] Investigate removing memories or behaviours.
- [ ] Investigate complete explainability of each token.
- [ ] Investigate model concept merging or diffing. Start with base ↔ instruct pairs: which verified neurons keep their hypothesis after instruction tuning, and which change. Crosscoders (Lindsey et al., 2024) are the learned-basis comparison.
- [ ] Developmental tracking: follow VERIFIED neurons through Pythia and OLMo training checkpoints. When does the pole appear, does its hypothesis hold from then on, and does its birth line up with loss plateaus or with changes in the local learning coefficient (Lau et al., 2023; the `devinterp` library)?
- [ ] Training-data links: on OLMo (open data), count a verified neuron's trigger n-grams and contexts in the pretraining corpus (e.g. with infini-gram) to test whether concept frequency predicts which neurons become clean. Gradient-based attribution (TracIn, EK-FAC influence functions; Grosse et al., 2023) is only affordable on Pythia-70M-scale models.
- [ ] A registry coverage metric. Mean-ablate every neuron in a layer that isn't VERIFIED, or replace each verified pole with its hypothesis-simulated activation, and measure the share of loss recovered. This is a running measure of how much of the model the registry explains, and a concrete form of "complete explainability of each token".
- [ ] Standard benchmarks once attention heads land: MIB (Mueller et al., 2025) for circuit localisation, and RAVEL (Huang et al., 2024) for whether verified neurons separate an entity's attributes (a city's country from its continent).
- [ ] Factual recall and editing: use causal tracing (Meng et al., 2022) to check whether verified neurons sit on the fact-recall path at the subject's last token, before trying to insert or remove memories. Keep in mind the critiques of the knowledge-neuron view (Niu et al., 2024).
