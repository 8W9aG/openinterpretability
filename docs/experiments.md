# Running experiments

The README's [Quickstart](../README.md#quickstart) is the shortest path to a first experiment. This page has the details. It covers which LLMs can propose and review, the scan corpus and caches, and running many experiments at once. It also covers the other unit types and how to contribute a result. How verdicts are decided is in [PROTOCOL.md](PROTOCOL.md).

## Choosing the LLMs

Any LLM can write the proposals and reviews. `--agent-model` takes either:

- a [LiteLLM](https://docs.litellm.ai/) model string, with that provider's API key set, e.g. `export ANTHROPIC_API_KEY=...` for `anthropic/claude-opus-5-5` (the default).
- `external/<name>` (e.g. `external/claude-opus-5-5`), which needs no keys. The command writes `exchange/<Agent>.prompt.md`, including the JSON schema of the answer, and waits for `exchange/<Agent>.response.json`. A coding agent, a chat window or a person can write the answer. Name the model that actually answered, because that name is recorded.
- answer files, which need no keys and no waiting process. `audit --evidence-out` and `review --emit-prompt` write the prompt and exit. `audit --evidence … --hypothesis …` and `review --probes …` take the answer as JSON. In Claude Code, the `neuron-proposer` and `neuron-reviewer` subagents (`.claude/agents/`) write those answers:

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

  To record what each subagent cost, run `uv run openinterp usage <agent ID> --answer <its answer file>` after it finishes, before the command that reads the answer. It reads the subagent's Claude Code transcript and writes `<answer>.usage.json` next to the answer. `audit`, `review` and `score` then record its tokens and list-price cost (`openinterp/core/pricing.py`) instead of only the prompt's length. `openinterp costs --wiki-dir wiki` reports cost per experiment by role and model.

  The reviewer subagent has no tool that runs the model, so it can't see probe activations before its probe set is final. The two subagents run on different models (Opus and Sonnet), so the review counts as independent. Pass each one's reported model ID to `--agent-model`.

The record notes whether the reviewer was a different model from the proposer (`independent_review`). A different model is better, but it isn't required.

The default agent model is `$OPENINTERP_AGENT_MODEL`, falling back to `anthropic/claude-opus-5-5`, and `review` is an alias for `verify`. If the agents answering prompt files sit behind a proxy that hides the real model (e.g. Claude Code routed through LiteLLM to Grok), set `OPENINTERP_MODEL_ALIASES` so records name the model that answered: `claude-haiku-5-5=grok-4-fast,claude-opus-5-5=grok-4`, or one bare name for every `external/` model.

## The scan corpus

`audit` scans `./corpus.jsonl` unless given `--corpus`, and stops if it isn't there. Almost every record in the registry scanned `corpus.jsonl`, the first 500 documents of [pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k), each cut to 3,000 characters (SHA-256 `ded48cfc…`, git-ignored). Rebuild it byte for byte, so your scans are comparable with the registry's:

```bash
uv run openinterp corpus --dataset pile-10k --first --ascii --revision 127bfedcd5047750df5ccf3a12979a47bfa0bafa --out corpus.jsonl
```

Its sidecar, `corpus.source.json`, is committed, and `openinterp build` uses it to show the source of records that scanned this corpus before the sidecar existed, marked as inferred. The dataset card gives the licence as `bigscience-bloom-rail-1.0`, a model licence, and each Pile component keeps its own terms, so the corpus itself isn't published. For another corpus, `openinterp corpus` samples documents from a Hugging Face dataset, reading only the Parquet row groups it needs. Without `--dataset` it samples 500 documents in about a minute from `mixed`, a versioned blend of presets, so statistics over random neurons aren't tied to one domain. The blend is 69 each from FineWeb-Edu, C4 and the uncopyrighted Pile, and 52 of GitHub code. It adds 52 Wikipedia articles in ten languages and 51 FineWeb-2 web pages in 22 languages besides English. It also adds 35 WikiText articles, 35 of OpenWebMath, 34 UltraChat conversations and 34 scientific papers from peS2o, half abstracts and half full texts (`BLENDS` in `openinterp/blends.py`). Qwen2.5's chat template writes the conversations, so they suit instruct models, and every model scans the same text. Each document records its preset as `source`, which the scan reports as a label set, as it does languages. Code documents also record their programming language, and a scan parses them into syntax labels (`openinterp/syntax.py`). From `mixed-v7`, a part without exact labels gets detected ones: its documents' language (fastText `lid.176`) and programming language (Magika). A blend is never changed once released, so `mixed` names its latest version (today `mixed-v7`) and the sidecar records which:

```bash
uv run openinterp corpus --out mixed.jsonl             # --dataset fineweb-edu, pile-10k, pile-uncopyrighted, c4, wikitext, tinystories, github-code, wikipedia-multilingual, fineweb-2, open-web-math, gsm8k, ultrachat or pes2o for one dataset (--chat-template MODEL writes ultrachat with that model's template), --seed N for another sample, --detect-lang to label each document's language, --detect-code to label each document's programming language
```

It also writes `<name>.source.json` (here `mixed.source.json`), which pins the dataset commit, files, seed and licence. For a blend it holds each preset's own source. `openinterp corpus --dataset <preset>`, with that preset's share of `--docs` and the same seed, rebuilds each one byte for byte. `openinterp audit --corpus mixed.jsonl` records the sidecar in the submission, and warns about a corpus without one, and experiment pages show the licence next to the excerpts. `audit` scans a seeded sample of up to 200,000 tokens of the corpus, the same documents every time, and records the sample in provenance. Set it with `--max-tokens` and `--sample-seed`. `--max-tokens 0` scans all of it. Any Hugging Face Parquet dataset works: `--dataset <org/name> --files '<glob>'`. For long documents, raise `--max-chars` and scan them as overlapping windows with `openinterp audit --window-tokens 512`, which are recorded in provenance.

**Scan labels.** A scan labels personal data, toxic text, hate speech, postal addresses and people's names in its corpus. These labels measure how much of a unit's activation they explain. The first scan downloads spaCy's name model (about 460 MB) and two text classifiers (about 500 MB each) once. It also takes about 7 minutes longer per 500 documents. Later scans of the corpus read the results from disk.

`openinterp concepts` writes human annotations for labelled text rather than a scan corpus. It writes them as the character spans `openinterp probe-compare` reads. The annotations are CoNLL-2003 entities, Universal Dependencies parts of speech and relations, 20-language identification and AG News topics. Run it without `--label` to list a dataset's labels.

## Caches

Builds keep what they read on disk by dataset commit, up to 5 GB, so a repeated build reads far less from the Hub. `openinterp corpus-cache` shows or clears it, and `OPENINTERP_CORPUS_CACHE=off` turns it off.

A scan's forward pass captures the whole layer, so `audit` caches every neuron's activations in that layer in `~/.cache/openinterp/scans`. `$OPENINTERP_SCAN_CACHE` moves it, and `off` disables it. A later audit of any neuron in the layer replays the cache instead of running the model, and gets the same numbers. It needs the same model, corpus options and dtype. The cache is capped at 40 GB (`$OPENINTERP_SCAN_CACHE_GB`). A 200,000-token Qwen2.5-0.5B layer in float32 takes about 4 GB, and the least recently used layers go first. `openinterp scan-cache` lists or clears it. A scan on the ANE has its own entries, because its activations are fp16. For a neuron's per-token activations over the scan, e.g. every occurrence of one token, use `openinterp activations --model … --layer … --neuron … --corpus corpus.jsonl` (`--token`, `--min-strength`, `--top 0` for all, `--format jsonl`) rather than writing a scan loop.

## Devices

`--device auto`, the default, uses a CUDA GPU if there is one. On a Mac it uses Apple's GPU (MPS) only when the GPU is free, and else the CPU (`openinterp/harness/devices.py`). Two things make the GPU busy:

- **Another `openinterp` job.** A process that runs a model on MPS keeps a lease, an empty file named for its process id in `~/.cache/openinterp/devices/mps/`. `$OPENINTERP_DEVICE_LEASES` moves the folder, and `off` turns leases off. A lease of a process that has stopped doesn't count.
- **Other programs.** If no other job has a lease, the planner reads the GPU's utilization from `ioreg`. It takes the mean of 5 readings over about 2 s, because one reading can change from 0% to 50% in a second. Above 50% the GPU is busy (`$OPENINTERP_GPU_BUSY_PERCENT`). That threshold is provisional, until it is measured.

When the GPU is busy, the order is MPS, then Apple's Neural Engine (ANE), then the CPU. The job says which it chose, and why:

- **The ANE, if it is free.** It needs an Apple-silicon Mac (where coremltools installs), and a GPT-2 or Qwen2 model loaded whole. Then the scan's forward pass runs on the ANE through Core ML (`openinterp/harness/ane/`). Everything else runs on the CPU. The ANE computes in fp16, so the scan measures each pole's strongest candidates again on the CPU. That is the top 60, and any within 6% of the 20th, at most 120 (`openinterp/harness/scan_backend.py`). So the top examples and the scale that `openinterp reproduce` checks are the CPU's numbers. The histogram, partners and confounds keep the ANE's values, which correlate with fp32 at r ≥ 0.9998. A job on the ANE keeps a lease in `~/.cache/openinterp/devices/ane/`, and the record's `environment.scan_backend` is `coreml-ane`.
- **Else the CPU,** for everything.

The CPU is 4–10× slower than a free GPU, and the ANE is 1.5–1.7× slower (`docs/analyses/ane-feasibility.md`). But two jobs on one GPU slow down each other, and once a training run that shared the GPU with `make test` gave wrong results. Take an audit of GPT-2 L5 with 128-token windows over `corpus.jsonl`. It took 77 s on the CPU and 30 s on a free GPU. With the scan on the ANE and the rest on the CPU, it took 55 s. The top examples were the same each time.

`--device mps` uses the GPU even when it is busy. `--device ane` runs the scans on the ANE, and the rest on the GPU if it is free, else the CPU. A process plans once, and each model it loads uses the same device. The record's `environment.device` says which one ran.

Before a scan runs on the ANE, its model is converted to Core ML and compiled, once per layer and input shape. That takes 5–50 s, and the ANE's own compile when it first loads the model takes 10–35 s more. So converted models stay in `~/.cache/openinterp/coreml` (`$OPENINTERP_COREML_CACHE`, `off` to disable), capped at 20 GB (`$OPENINTERP_COREML_CACHE_GB`), least recently used first. An entry is keyed by the model, its revision, the layer and the shape. It is also keyed by a hash of the conversion code, the coremltools and torch versions, and the macOS version and chip. So a model update, a change to the conversion code or a macOS update builds new entries. A new entry is loaded once at its final path, because macOS keeps the ANE's compile by path. Then it is checked against fp32 on a test batch, and an entry more than 3% off is removed and not used. A GPT-2 layer is about 150 MB.

If the GPU fails during a command, for example because it is out of memory or has no kernel for an operation, the command runs again from the start on the CPU. It says why, and it runs in a new process, so none of the GPU's state stays. A command runs again only once (`$OPENINTERP_DEVICE_FORCE` marks the retry), and files that the first run wrote are written again.

## Memory

The weights set the floor, and what a scan or a review adds on top of them is small. These are peak memory footprints, measured with macOS `/usr/bin/time -l` on CPU in float32. The scan is the default 200,000-token `audit` scan, and the review is `openinterp reproduce` of one record.

| Model | Weights (float32) | Scan | Review | Machine |
| --- | --- | --- | --- | --- |
| GPT-2 small | 0.5 GB | 3.1 GB | 1.3 GB | 8 GB |
| Qwen2.5-0.5B | 2.0 GB | 5.4 GB | 3.0 GB | 8 GB |
| Qwen2.5-1.5B | 6.2 GB | 9.2 GB | 7.4 GB | 16 GB |

The joint ablation's KL in blocks of 64 positions lowered the reviews. On one day, a Qwen2.5-1.5B review peaked at 7.9 GB before that change and 7.4 GB after. An earlier measurement on a quieter machine was 6.9 GB. The last column is the smallest common machine with room for the operating system as well. Resident memory (RSS) reads 1–4 GB higher, because it also counts the weight files that the loader maps from disk. The system can drop those pages when it needs the memory.

Until November 2026, a Qwen2.5-0.5B review took 19.4 GB and its scan 7.0 GB. Most of it was logits: an array of batch × sequence × vocabulary, 1.2 GB for 2,048 tokens at Qwen2.5's 151,936 tokens. Strength reads, the scan and the lenses computed logits and dropped them, and one check padded every probe to the longest. Tests in `tests/test_agents.py` now fail if a scan or the lens computes logits, or if a review computes them over padding.

`scripts/memory_benchmark.py` measures the peak resident memory of a scan and a review of GPT-2 and Qwen2.5-0.5B. It compares them with `scripts/memory_baseline.json` and fails if one is more than 25% higher. The `release` skill runs it. Every record now holds its memory. Each stage in `provenance.costs.stages` has `peak_rss_mb`, the process's peak resident memory when the stage ended. On CUDA, `peak_cuda_mb` holds the stage's own peak. Experiment pages show the largest peak.

To use less:

- **bfloat16 on Apple's GPU.** `--device mps --dtype bfloat16` halves the weights, and on MPS it is about as fast as float32. On a 20,000-token Qwen2.5-0.5B scan it took 85 s and 5.2 GB, against 74 s and 6.7 GB in float32. On CPU, bfloat16 is far slower: the same scan had not finished after 47 minutes, against 3.3 in float32. A record pins its dtype, so a bfloat16 record reproduces only in bfloat16, and its numbers differ a little from float32's.
- **CPU on a small Mac, for a tight fit.** On Apple silicon the GPU shares the machine's memory, and MPS needs more of it than the CPU does. A 20,000-token Qwen2.5-0.5B audit peaked at 4.9 GB on MPS in float32 and 3.4 GB on CPU. For Qwen2.5-1.5B it was 11.3–13.5 GB on MPS, varying between identical runs, against 8.2 GB on CPU. Each stage gives back PyTorch's MPS cache as it ends (`costs.add_stage`). Before that, the 1.5B audit took 15.4 GB on MPS. The rest is how the MPS allocator places tensors, not memory that openinterp holds. It carves medium tensors out of large shared heaps and frees a heap only when every tensor in it is gone. So the 1.84 GB of Qwen2.5-0.5B's weights took 2.63 GB of Metal memory after loading. Emptying the cache after every pass didn't lower the peak. `PYTORCH_MPS_LOW_WATERMARK_RATIO` lowered it by 0.1–0.6 GB in some runs and not in others, and made some 1.5B runs slower. Packing the weights into one buffer per dtype removed the 0.8 GB at load. But each pass then copied its views, and the audit peaked at 7.4 GB. So none of them is used.
- **One session at a time.** Each session holds its own copy of the model.

The scan cache doesn't add to this. A scan writes each batch to the cache file as it goes, and a replay maps the file from disk. CUDA is not measured yet, so Colab's figures may differ. Quantised (8- or 4-bit) loading isn't supported, because it changes the activations, and records don't pin it.

### Models larger than memory

`--memory stream` runs a model that doesn't fit in RAM, slowly. The decoder layers stay in the checkpoint's safetensors files on disk. Each forward pass reads a layer's weights when it reaches it, and frees them after (`openinterp/harness/memory.py`). The embeddings, the final norm and the output head stay in memory. So the peak is about one layer, those modules and one batch's activations. The layers are read in place, so streaming needs no more disk than the download. A mixture-of-experts checkpoint that stores each expert apart is the exception (below).

It is not the default. With `--memory auto`, the default, a command streams only when it has to. That is when the weights at the chosen dtype, plus 3 GB, are more than the free memory: the GPU's with `--device cuda`, else RAM and swap. It then prints why. `--memory normal` never streams, and `$OPENINTERP_MEMORY_GB` plans for that much memory in place of what is free.

With `--device cuda` a streamed model runs on the GPU. The embeddings, the final norm and the head stay there, and each layer is copied there when a pass reaches it. Each tensor is sent in the checkpoint's dtype and converted on the GPU. On an RTX 4070 Ti, a float32 pass of Qwen2.5-0.5B took 0.66 s that way, and 2.8 s converted on the CPU first. A 5,000-token Qwen2.5-0.5B audit took 44 s streamed onto the GPU and 27 s loaded whole on it. Qwen2.5-3B in float32 needs about 14.5 GB, and the GPU had 9.1 GB free. With `--memory auto` its 5,000-token audit streamed onto the GPU and took 167 s. Its unit's activations matched the model loaded whole on the CPU within 2.4e-6 of their peak, and its logits within 3.3e-5. The same audit on the CPU didn't finish in 30 minutes. Other runs shared that GPU and CPU, so these times are rough. `tests/test_cuda_streaming.py` checks that a model streamed onto a GPU measures what one loaded whole there does. It runs only where CUDA is, which CI's runners are not. With `--device mps` a streamed model runs on the CPU, since `from_pretrained` with a device map on MPS ends in a segmentation fault.

The numbers don't change, so a streamed record reproduces in normal mode. A streamed 20,000-token Qwen2.5-0.5B scan matched a normal one in every field but two, which differ by 0.00001. Tests in `tests/test_memory.py` check this, and that each direct read of a weight gets the real one.

Every pass reads its layers again, and that is the cost. On a machine whose file cache holds the checkpoint it is small. `accelerate` opened the checkpoint file again for each tensor, which was 5.6 s of a 5.9 s pass over 16 tokens on Qwen2.5-0.5B. So the streamed model keeps each file open, and that pass now takes 0.2 s, against 0.14 s whole. With `OPENINTERP_MEMORY_GB=1`, a streamed 5,000-token Qwen2.5-0.5B audit took 144 s and peaked at 1.5 GB. A normal one took 125 s and peaked at 3.1 GB. Before the files stayed open, the streamed audit took 11 minutes. Its float32 weights are 2.0 GB. A streamed review of one record peaked at 2.0 GB. A streamed 5,000-token Qwen2.5-1.5B audit, whose float32 weights are 6.2 GB, peaked at 2.4 GB. It took 12–17 minutes beside other runs. Before the files stayed open, it didn't finish in 50 minutes. Its embedding and output head are one 0.9 GB matrix that stays in memory. So that is the least a streamed 1.5B run can take.

With less memory than the checkpoint, each pass reads from the disk itself. A streamed scan uses batches as large as half the planned memory allows, so it reads each layer a few times. A streamed model therefore runs more rows per pass where rows are independent (`memory.stream_rows`). That covers patching, the joint ablation's clean and ablated runs, integrated gradients and the lens. As many rows as a quarter of the planned memory holds go into each pass, so only float rounding changes. Two passes that read one layer now stop there, and patching reads each target once. The review also runs the items of its loops at once, layer-major (`harness/layer_major.py`). That covers probes, counterfactual pairs, the causal pairs, the joint ablation's prompts and `reproduce`'s top examples. Each item runs in its own thread with the model's own forward, so each pass computes what it would alone. A scheduler loads one layer at a time for every pass that waits at it. The streamed Qwen2.5-0.5B review now reads 31 GB from the checkpoint, against 147 GB, and the audit 24 GB, against 47.5 GB. Those figures had the checkpoint in the file cache. To measure without it, cap the command's memory below the checkpoint's size, so the file cache can't hold it. On Linux, `systemd-run --user --scope -p MemoryMax=3G openinterp audit …` does that without root. A streamed 2,000-token Qwen2.5-3B audit on an RTX 4070 Ti, whose checkpoint is 5.8 GB, then read 18.5 GB from the disk. It took 172 s, against 135 s with the checkpoint cached, and its evidence was the same. That disk read 0.6 GB/s, with other runs sharing it. Asking for the next layer while one runs (`posix_fadvise`) took 179 s, so it isn't used. On a GPU a layer computes in far less time than it takes to read, so there is little for the read to overlap. Under the 1 GB budget the review's peak rose from 1.7 GB to 2.0 GB. For a model of about 120B a pass reads about 240 GB, and then the number of passes sets the time (TODO, Performance). Gradients through streamed layers would keep every layer's weights for the backward pass. So attribution recomputes the layers below the unit in that pass, and reads them from disk again. The weights are also frozen at load, since the harness takes gradients only for inputs and activations. A weight that required a gradient made that audit peak at 5.2 GB, not 2.4 GB. A large model also needs bfloat16, because float32 doubles its size, and bfloat16 on CPU is about 10 times slower than float32 (above).

A record says that its run streamed (`environment.streamed`), and it reproduces in either mode. A mixture-of-experts model runs its experts with openinterp's own forward (`harness/moe.py`, `docs/moe.md`). Streamed, that forward reads only the slices of the experts that a pass's tokens are routed to. Granite 3.0 1B-A400M has 32 experts per layer and 8 per token. A streamed one-token pass read 1.41 GB, against 4.78 GB with each layer's experts read whole. A 13-token pass read 4.15 GB, since its tokens reach most experts. In Mistral Large 4 the same choice is between about 113 MB and 14.5 GB of experts in memory at once. Granite stores its experts stacked, so they are read in place. A checkpoint that stores each expert apart (Mixtral, OLMoE) is stacked by transformers as it loads. Streaming then writes the stacks at the chosen dtype to the offload folder (`$XDG_CACHE_HOME/openinterp/offload`), and reads its slices from there.

## Letting an agent run it (Claude Code)

The repo's skills load automatically, and `choose-next-experiment` defines the workflow. The agent:

1. picks an experiment with `openinterp next`, which is randomised so contributors don't collide
2. claims it with a draft PR
3. reads the neuron's history in `wiki/`
4. runs the audit and the independent review
5. posts the result.

The other skills (`native-neuron-audit`, `vocab-projection-lookahead`, `interpretability-illusions-review`, …) supply the methodology. Review the PR before marking it ready. To use API-key models instead of the subagents, name them in the request, e.g. "Use openai/gpt-5 as the reviewer model", with that provider's key set in the same shell.

**In a loop.** `scripts/experiment-loop.sh` runs experiments back to back, each in a fresh headless Claude Code session (`claude -p`). It stops when it reaches `--max` or `claude` exits with an error (for example at the usage limit). Claude drives the workflow, and you choose the LLMs that propose and review:

```bash
scripts/experiment-loop.sh --proposer anthropic/claude-opus-5-5 --reviewer openai/gpt-5 --max 10
make loop LOOP_ARGS="--reviewer openai/gpt-5"   # the same, via make
```

It works in a separate worktree (`../<repo>-loop`) reset to `origin/main` before each experiment, so your checkout is untouched, and writes one log per experiment to `logs/loop/`. Each session follows `land-experiment`. Each new scan builds a fresh `mixed` corpus from its draw's seed (pass `--corpus` to use one corpus instead). A review of an existing record rebuilds that record's corpus with `openinterp corpus --same-as`. It posts the PR, waits for CI to go green and merges it if the PR only touches `wiki/` and your `gh` account has write access. Otherwise it leaves the PR open for review and moves on. `--driver` picks the Claude model that drives the session. The default is Haiku, which landed every experiment in two pilots at about a tenth of Sonnet's cost (`docs/analyses/driver-model-pilot.md`). `--epsilon` sets how often `openinterp next` explores new neurons, and a proposer or reviewer of `external/<name>` makes the driving session answer that role itself. The session can edit files and run `uv run`, `make`, `git`, `gh pr`, `gh run` and `gh repo view` without asking. See `ALLOWED_TOOLS` in the script. After each experiment it prints what that experiment cost and a running total. The total covers wall time per `openinterp` command and stage, LLM tokens and cost, and the driving session's turns, tokens and cost. Every `openinterp` command appends its measurements to the ledger in `$OPENINTERP_COSTS_LOG` when that is set, and `openinterp costs <ledger> [--session <claude json>]` summarises one. Records store their own experiment's costs in `provenance.costs`. Run `scripts/experiment-loop.sh --help` for every option.

**Several sessions at once.** Give every session that runs experiments its own working directory, whether it's a loop or an interactive one. Git keeps one current branch per directory, so sessions that share a checkout switch branches under each other. One session's commits end up on another's claim branch and PR. A `git worktree` per session is enough. Worktrees share the git objects and the Hugging Face model cache, but `corpus.jsonl` is untracked, so copy it in (its sidecar is tracked):

```bash
git worktree add --detach ../openinterpretability-2 origin/main
cp corpus.jsonl ../openinterpretability-2/
cd ../openinterpretability-2 && uv sync && claude
```

Two `experiment-loop.sh` runs need different `--worktree` directories, since both default to `../<repo>-loop`. A branch can be checked out in only one worktree, which is why the workflow starts each experiment from a detached `origin/main`. Each session holds its own copy of the model, up to 9.2 GB for a Qwen2.5-1.5B scan (see *Memory*). So memory, not checkouts, limits how many sessions run in parallel.

## Running it yourself

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

For follow-ups that `next` suggests (revise, re-review, opposite pole), run the commands it prints. Add `--corpus corpus.jsonl` to any `audit`.

**Attention heads** go through the same commands: `--layer 5 --head 1` audits one head (`--unit head --random` samples one), and every later step reads the head from the submission. A head's strength at a token is the distance between its write to the residual stream and its mean write over the scan, `|W_O (z − z̄)|` ([PROTOCOL.md](PROTOCOL.md) §1).

**Dictionary latents.** Latents of published sparse dictionaries are the learned-basis baseline: `openinterp audit --dictionary gpt2-small-transcoders --random --window-tokens 128` audits a random latent of Dunefsky et al.'s GPT-2 small transcoders under the same protocol. The dictionary names its model, and its weights are pinned in the submission. These transcoders fit only the first 128 positions of a context. So their scans, and the GPT-2 neuron scans they are compared with, use 128-token windows. Feature verdicts are reported apart from native units, so the two shares VERIFIED can be compared. The home page compares them per dictionary, over randomly selected units only (`openinterp/builder/learned_baseline.py`). Presets are in `openinterp/harness/dictionaries.py`. Besides the GPT-2 transcoders, there are Gemma Scope's transcoders and MLP-output SAEs for Gemma 2 2B (`gemma-scope-2b-transcoders`, `gemma-scope-2b-mlp-saes`). There are circuit-tracer's transcoders for Llama 3.2 1B and Qwen3 0.6B (`llama-3.2-1b-transcoders`, `qwen3-0.6b-transcoders`). There are also Gemma Scope's residual-stream SAEs (`gemma-scope-2b-res-saes`), and Llama Scope's residual-stream SAEs, MLP-output SAEs and transcoders for Llama-3.1-8B (`llama-scope-8b-res-saes`, `llama-scope-8b-mlp-saes`, `llama-scope-8b-transcoders`). Scan them with `--dtype bfloat16`, since 8B weights in float32 take 32 GB. No public dictionary covers Qwen2.5-0.5B, so `qwen2.5-0.5b-transcoders` was trained here with `scripts/train_transcoders.py`. Its card is in `hf_models/`. [dictionaries.md](dictionaries.md) explains what each preset adds. A residual-stream latent reads and writes the residual stream after its layer. So comparing it with that layer's MLP neurons changes the site as well as the basis. They fit through position 1024, so their scans need `--window-tokens 1024` or less. Each preset's input site was checked by how much of the MLP output it reconstructs. The comment above each preset gives the numbers.

## Contributing a result

The planner reads open PRs to avoid duplicate work, so claim the neuron early. The branch and PR title use the neuron's slug with `/` replaced by `-`, e.g. `Qwen--Qwen2.5-0.5B-L11-N4255-positive`:

```bash
git switch -c experiment/Qwen--Qwen2.5-0.5B-L11-N4255-positive
gh pr create --draft --title "experiment: Qwen--Qwen2.5-0.5B-L11-N4255-positive (explore)"
# …run the experiment, then:
make check                                # lint and tests too (make check lint test) if you change code
git add wiki && git commit -m "Experiment: L11/N4255+ (NEEDS_REVISION)" && git push
gh pr ready
```

To be credited, set `OPENINTERP_CONTRIBUTOR` to a public handle such as your GitHub username (never an email), or pass `--contributor` to `audit` and `verify`. It is recorded beside the agent model, as the proposer's contributor on the submission and the reviewer's on the verdict. The site's Agents page lists every contributor's experiments, VERIFIED results, well-probed rejections and reviews. Nothing is recorded without it: git config is never read. The loop script's sessions inherit the variable.

Commit `wiki/`. Each experiment adds its own record (`<id>.json`) and page (`<id>.md`) and never edits another's. So any number of experiments on one neuron can be in flight at once. Never edit generated pages except their **Discussion** section. The indexes (`wiki/README.md` and each neuron's `README.md`) are rebuilt locally by `openinterp wiki` and are gitignored, so experiment PRs never conflict over them. CI rejects a PR whose record and page don't match, and once merged the deploy publishes the result within a minute or so. What CI checks is in [development.md](development.md#pull-requests).

**The experiment wiki.** Every completed experiment gets a Markdown page and an archived copy of its full record (`<id>.json`) in [`wiki/`](../wiki/), whatever its verdict. `openinterp verify` and `openinterp reevaluate` write it automatically, and `openinterp wiki` backfills pages for existing records. Each page records the hypothesis, top examples, gate results, probes, counterfactuals and next steps based on the gates that failed. A page is written once and never regenerated, so its Discussion section is safe to edit. Each neuron also gets a history page (`README.md` in its folder), and `wiki/README.md` indexes every experiment. Both are generated by `openinterp wiki` and not committed, so run it after pulling to browse them locally. The site shows the same history.

## Claude Code skills

`.claude/skills/` holds research skills that Claude Code loads automatically in this repo. Each one condenses a set of papers into a procedure that uses the `openinterp` harness:

| Skill | Use it to | Based on |
| --- | --- | --- |
| `choose-next-experiment` | Pick the next experiment with `openinterp next`: randomised, explore vs exploit, herd-resistant | ε-greedy and Boltzmann sampling over follow-ups |
| `write-paper` | Write a paper in `papers/` with checked references, and figures and animations tied to its results | |
| `land-experiment` | Run experiments end to end and merge them: `choose-next-experiment`, then build, wait for green CI and merge (with write access), else move on | — |
| `native-neuron-audit` | Avoid misreading a neuron (context, data, poles, SwiGLU regimes) | Goodfire, *Already Clean* (2026) |
| `vocab-projection-lookahead` | Decide retrospective vs lookahead | Gurnee et al. 2026 (Jacobian lens), Geva et al. 2021/2022, logit lens, Voita et al. 2023 |
| `neuron-family-triage` | Pre-classify neurons from weight statistics | Gurnee et al. 2024, Stolfo et al. 2024, Voita et al. 2023 |
| `explanation-simulation-scoring` | Score explanations by predicted activations | Bills et al. 2023, Huang et al. 2023 |
| `activation-patching-practice` | Choose causal metrics and corruptions, and handle self-repair | Meng et al. 2022, Zhang & Nanda 2023, Heimersheim & Nanda 2024, McGrath et al. 2023, Rushing & Nanda 2024, AtP* |
| `neuron-steering-tests` | Test a hypothesis by clamping the neuron and generating | Turner et al. 2023, Goodfire 2026 |
| `sparse-probing-search` | Find neurons for a labelled concept | Gurnee et al. 2023, Bolukbasi et al. 2021 |
| `interpretability-illusions-review` | Write probes as the reviewer and spot illusions | Bolukbasi et al. 2021, Huang et al. 2023, Makelov et al. 2023, Sun et al. 2024 |
| `attention-head-audit` | Extend the registry to attention heads and transcoder baselines | Olsson et al. 2022, Wang et al. 2022, Conmy et al. 2023, Dunefsky et al. 2024, Paulo et al. 2025 |
| `single-source-of-truth` | Define each constant, threshold or list once, and pin copies that must exist with a test | — |
| `commenting` | Write comments and docstrings that say why, in ASD-STE100 Simplified Technical English | ASD-STE100 |

## Definitions

- **Neuron activation:** the input to the MLP's down/output projection. For gated MLPs this is `act(gate(x)) * up(x)`.
- **Pole:** the positive pole is `clamp(a, min=0)` and the negative pole is `clamp(a, max=0)`. Pole *strength* is always reported as a non-negative magnitude.
- **Attention head:** one head's slice of the attention output projection's input. Its strength is `|W_O (z − z̄)|`, with z̄ its mean output over the scan.
- **Counterfactual Δ:** `(peak strength on base − peak strength on counterfactual) / max activation`.

## Models

The models `openinterp next` explores (`[tool.openinterp] models` in `pyproject.toml`). What a pole means depends on the MLP's activation. Each model's terms are in `[tool.openinterp.licenses]`, and its site page shows them.

| Model | MLP | Negative pole | Scan |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | SwiGLU, `silu(gate(x)) * up(x)` | Unbounded: `up(x)` carries the sign | Whole documents |
| `Qwen/Qwen2.5-1.5B` | SwiGLU, as Qwen2.5-0.5B: 28 layers × 8,960 neurons | Unbounded: `up(x)` carries the sign (about half of all values are negative in every layer) | Whole documents |
| `openai-community/gpt2` | GELU, `gelu_new(c_fc(x))` | GELU's dip, at most 0.170 in every layer and reached at a pre-activation of about −0.75. 73–88% of tokens are below zero, and a strongly negative input reads as inactive | 128-token windows (`[tool.openinterp.window_tokens]`), matching its transcoders. GPT-2 itself reads 1,024 positions |

Qwen2.5-1.5B's final norm is `model.norm` (RMSNorm) and its unembedding is tied to the input embedding, as in 0.5B. GPT-2's final norm is `transformer.ln_f` and its MLP norm `ln_2`, which the transcoders' `mlp_in_normalized` site standardises.
