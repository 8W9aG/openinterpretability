# gBench: performance and capability benchmarks for open models

**Citation.** Luciano Martins and Frank Li (Google DeepMind). "gBench: performance and capability benchmarks for open models". Software, [google-deepmind/gbench](https://github.com/google-deepmind/gbench), read at commit `d3e4283162d6c34353fa933284135d2a7ef3b9fe` (last push 2026-11-24, created 2026-10-30). Also on [PyPI](https://pypi.org/project/gbench/) as `gbench`. Apache-2.0, copyright 2026 Google LLC. No paper, venue or blog post: a search on 2026-11-26 found none, and the README names none. The suites' datasets keep their own licences.

**Status.** Planned. Its idea of a pinned, hashed scaffold beside each score fits this repo's explainer benchmark (#6985). Its capability suites don't fit the checks after a weight change, for the reasons below.

**Read.** On 2026-11-26, from a download of the repository at the commit above, by Claude (Opus 5.5). Read in full, in the source's own text: `README.md`, `docs/GOSPEL.md` and `docs/golden-set.md`. Read in part: `docs/performance.md` (its section on repeatability), `docs/scaffold-versioning.md` (its first 60 lines) and `docs/THIRD_PARTY_DATASETS.md` (its first 40 lines and its ARC lines). We searched `docs/eval-suites.md` and `docs/results.md` for decoding, ARC and provenance. Not read: the code, the tests, the other docs, the per-suite guides and the notebooks. No result or score was run.

## Summary

**Claim.** One harness should measure both how fast an open model serves and how well it answers under that serving setup. It does both through the same OpenAI-compatible `/v1` interface, so that results from different hardware and frameworks compare.

**Method.**

- *Serving performance.* Latency and throughput under fixed workload shapes, over several iterations. An open-loop stress sweep then certifies the highest request rate that meets limits on time to first token and inter-token latency.
- *Capability, in three tiers.* 86 academic and domain suites in `--evals all`, a 16-case deterministic "Golden Set", and a 60-scenario agentic suite from a pinned GemmaClaw commit.
- *Scaffold versioning.* Each run is a pair. `subject_id` names the model and its serving format. `scaffold_id` hashes eleven fields, such as the prompt, schema and scoring versions, the decode settings and the dataset snapshot.

**Findings, as the source states them.** The source reports no model scores. Its stated rules:

| Rule | Text |
| --- | --- |
| Repeatability | A coefficient of variation at or below `--max-cv-percent` (default 5.0) is valid. Above twice that, the run is "invalid (poor repeatability)". |
| The scaffold | "Pin the scaffold as tightly as the weights, otherwise your benchmark is measuring a moving target." |
| Comparing runs | A run whose scaffold moved while its subject held is "a new experimental condition". When both moved, the comparison is "uninterpretable". |
| Honest scores | "a suite that cannot produce a trustworthy score hard-errors rather than emitting a misleading 0" |
| Chat-template pairs | Four cases put an assistant turn, a tool turn or a multi-turn history on the wire. Each has a twin that asks for the same answer in one plain user turn. A failed case beside a passing twin points at the template. |
| Dataset licences | gBench doesn't redistribute benchmark data. A score is "subject to that dataset's own license and terms of use", and the inventory is a snapshot from 2026-09-14. |

## Where the claims go beyond the evidence

**The chat-template isolation is untested, and the authors say so.** The Golden Set doc states that "The isolation is a structural argument, not a measured one". It adds that nobody has run the pairs against a deliberately broken template on real weights.

**A tool, not a study.** The source shows no result for any model, so its claims are about what the harness records, not about models.

**Licence status can drift.** The dataset inventory is a dated snapshot, and the source asks users to re-check the tags before a release.

The authors state each of these limits themselves.

## Use in this repo

**Capability after a weight change: gBench doesn't fit.** `papers/sequential-tuning` scores zero-shot ARC-Easy by the highest mean log-probability per token, beside held-out losses. It runs small base models loaded in process. gBench scores served chat models through `/v1`, by their generated text, and sizes GPUs for models up to 20B and beyond. Its suites include ARC-AGI, not ARC-Easy. A log-probability check of a base model has no `/v1` equivalent in it, so it wouldn't make those checks broader here.

**Chat templates: the same control, at another level.** gBench's twins hold the content and change only its wrapping, then read the output. `papers/template-or-tuning` holds the content and changes the wrapping too, but it reads neuron activations, not answers. The two agree on the design. gBench's check can show that a template is broken. It can't show what the template does to a unit's explanation, which is the paper's question.

**The explainer benchmark: one idea to take.** `docs/benchmark.md` freezes each version, records its licence in `manifest.json`, and says that the judge model changes the scores. A summary records the version, the judge model, the split and the order. It doesn't record the judge's prompt or decode settings, which gBench pins in `scaffold_id`. #6985 adds that, so two summaries compare only under the same scaffold.

**Licences: already handled differently.** The explainer benchmark's data is this registry's own, under CC-BY-4.0, so it has no third-party dataset to inventory.
