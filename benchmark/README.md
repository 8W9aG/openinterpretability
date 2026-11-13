# Explainer benchmark data

Frozen versions of the public benchmark for tools that explain neurons. Each folder (`v1/`, …) is one version. A version never changes after its first publication.

`results/` holds published scores, one folder per version and explainer and judge, for example `results/v1/registry.claude-haiku-5-5/`. Each split has its per-unit rows (`dev.jsonl`) and its summary (`dev.summary.json`).

Read [docs/benchmark.md](../docs/benchmark.md) for what the benchmark measures, how its units were selected, the file formats and how to score an explainer. The data is under CC-BY-4.0, like the records it comes from.
