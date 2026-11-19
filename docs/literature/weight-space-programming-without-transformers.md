# Weight-Space Programming Doesn't Need Transformers

**Citation.** Hyunjong Ok and Jaeho Lee, Pohang University of Science and Technology (POSTECH). "Weight-Space Programming Doesn't Need Transformers". arXiv 2611.22055, v1 17 November 2026, the only version. The arXiv comment says "Preprint (Work in progress)". Subjects are cs.LG, cs.AI and cs.CL. [Paper](https://arxiv.org/abs/2611.22055), [code](https://github.com/effl-lab/Weight-Space-Programming), [project page](https://wsp-no-transformer.github.io). Preprint, no venue given. The code repository exists and holds a harness, baselines, prompts and models. We did not run it. The paper gives no funding statement. The three LLMs that write the weights are Claude Opus 5.5, GPT-6.1 Sol and GPT-6 Astra. One image test also uses Claude Fable 5.2.

**Status.** Planned. #4782 would use it in its strand on weight-space programming and fast weights, and it is evidence for #5144 and #3446. No code, skill or protocol in this repo depends on it, and this page proposes no new issue.

**Read.** On 2026-11-19 we downloaded the v1 PDF from `https://arxiv.org/pdf/2611.22055` and changed it to text with `pdftotext`. We also read the abstract page, to check that the ID gives this title and these authors. A model read all 33 pages of that text: Sections 1–9, the references and Appendices A–H, with Tables 1–26 and the figure captions. We did not see the figures as images, so numbers that only a figure gives are not here. One conflict of interest: the model that wrote this page is Claude Opus 5.5, one of the paper's three writers.

## Summary

**Claim.** When an LLM writes a model's weights by hand, with no gradients, a Transformer is a poor structure to write into. Free to design the model, three frontier LLMs each build a sparse fact table that a fixed procedure reads. These tables score higher on ARC than Transformers that the same LLMs wrote, with far fewer parameters. The authors then design a hashed associative memory that keeps the written table and runs faster than a Transformer of its size.

**Method.**

- **Weight-space programming (WSP).** An LLM writes the weights of a separate target model. The rules forbid gradient training and statistical fitting on data. They allow formulas that turn hand-written text into weights.
- **Writers and sandbox.** Claude Opus 5.5 (high reasoning effort), GPT-6.1 Sol and GPT-6 Astra work in a sandbox with no network. Each run sees ten labelled ARC questions, five from ARC-Easy and five from ARC-Challenge. It sees no test accuracy.
- **Two regimes.**
  - **Fixed architecture.** The writer edits a random initialisation of 15 released architectures (Transformer decoders and encoders, Mamba-2, RWKV-7). Or it writes every weight of a given two-layer Transformer with mean pooling.
  - **LLM-designed.** The writer delivers an architecture file, a build script and plain-text knowledge files. A static check forbids Python control flow in the forward pass.
- **Lineages.** Each generation has two runs. The run with the higher accuracy on 862 validation questions is the parent of the next generation. A lineage stops when validation accuracy stops rising, or after generation 10.
- **The authors' model.** A fixed forward pass maps words to concepts and takes the 16 strongest concepts of the question. It reads the 120 concept pairs at hashed rows of a pair table, and adds one product with a concept table A. The LLM writes the tables.
- **Test.** The 3,530 four-option questions of the ARC-Easy and ARC-Challenge test sets. Other tests are SciQ, OpenBookQA, MMLU, MathBench, SQuAD 1.1, CNN/DailyMail, AG News and CIFAR-10.
- **Baselines.** Gradient descent from random initialisation on the 3,358 ARC training questions, with one seed. Fine-tuned pretrained models. BM25 search over the text that each writer wrote.
- **Controls.**
  - **Line removal.** Remove one written line, rebuild, and count changed test answers, for 40 random lines per model.
  - **One best fact.** Keep only the single best-matching fact per option.
  - **Test-set overlap.** Four n-gram and content-word rules, a deletion test and a masked-option recall test.
  - **Restricted runs.** Runs that forbid practice questions, and runs that forbid fact rows.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Claude Opus 5.5, LLM-designed, generation 1 (0.94M parameters) | ARC test 50.8% |
| Claude Opus 5.5, LLM-designed, generation 5 (3.35M parameters) | 54.4%, the best lineage |
| Claude Opus 5.5, authors' model, generations 1 and 5 | 49.6% and 52.4% |
| Claude Opus 5.5, given two-layer Transformer (3 widths) | 36.1% |
| Claude Opus 5.5, 15 released architectures from random initialisation | 43.1% mean |
| GPT-6.1 Sol and GPT-6 Astra, best LLM-designed generation | 47.4% and 46.2% |
| Gradient descent, two-layer Transformer of an LLM-designed size (3.33M) | 29.8% |
| Fine-tuned Qwen3.5-0.8B | 82.1% |
| Size ratio in the abstract | Qwen3-1.7B written from random initialisation (1.72B, 48.7%) against the 0.94M LLM-designed model gives about 1,800× |
| One best fact per option only, Claude Opus 5.5 generation 1 | 49.1% against 50.8% for the full model |
| One removed line, LLM-designed and authors' models of Claude Opus 5.5 | 0.5 to 1.8 changed answers on average, of 3,530 |
| One removed line, written Transformers of Claude Opus 5.5 | 4.8 answers, and 1,350.7 when one random stream draws every vector |
| One removed line, written Transformers of GPT-6.1 Sol and GPT-6 Astra | 10.8 and 20.3 answers |
| Six written two-layer Transformers of Claude Opus 5.5 against BM25 over their own text | below BM25 by 8.0 points on average |
| Written two-layer Transformers, all three writers | below BM25 in 20 of 23 runs |
| Table models, all three writers | above BM25 in 21 of 24 runs |
| SciQ, Claude Opus 5.5's own measure of its written Transformer | F1 0.37 without interference, 0.24 to 0.25 inside the Transformer |
| GPU time for 3,530 questions, authors' model, generations 1 and 5 | 18 ms, 7× and 17× faster than a same-sized Transformer at its fastest input |
| 46 LLM-designed variants on the GPU | one beats a same-sized Transformer, and the median is 9.4× slower |
| Pair table folded from 1,048,576 to 65,536 rows | 49.6% → 49.8%, and 4,096 rows gives 44.0% |
| WSP on pretrained checkpoints, 87 runs | 48 change under 1% of weights, and 43 of those stay within one point |
| Same, overwrites | 30 overwrite the checkpoint, and each of the 11 on encoders above chance lowers accuracy |
| Same, strong checkpoints | none at 72.3% or above is overwritten, across 28 runs, each changing at most 0.07% of parameters |
| UltraBERT-380M edited by two writers | 63.7% → 33.1% and 33.2% |
| Models written for ARC, run on SciQ, OpenBookQA, MMLU | about 47.7–50.1%, 30–36%, near 25% chance |
| CIFAR-10 raw pixels, eight written architectures | 21–28% (chance 10%) |
| CIFAR-10 on DINOv2 tokens, written table with no labels | 96.1%, and gradient descent with 100 labels 95.8% |
| AG News, four architectures written with no data | about 84%, and gradient descent needs 2,470 to 9,201 labels to match |
| Test-set overlap, Claude Opus 5.5 generation 5 | 2.1% of ARC test items against 2.0% of unseen ARC training items, and deleting the 68 flagged lines gives 54.3% |

## Where the claims go beyond the evidence

- **"Doesn't need Transformers" compares written knowledge with gradients on 3,358 items.** The written models carry what the writer LLM learned in its own pretraining. The gradient-descent baselines train from random weights on the ARC training set alone, with one seed. "WSP reaches higher accuracy than training from scratch" is true but unsurprising. The authors say that pretrained models stay far more accurate.
- **The interference reading of the written Transformer has another cause to rule out.** The given Transformer pools the mean over all tokens and reads one score vector. That design alone could put it below BM25, whatever the key overlap. The paper calls the result "consistent with interference". Its only direct measure is one that Claude Opus 5.5 made on its own practice questions.
- **Parameter counts mix two kinds of count.** A written table counts non-zero entries and their indices. A released architecture counts every parameter, though the median run changes 80% of them and leaves the rest random. The final Claude Opus 5.5 model has 112M elements and 3.35M counted parameters. The 1,800× ratio depends on this rule.
- **Few runs, and no intervals.** Most settings have two runs. The line-removal test samples 40 lines per model and reports means. No table gives a confidence interval, a seed spread or a test for differences of one to three points.
- **The authors designed their model after reading test-set analyses.** Tables 5, 6 and 18 use test accuracy, and Table 18 maps each design choice to one of these observations. The paper does not say how J = 16 and the prime were chosen. The writers never saw test accuracy, but the authors did.
- **The authors check contamination, but the null is not clean.** The masked-option recall rate is 15.3% on ARC test and 16.3% on ARC training. Both sets may be in the writers' pretraining data, so equal rates don't show that the writers saw neither. The overlap and deletion tests are more direct, and they support the authors.
- **Why the LLMs choose tables stays open, as the authors say.** They state that their data do not separate a structure that suits writing from one that the LLMs know. When row tables are forbidden, Claude Opus 5.5 builds another memory read by key.
- **Speed rests on one GPU.** All times are on one RTX 3090, and the timed Transformer has random weights. The authors state this limit themselves, and the concept-count limit (about 15,000 to 32,000 concepts with a dense table).

The authors state other limits themselves. The written models recall facts and chain none, and ARC-Challenge stays near 38%. Models written for ARC are at chance on MMLU. WSP fails on raw pixels, because a pixel weight has no meaning that a writer can name.

## Use in this repo

- **Gradient descent trained the registry's models, so the paper's units are not its units.** Every unit in a written table is a line whose meaning the writer stated before the weights existed. The registry reads neurons that gradient descent placed, with no stated meaning. So the line-removal result says nothing direct about a VERIFIED pole. The paper reports nothing on neurons of trained models, explanations, probes or random-unit nulls.
- **The BM25 control matches the confound baseline.** The paper asks whether a written model beats a plain search over its own text. `beats_confound_baseline` (`docs/PROTOCOL.md` §4) asks whether an explanation beats token identity, position, probe length or "never active". Both test whether the structure adds anything beyond the content. Ours is already a gate, so nothing changes.
- **Edit locality is #4782's interference question, in the authors' own written models.** One removed line changes 0.5 to 1.8 of 3,530 answers in a table model, and 4.8 to 20.3 in a written Transformer. This supports #4782's premise that dense overlapping keys spread an edit. It does not measure a trained model. `docs/analyses/base-instruct.md` is the repo's trained-model case: 252 of 311 Qwen2.5-0.5B hypotheses keep under instruction tuning, 81%.
- **Evidence for #5133 and #5118, at the whole-model level only.** §7.1 edits pretrained checkpoints. None of the 28 runs on checkpoints at 72.3% or above changes more than 0.07% of parameters, and they stay within about one point. The 11 overwrites of encoders above chance each lose accuracy. The paper measures side effects on ARC alone, with no per-unit read. This fits the gap in `docs/literature/fixing-models.md`: an edit tested on its own task only.
- **Evidence for #5144: a weight-only reading works when each weight has a stated meaning.** The written tables are bag-of-concept models, close to token-set rules, with a few hand-written rules for negation and order. They chain no facts and stay near 38% on ARC-Challenge. On raw pixels the writer cannot name a weight, and WSP fails. This supports #5144's framing: a weight-only check, like ROTATE (#3446), can reach token-set rules, and context rules need activations. It is an argument, not a count, so #5144 still needs its count.
- **The MLP reading is the key-value view that the repo uses.** Equation 1 reads an MLP's first-matrix rows as keys and GELU as the selection function. That is the view of `docs/literature/ffn-key-value-memories.md`, which `openinterpretability/harness/jlens.py` (`vocabulary_lens`) uses.
- **The ground-truth study is a small case of weight-space programming.** `docs/analyses/ground-truth-calibration.md` rewrites 12 GPT-2 neurons' input weights to plant known rules. It fits a logistic probe, so it is not WSP by this paper's rules, which forbid fitting on data. Its gate kept 7 of 12 plants and no rule that needs context, which matches the paper's limit. The paper's written Transformers are classifiers with mean pooling, not language models, so they cannot be scanned by the audit pipeline. #3584 (Tracr and InterpBench) remains the better source of known-ground-truth units.
- **Neighbours in #4782's strand.** `hypernetworks.md` and `fast-weight-memories.md` have a trained network write weights. This paper has a pretrained LLM write them with no gradients, and the authors say that WSP shares only the name with fast weight programmers. `models-rewrite-own-weights.md` has a model edit its own weights and finds a direction, not a unit. `conditional-memory-via-scalable-lookup.md` reviews Engram, the trained hashed lookup that the paper compares with its own model. `lora-learns-less-forgets-less.md` covers the gradient-trained low-rank update, which the paper names but does not test.
- **No new issue.** The two results that bear on open work go to #5144 and #5133 as evidence. The paper's own models can't be audited by our pipeline, and the fix campaign is the maintainer's to design.
