# Task Vectors Can Be Linearly Decomposed into Interpretable and Controllable Components

**Citation.** Yukiya Horiba (Keio University), Hakaze Cho (RIKEN and Tohoku University) and Takahiro Yakoh (Keio University). "Task Vectors Can Be Linearly Decomposed into Interpretable and Controllable Components". arXiv:2610.41743 [cs.LG], v1, 30 October 2026. [Paper](https://arxiv.org/abs/2610.41743). Preprint, no venue given. The paper gives no code or data link. Funding is from an SPDR grant and RIKEN's programs for junior scientists. **Mismatch:** the abstract says "multiple model families", but every result in the text is on Llama. The paper lists three Qwen 3.5 models and reports no numbers for them.

**Status.** Not applicable. The paper's task vector lives in the residual stream, and its two directions come from the label tokens' unembedding rows. It studies no neurons or heads, and the repo uses no in-context task vectors. The idea that carries over is in *Use in this repo*.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the arXiv abstract page and the full HTML text, v1, in two parts. We have every section and Appendices A–G in the source's own words, with Tables 1 and 2. We saw Figures 1–6 only as captions. So the layer, random-control and Japanese-task results rest on the paper's text. We did not read the PDF. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** An in-context task vector splits into three parts. One is a task-recognition (TR) direction that raises the candidate labels against the rest of the vocabulary. One is a task-learning (TL) direction that separates the right label from the wrong ones. The third is a residual. Reweighting the TR and TL parts improves the vector's accuracy, and each part has the effect that its name says.

**Method.**

- **The task vector.** v = h_task − h_query-only at the last token of one layer, averaged over 256 queries. The task comes from an instruction or from demonstrations (four per label).
- **The split.** TR is the all-ones direction over the candidate labels, and TL is a one-against-the-rest direction. Both map into the residual stream through the labels' unembedding rows. Middle layers use a metric from the softmax over the labels. The projections are regularised least squares. The residual is what is left.
- **The intervention.** u = α·v_TR + β·v_TL + v_residual, scaled to the norm of the last hidden state. Bayesian optimisation picks (α, β) in [0, 2]² for each layer, from 42 evaluations.
- **Models and tasks.** Llama 3.2 1B and Llama 3.1 8B, on SST-2, SUBJ, Financial PhraseBank and AG News, plus a Japanese financial task (JF-ICR).
- **Controls.** The original vector, random Gaussian directions, and random directions orthogonal to TR and TL, over five seeds.

**Findings, as the source reports them.** Accuracy in %, from Table 2, on the 500-example test set.

| Setting | Original vector → optimised |
| --- | --- |
| Llama 3.2 1B, SST-2, instruction | 11.2 → 79.6 |
| Llama 3.2 1B, SST-2, demonstrations | 80.4 → 82.8 |
| Llama 3.2 1B, AG News, instruction | 0.0 → 50.6 |
| Llama 3.2 1B, Financial PhraseBank, demonstrations | 60.4 → 63.4 |
| Llama 3.1 8B, SST-2, instruction | 0.4 → 84.2 |
| Llama 3.1 8B, SUBJ, demonstrations | 58.4 → 64.4 |
| Llama 3.1 8B, AG News, instruction | 14.8 → 65.8 |
| All 16 settings | Every one gains on the test set |
| Random directions (text only) | They "can also improve" accuracy. TR and TL gain more for instruction vectors, and less clearly for demonstration vectors |
| JF-ICR, Llama 3.1 8B (text only) | "little or no gain" |

## Where the claims go beyond the evidence

- **The baselines are chosen to lose.** In Section 4.1 and Appendix E, the layer is the one where the *optimised* vector scores best. The original vector is read at that layer. Several originals are near 0% (0.0, 0.4, 14.8). So the largest gains start from vectors that fail at that layer. Section 4.3 gives another rule, which picks the layer from the original vector's accuracy.
- **The search is selected on the same data.** The layer and (α, β) are maxima over 42 evaluations on the optimisation set. The test set guards against this, but its gains are as large, and some are larger.
- **No intervals, and small gains on demonstrations.** One run of the main method, with no spread. At n = 500 and 80% accuracy, a 95% interval is about ±3.5 points (our arithmetic). Several demonstration gains are 2–3 points.
- **The random control nearly matches.** Random directions also improve accuracy, and for demonstration vectors unconstrained random directions beat the orthogonal ones. The paper gives these numbers only in a figure, with no spread over the five seeds.
- **One model family.** Only Llama results appear, against the abstract's "multiple model families".
- **Two rows are identical.** For Llama 3.2 1B, SUBJ by instruction and by demonstrations match in every column. The paper does not say why.
- **"Magnitude" is direction.** The intervention is rescaled to a fixed norm, so α and β set the mix of directions, not their size.
- **"Orthogonal but sometimes dependent."** Section 5 says this of TR and TL. The method does not make them orthogonal, and the paper shows no test.
- **No code.**

The authors state several limits themselves. The split "is not guaranteed to provide the optimal directions". The isolated tests "do not establish that these components fully capture task recognition and task learning". The middle-layer metric is "a proxy for output sensitivity". They call their explanations hypotheses.

## Use in this repo

- **Not a weight task vector.** The *Bridges* pages ([`task-arithmetic.md`](task-arithmetic.md), [`ties-merging.md`](ties-merging.md)) use weight differences. This paper's vector is an activation at one position, so it has no bearing on #4782's question of how weight edits move units.
- **No units.** The two directions are spans of label unembeddings in the residual stream. The registry's units are MLP neuron poles and attention heads, and the paper never reads one.
- **One idea carries over: split a direction by what it does to the output.** TR asks "does it raise the answer set", and TL asks "does it pick the right answer in that set". A pole's output weights could be read the same way through the unembedding. The `vocab-projection-lookahead` skill already reads a pole's output through the vocabulary, so this adds no new test. It is a way to word a lookahead claim.
- **A warning for steering fixes.** If #4784 ever chooses a steering vector by search, the layer and the baseline must be chosen before the search. This paper's near-0% baselines show what happens otherwise.
- **No follow-up issue.**
