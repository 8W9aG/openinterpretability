# Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors

**Citation.** Thomas Hartvigsen (University of Virginia and MIT), Swami Sankaranarayanan (Sony AI), Hamid Palangi (Microsoft Research), Yoon Kim (MIT) and Marzyeh Ghassemi (MIT). "Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors". arXiv 2211.11031v5, 18 October 2023. The first version is v1, 20 November 2022. [Paper](https://arxiv.org/abs/2211.11031). [Code](https://www.github.com/thartvigsen/grace). The arXiv comment says "Accepted to NeurIPS 2023". We did not check the proceedings page. The page we read gives no funding or tool disclosure.

**Status.** Not applicable for now. No code, skill or open issue uses GRACE. It bears on the fix campaign, which the maintainer designs, so this page records evidence and one question, and it opens no issue.

**Read.** The arXiv abstract page and the v5 HTML, on 2026-11-18, read through WebFetch in three passes. A model read the text for us. It gave short verbatim quotes (at most 125 characters each), Table 2 cell by cell, and a summary of the rest. So the method and the findings below rest on those quotes, on Table 2 and on the model's summary. The second pass covered characters 40,000 to the end, 98,581. We did not see the figures, the PDF or the code. The holdout (rephrasing) results are in figures only, so we have no numbers for them.

## Summary

**Claim.** Deployed models make errors that arrive one at a time. Earlier editors degrade over many sequential edits. GRACE writes each fix into a discrete, local codebook at one layer and leaves the model's weights unchanged. The authors call it "the first method enabling thousands of sequential edits using only streaming errors".

**Method.**

- **Adaptor.** GRACE wraps one layer of a frozen model. Its codebook holds keys, values and a deferral radius ε for each key.
- **Keys and values.** A key is the cached activation that goes into the wrapped layer. A value is a new output for that layer, initialized at random and trained for 100 Adam steps on the edit's loss.
- **Inference.** If the nearest key (Euclidean distance) is within its radius, the layer outputs that key's value. If not, the model runs as before: "If a new input is unlike any cached keys, GRACE simply defers to f's pretrained weights."
- **Codebook upkeep.** A query outside every radius adds a key with radius ε_init. A query near a key with the same label expands that key's radius. A query near a key with a different label splits them: both radii shrink, and the query gets a new key.
- **One hyperparameter.** The paper calls ε_init the sole GRACE hyperparameter. For zsRE the search space was 0.1, 0.5, 1.0 and 3.0, and the main results use 0.5.
- **Tokens.** For classification, the value goes to all tokens. For generation, it replaces only the last token of the prompt.
- **Edited layers.** GPT2-XL: "the fully-connected component of the thirty-sixth layer" (`transformer.h[35].mlp.c_fc`). BERT: `bert.encoder.layer[10].output.dense`. T5: the paper names two different layers (see below).
- **Tasks.**

  | Task | Model | Edits | Retention set |
  | --- | --- | --- | --- |
  | Question answering | T5 (60M) | zsRE, 1000 edits | Natural Questions, 1000 samples |
  | Label shift | BERT | SCOTUS 1991–2009 | SCOTUS 1982–1991 |
  | Hallucination | GPT2-XL (1.5B) | 1392 sentence edits | OpenWebText, 1000 samples |

- **Hallucination edits.** The data comes from SelfCheckGPT, where the "authors prompt GPT-3 to generate 238 wikipedia-style biographies". The authors replace each wrong sentence with the matching sentence of the true Wikipedia entry. The prompt is "the GPT-3-generated text up until the hallucination".
- **Metrics.** TRR is test retention, the behaviour on the retention set. ERR is edit retention: earlier edits still hold. ARR is accurate retention, on sentences that were already right, and #E is the number of edits. The hallucination task uses perplexity, where lower is better.
- **Baselines.** Finetuning (FT), FT+EWC, FT+Retrain, MEND, Defer (after SERAC), ROME (hallucination only) and Memory (a memory-network ablation).

**Findings, as the source reports them.** All rows are from Table 2, after the whole stream of edits.

| Setting | GRACE | Best other method |
| --- | --- | --- |
| zsRE, T5, F1, average of TRR and ERR | .82 (TRR .69, ERR .96) | FT .69 (TRR .56, ERR .82) |
| SCOTUS, BERT, accuracy, average | .82 (TRR .81, ERR .82) | FT+Retrain .75 (TRR .67, ERR .83) |
| Hallucination, GPT2-XL, TRR perplexity | 15.84 | Memory 25.47 |
| Hallucination, ERR perplexity | 7.14 | FT 28.14 |
| Hallucination, ARR perplexity | 10.00 | Defer 10.04 |
| Hallucination, ROME | – | TRR 30.28, ERR 103.82, ARR 14.02 |
| Hallucination, FT | – | TRR 1449.3 |
| Time per edit, seconds (standard deviation) | .13 (.02) | Defer .07 (.02) |

Other results from quoted text:

- With ε_init = 0.5, zsRE ends with "137 keys/1000 edits".
- Larger ε_init gives smaller codebooks and "better generalization". The codebook has 547 entries at ε_init = 0.1 and 150 at 3.0. At 0.1, "60% of the holdouts do eventually have a key".
- An edited T5 was "only 1.32x slower than an unedited model on average".
- For GPT2-XL, "layers later than 35 led to less-consistent edit success".

## Where the claims go beyond the evidence

**"Thousands of sequential edits".** The main table stops at 1000 zsRE edits and 1392 hallucination edits. A figure caption mentions "unseen rephrasings of all 3k edits", and the authors leave "about 5k" edits as future work. So "thousands" rests on one setting that we saw only in a caption.

**Only figures show generality.** The text gives no holdout F1 or accuracy. A local edit must reach rephrasings to be a fix and not a lookup. That is the property the text measures least.

**The retention sets are far from the edits.** Natural Questions, an older SCOTUS span and OpenWebText test that unrelated inputs keep their behaviour. They don't test near neighbours that should *not* change, such as a prompt that shares the edit's form but has a different right answer. The split rule exists for that case, but the paper reports no measure of it.

**No seeds or intervals.** Table 2 gives one run per cell. Only the time column has a spread. The value initialization is random, so a seed spread would matter.

**Close margins.** On hallucination ARR, GRACE (10.00) and Defer (10.04) are almost equal. Defer is worse on TRR (8183.7), so the table does favour GRACE overall.

**Inconsistencies within the source.**

- SCOTUS #E is 381 in Table 2, and the appendix gives "252 keys/375 edits".
- The hallucination set has 592 already-right outputs in the main text and 516 in the appendix, as the model read them.
- For T5, Appendix A says "last dense layer of the last encoder block", and the text also names `encoder.block[7]`. Appendix B.1 says `encoder.block[4].layer[1].DenseReluDense.wo`.

**Limits the authors state.** The authors say that similarity search slows inference, and that they did not test edits at several layers. They also say that "Behaviors are edited in isolation", so related facts don't update. That last point is the main limit for a fix that should generalize.

## Use in this repo

**GRACE fixes without an explanation.** A key is the activation at the failing input. A value is trained on the right label. Nothing in the method reads a unit or a route. So if the campaign wants a baseline that ignores the explanation, GRACE is one candidate. Its result would show what an explanation adds. This is evidence for the maintainer, not a design.

**On GPT2-XL the edit overwrites neuron pre-activations.** The wrapped layer is `mlp.c_fc`, so a value replaces the input to every MLP neuron of layer 36 at the last prompt token. The registry explains MLP neuron poles, so in principle it can read a GRACE value: which explained poles the value turns up or down. The paper does not do this. It treats the codebook as inspectable only in the sense that one can list its keys.

**The weights do not change, so the registry's records stay valid off the edit.** Inputs outside every radius run the original model. So VERIFIED pole records and sister rows (`docs/analyses/sister-neurons.jsonl`) still describe the model on those inputs. A weight edit, such as ROME or finetuning, does not keep that property. Table 2 shows the cost of weight edits there: FT's TRR perplexity is 1449.3 against GRACE's 15.84.

**What it predicts for the #4781 failure.** EleutherAI/pythia-70m answers " days" to "A week has seven days and a year has twelve" (p 0.73, " months" at rank 4). The VERIFIED route copies the first clause's unit through the copy heads L3/H3, L4/H0 and L4/H7, and no verified unit supplies " months" (`changelog.d/4781.added.md`). A GRACE edit at a late layer would key on the last token's activation and output a value that gives " months". GRACE does not fix the route. The copy heads would still carry " days". The value would only override the result where the key matches. It would reach a rephrasing only if that rephrasing's activation falls inside the radius, and the paper gives no numbers for that.

**The shortlist is a ready test of generality and side effects.** `docs/analyses/hallucination-shortlist.jsonl` has 35 prompts with one right answer each. The 6 kept prompts all copy the first clause's unit, for example " seconds" for "A minute has sixty seconds and an hour has sixty". The other 29 the model answers right. One edit on the " days" prompt gives a direct split. The 5 other kept prompts test whether a fix of one instance reaches the failure. The 29 right prompts test near neighbours that must not change. The paper's retention sets test neither. The `neuron-steering-tests` skill's ablation and clamp reads use the same kind of before-and-after next-token measure.

**The coverage measures can show collateral damage.** Prompt coverage is the share of the answer that the explained units bring back alone (93.8% on the " days" prompt). Layer coverage (`openinterpretability/coverage.py`) works on the corpus loss. It is the share of each MLP layer's effect that the VERIFIED poles keep. A GRACE edit does not change the activations of units upstream of the wrapped layer. Off the edit's radius, it changes nothing at all. So a change in layer coverage after an edit would show corpus tokens that fall inside a radius. That is a direct check of the paper's claim that unrelated inputs keep their behaviour.

**Question for the maintainer.** Should the fix campaign include a fix that uses no explanation, such as GRACE, as its control? Only the maintainer can decide this, so this page records it as a question and opens no issue.
