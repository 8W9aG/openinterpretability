# A Bias in Disguise: Tiny Gradient Edits Stop Thinking Loops

**Citation.** Hussamuddin Syed, Independent Researcher, sole author. "A Bias in Disguise: Tiny Gradient Edits Stop a Small Language Model's Thinking Loops and Complement a Frontier Model's Weight Edit". [arXiv:2611.33133](https://arxiv.org/abs/2611.33133) [cs.LG, cs.CL], version 1 of 24 November 2026, the only version. Preprint, no venue given. The licence is CC BY 4.0. The text links no code or data, but it evaluates the public checkpoint of the other edit ([RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)). It states no funding. The compute was rented GPUs, "about $49" for the whole project. The acknowledgements say that Claude (Anthropic) helped write the code and the draft, and that the author checked every result.

**Status.** Planned. The fix campaign (#5220) plans small weight edits. This paper shows that a gradient-selected sparse edit can work as a constant bias rather than through units with a meaning. No code, skill or protocol in this repo depends on it. The maintainer designs the fix campaign, so this page reports evidence only.

**Read.** On 2026-11-28 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It returned the whole text, Sections 1 to 9, the acknowledgements and the references, with Tables 1 to 7 as text. Quotes were capped at 125 characters, so the rest came back paraphrased. The HTML is a LaTeXML conversion, and some symbols in it are garbled. We saw only the captions and labels of Figures 1 and 2, not their data. We did not read the PDF.

## Summary

**Claim.** With thinking on, Qwen3.5-0.8B rarely closes its think block within 4,096 tokens, and so it scores 15.4% on MATH-500. Training as few as 140 gradient-selected weights on the model's own answers lets it finish, and lifts the score to 39.0–45.5%. The edit mostly adds a constant vector to the stream, "a bias in disguise", because the model has no bias terms. Added at half strength to another, gradient-free edit of the same model, it reaches 62.2% under that edit's protocol.

**Method.**

- **Model.** Qwen3.5-0.8B, 24 layers, with linear-attention and full-attention layers in a 3:1 pattern. No linear layer has a bias.
- **Data.** The base model, with thinking off, answered each of the 7,498 MATH training problems four times. The answers graded right, at most two per problem, gave 7,111 examples. Each was rewritten as a thinking trace that closes its think block and then gives the answer. The paper says "No test problem and no other model is involved."
- **Choosing the weights.** The paper sums the absolute gradient of each weight of the 150 projection matrices. It sums over 64 micro-batches of about 4,096 tokens. The top $k$ weights across the model become trainable. This is a gradient-magnitude form of the FISH mask (Sung et al.). The control takes the same number of weights at random.
- **Training.** Only the $k$ chosen values change: AdamW at a learning rate of $3 \times 10^{-4}$, one epoch of 361 steps.
- **Evaluation.** MATH-500 with thinking on, a budget of 4,096 tokens and two sampled seeds. An answer counts as closed if it finishes and contains the closing think tag. Comparisons use exact McNemar tests, or sign tests where each problem has two samples. GSM8K, HumanEval and LiveCodeBench test transfer.
- **The other edit.** GPT-6 Astra, as in *Sparks of Weight-Space Programming* (`rikiyat-weight-space-programming.md`), edited 64 columns of the MLP output matrix in layers 15 and 19 with no gradients. In each layer, the edit is rank one along one direction. The paper adds its own gradient edit, at strength 1, 0.5 or 0.25, to that checkpoint. It evaluates the sum under Astra's protocol of 32,768 tokens and two samples per problem.
- **Locating the fix.** The paper reads the channels' activation statistics on 19 thinking traces, and ablates and trains single columns. A hook then adds three constant vectors to the unedited model.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Base model | MATH-500, thinking on, 4,096 tokens | 15.4%, with 1 or 2 of 500 answers closed |
| Base model, thinking off | MATH-500, greedy, 2,048 tokens | 41.0% |
| Gradient edits | MATH-500, thinking on, 4,096 tokens, mean of two seeds | 39.0% at 140 weights, 42.9% at 1,398, 45.5% at 139,750 |
| Random weights | Same setting | 16.3% at 1,398 weights, 43.4% at 139,750 |
| Where the 1,398-weight edit lies | Weight counts | 888 of its weights are in four input columns of `linear_attn.out_proj` in layers 0–2. For the 140-weight edit, 113 of 140 are |
| Those four columns alone, trained | 888 weights | 25.0% and 24.8% |
| Four random columns of the same matrices | 4,096 weights | 20.4% and 20.6% |
| Three constant vectors added to the base model | First 96 MATH-500 problems | 45.8%, against 42.7% for the edited model and 11.5% for the base |
| Transfer of the 139,750-weight edit | GSM8K, thinking on | 13.6% → 56.1% ($p < 10^{-100}$) |
| Transfer | HumanEval, thinking off | 34.1% → 29.3% ($p$ = 0.15) |
| Astra's edit, reproduced | MATH-500, Astra's protocol, strict | 52.1% |
| Astra's edit plus the 139,750-weight edit at half strength | Same | 62.2% (+10.1, $p = 3 \times 10^{-9}$) |
| The same at full strength and at quarter strength | Same | 49.7% and 59.8% |

**Where the claims go beyond the evidence.**

- **The combined score is tuned on the test set.** The strength of the gradient edit in the sum was chosen from three values on MATH-500 itself, with no held-out split. The author states this: 62.2% is "the best of three". Quarter strength also beats Astra's edit, by 7.7 points, but full strength is 2.4 points below it.
- **A large edit is not small for this behaviour.** At 139,750 weights, random positions reach 43.4%, close to the gradient edit's 45.5%. So the gradient selection matters at 1,398 weights (16.3% at random), but not at the largest size. The author says so.
- **The bias test is small.** The constant vectors were tested on 38 teacher-forced traces and on the first 96 problems, with a different sampler. The author has not tested whether the rest of the edit, outside the four channels, also acts as constant vectors.
- **One model, one task, two seeds.** Every result is on Qwen3.5-0.8B trained on MATH, with two seeds and no intervals. Many comparisons are made with no correction for them. The author states the one-model limit.
- **The fix is mostly a budget fix.** The edits leave thinking-off scores unchanged at 40.2–41.4%, and the base model already scores 41.0% with thinking off. What the edit adds is that the model finishes within the budget.

## Use in this repo

- **The edited weights were chosen by gradient, not read as units.** The issue asks this. The paper picks weights by summed absolute gradient, with no reading of neurons. After training, it explains the edit at the level of channels. Four input channels of the linear-attention output matrices in layers 0–2 are nearly constant. Changing their write columns adds a near-constant vector. There are no explanations of single neurons, no probes and no feature labels. So the paper is evidence about where a sparse fix lands, not about fixes guided by an explanation.
- **A sparse fix can be a bias, not a unit.** The fix campaign (#5220) asks whether an edit aimed at explained units generalises better than other edits. This paper shows a case where the best sparse edit acts as a constant shift, through channels that are nearly constant. The channels' meaning plays no part. So an arm that edits explained poles needs a constant-shift arm of the same size beside it. Without one, the campaign can't tell a unit's meaning from a bias. This is our step from the paper, not the paper's claim.
- **Nearly constant channels are like default-on poles.** The four channels are chosen because their activation is nearly the same on every token. `papers/default-on-poles/` finds that most GPT-2 small neurons rest below GELU's minimum, so their negative side is on for almost every token. Holding that side at its mean costs 0.19 nats, but zeroing it costs 3.72. In both cases a nearly constant activation carries a large, steady part of the stream. The paper's channels are in linear attention, not in an MLP, so the match is only an analogy.
- **The two edits read as directions.** Astra's edit is rank one along one direction in each layer, and the paper reads its projection on looping and finished traces. A direction in the stream is not a neuron, and the review audits neurons. So neither edit gives the registry a unit to audit as it stands.
- **No follow-up issue.** The evidence bears on how #5220's arms are compared, and that design is the maintainer's. We record it here and as a comment on #5220, and we open no issue.
