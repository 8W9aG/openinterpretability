# In-context Learning and Induction Heads

**Citation.** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann and Amanda Askell. Then Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston and Andy Jones. Then Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish and Chris Olah. All are at Anthropic. "In-context Learning and Induction Heads". Transformer Circuits Thread, published 8 March 2022. [HTML, the canonical version](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). [arXiv 2209.11895](https://arxiv.org/abs/2209.11895), v1 24 September 2022, a print of the HTML page. The thread is not a peer-reviewed venue. The page's change log has one entry, dated 2022-09-20, which revises the definition of prefix matching. The authors release no code, models or data, and the models are Anthropic's own. We found no funding disclosure.

**Status.** Used. The registry's head profiles measure the paper's prefix-matching pattern on repeated random tokens (`openinterpretability/harness/heads.py`), and the `attention-head-audit` skill takes its holdouts from the paper. The paper also gives the mechanism that the #4781 answer route names for L3/H3.

**Read.** On 2026-11-18. The HTML page is about 395,000 characters, and image data takes most of it. A web reading of it returned only the abstract, the introduction and the definitions as text. So we read the arXiv PDF instead, as text that `pdftotext` extracted, and we read that text ourselves, not through a summary. We have the source's own text for the abstract, the introduction, Key Concepts and Arguments 1 to 6. We also have the Model Analysis Table's prose, Model Details, Unexplained Curiosities and Discussion. Comments & Replications, the change log, the footnotes and the author contributions are in the text too. We also have the appendix on where induction heads form, the ablation methods and the head evaluators. We only looked at the headings of Related Work, and we did not read the rest of the appendix. We did not see the figures, the Model Analysis Table's plots or the interactive views. So this page has no number read from a curve.

## Summary

**Claim.** Induction heads complete patterns of the form `[A][B] … [A] → [B]`. The paper gives "preliminary and indirect evidence" that they "might constitute the mechanism for the majority of all 'in-context learning' in large transformer models". The evidence is causal for small attention-only models and correlational for larger models with MLPs.

**Method.**

- **Definition.** An induction head has two properties on a repeated sequence of random tokens. *Prefix matching*: it "attends back to previous tokens that were followed by the current and/or recent tokens". *Copying*: its output "increases the logit corresponding to the attended-to token".
- **Evaluators.** Each score is the mean of 10 examples of 25 random tokens, without the most and the least common tokens. For prefix matching, the 25 tokens occur 4 times, after a start token. The copying score is the head's direct-path logit effect on the attended token, after mean subtraction and a ReLU, scaled to −1 to 1.
- **In-context learning score.** "The loss of the 500th token in the context minus the average loss of the 50th token in the context, averaged over dataset examples."
- **Models.** 34 decoder-only models, each from one training run, in four series. Small attention-only models and small models with MLPs have 1 to 6 layers, d_model 768 and 12 heads per layer. Their training ran for about 10 billion tokens, with 200 snapshots. Full-scale models range from 4 layers and 13M parameters to 40 layers and 13B parameters, with 14 or 15 snapshots. "Smeared key" models have 1 or 2 layers.
- **Arguments.** (1) The two changes happen at the same time in training. (2) An architecture change that makes induction heads easier moves the change. (3) Test-time ablation of single heads, with the attention patterns frozen. (4) Examples of abstract behaviour in induction heads of the 13B model. (5) The mechanism: a previous-token head, then K-composition into the induction head's keys. (6) Continuity from small to large models.
- **Abstract version.** The authors hypothesise "fuzzy" or "nearest neighbor" heads. These complete `[A*][B*] … [A] → [B]`, where A* ≈ A and B* ≈ B "are similar in some space".

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| All models with more than one layer, over training | In-context learning develops "abruptly in a narrow window early in training (roughly 2.5 to 5 billion tokens)" |
| In-context learning score, before and after the window | Less than 0.15 nats before. About 0.4 nats after, constant across model sizes and later training |
| Share of the final in-context learning | Over 75% forms in the window |
| Prefix-matching scores over training | Induction heads form in the same window. The one-layer model has none, and almost no in-context learning |
| Training loss | The window is "the only place in training where the loss is not convex" |
| Smeared-key models | In-context learning forms in one-layer models, and earlier in two-layer models |
| Head ablations, small attention-only models | "Almost all the in-context learning" comes from the induction heads. The authors ran no ablations on the full-scale models |
| 13B model: literal copying head, layer 21 of 40 | Copying 0.89, prefix matching 0.75 |
| 13B model: translation head, layer 7 of 40 | Copying 0.20, prefix matching 0.85 |
| 13B model: pattern-matching head (layer 8 of 40 in the table, 26 in the text) | Copying 0.69, prefix matching 0.94. About 65% of its attention from the colons goes to the right positions |
| Harry Potter text, start against end of the window | Repeated sequences become easier. A token "followed by a different token than it previously was" becomes harder, for example the first " Mrs Potter" after two " Mrs Dursley" |
| Redwood replication, two-layer attention-only models | An exact previous-token pattern recovers 99% of the loss difference of the ablation. A simple `[A][B]…[A]` pattern for the induction head recovers about 65%, and an `[A][B][C]…[A][B]` term adds 10% |

Footnote 17 adds that ablating most other heads increases the in-context learning score. The authors' reason is that some tokens can come from "normal prediction" or from in-context learning. If an ablation harms normal prediction, in-context learning predicts more of those tokens.

## Where the claims go beyond the evidence

- **The title's claim is wider than the causal evidence.** The ablations cover small models only. For large models the case rests on co-occurrence, three hand-picked heads and continuity. The authors say so in their own table of evidence strength, and they call the case "not conclusive".
- **One training run per model, and the same data order.** Every model on a dataset "saw the same examples in the same order". So the shared timing of the phase change could come from the data order. The authors name this in footnote 11.
- **Low time resolution for large models.** The full-scale models have 14 or 15 snapshots. The authors say that co-occurrence over 15 points "is less surprising and weaker evidence".
- **A confound the authors name.** The phase change may be when the model learns to compose heads at all. Then other composition heads could form at the same time and carry part of the effect.
- **The metric uses two set token indices.** The authors test other indices against the final token. But the loss at token 50 falls over training. So a constant 0.4 nats can hide a change of mechanism, as the authors also say.
- **The abstract examples are anecdotes.** Argument 4 shows three heads of one model on hand-made texts. The 65% figure is the only number for the abstract behaviour, and it has no baseline or interval.
- **No intervals or significance tests.** No number in the text has an interval, a seed count or a test.
- **Inconsistencies.** The pattern-matching head is at layer 8 of 40 in the table and at layer 26 in the text. Argument 1 puts the window at 2.5 to 5 billion tokens, and Model Details puts it at 1 to 3 billion for the small models. Argument 4 says "up to 12B parameter models", but the largest model has 13B. Argument 5 refers to the abstract heads of "Argument 2", which are in Argument 4.
- **The evaluators differ in one detail.** Footnote 31 says the copying evaluator omits the start token by mistake, which can lower the scores of some prefix-matching heads.
- **The authors release nothing.** The models, the data and the evaluator code are Anthropic's. A reader can't rerun the results, except through external replications such as the two in the paper's comments.

## Use in this repo

**The head profile measures the paper's prefix-matching half.** `pattern_scores` (`openinterpretability/harness/heads.py`) records previous-token, duplicate-token and induction scores, and `HeadProfile` (`openinterpretability/core/profiles.py`) cites this paper. Our version differs from the paper's evaluator in four details:

- 32 random tokens repeated twice, not 25 tokens repeated 4 times.
- Tokens drawn from the whole vocabulary, without the paper's exclusion of the most and least common ones.
- No start token.
- Only the attention side. The registry has no copying score, so "induction" in a record means attention to the right token, not a write of that token.

The registry reports the scores and does not gate on them.

**The audit skill uses the paper's test.** The `attention-head-audit` skill takes repeated random-token sequences as holdouts for an induction head, and repeats with the copy broken as hard negatives. Its positive control is the five known induction heads of GPT-2 small: L5H1, L5H5, L6H9, L7H2 and L7H10.

**The #4781 copy heads are not all induction heads by the paper's definition.** The answer route (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) names L3/H3 an "induction-style copy head", with L4/H0 and L4/H7. Their induction scores on 32-token repeats are 0.13 (L3/H3), 0.33 (L4/H7) and 0.05 (L4/H0). Uniform attention over the second copy would give about 0.02. The VERIFIED explanation of L4/H0 is a head for the comma that closes a comma-delimited insertion, not an induction head. L3/H3's VERIFIED explanation is a broad induction-style head on repeats of rare words and phrases. L4/H7's VERIFIED explanation is a broad head that grows with context and fires at the second member of a series. So, on our scores, L4/H7 has the clearest prefix-matching pattern, and L3/H3 has a weaker one. None of the three has a copying score.

**What the paper predicts for the " days" failure.** The prompt is "A week has seven days and a year has twelve". At ' twelve', no earlier ' twelve' exists, so a literal induction head has nothing to match. The paper's fuzzy form fits instead. A* is ' seven', and the layer-0 count-word neurons make it similar to A, ' twelve'. B* is ' days', and the head copies it. The paper predicts the copy of B*, not a mapped B such as ' months'. Its only mapped case is the translation head of a 13B model. The authors say that head's output "needs to be further processed by later layers". Pythia-70m has two layers after L3. The paper also predicts the cost: the Harry Potter case shows that a token with a new continuation becomes worse after the phase change. " months" after "a year has twelve" is that case.

**The answer record's probes fit this prediction, with one exception.** The model copies the earlier noun whatever is true. "Twelve months make a year and seven" gives ' months', and "A week has seven colours and a year has twelve" gives ' colours'. "There are seven days in a week and twelve" gives ' days', so the count word alone is enough for the match. The exception is about which earlier match wins. "A week has seven days and a year has twelve months and a decade has ten" gives ' days', not ' months'. A head that copies after the most recent match would give ' months'. The paper says nothing on how a head chooses among several earlier matches. So it does not explain why the first clause wins.

**What the paper predicts but the repo has not measured.** Footnote 17 implies that the copy and the model's in-weights prediction compete for the same token. Ablating L3/H3 and L4/H7 alone, and reading the rank of ' months', would tell whether pythia-70m has a "year → months" fact that the copy hides. That is question 1 of #5015, and the result belongs there as a comment. The paper also predicts that the copy appears when the induction heads form. Pythia's checkpoints make this testable (follow-up below). `papers/neuron-development/` already reads Pythia-70m at 20 checkpoints. Pythia trains on about 300B tokens in 143,000 steps (Biderman et al., 2023, not reread for this page). So the paper's 2.5 to 5 billion tokens would be near steps 1,200 to 2,400. That is a scale assumption, because the paper's window is for Anthropic's models.

**Evidence for the fix campaign, not a design.** In small models the paper finds that induction heads give almost all in-context learning. If that holds for pythia-70m, the " days" copy is a side effect of a general mechanism, not a fault of one prompt. A change that removes the copy at L3/H3 or L4/H7 would then also change right copies. An example is ' legs' for "A spider has eight legs and a dog has four". The answer record gives ' legs' for that prompt. The maintainer designs the campaign.

**A question for the maintainer.** The paper defines an induction head by prefix matching and copying together. The registry records only the attention side, and the explanations still say "induction-style". Should that word need a copying score? The scores are not gated, so this changes labels, not the decision rule.

**Follow-ups.**

- Add the paper's copying score to `HeadProfile` (area: pipeline, priority: medium). It is the OV half of the definition, and it would tell whether L3/H3 and L4/H7 write the token they attend to.
- Track the " days" answer and the copy heads' induction scores over Pythia-70m's checkpoints (area: research questions, priority: medium). The paper predicts that the wrong answer appears when the induction heads form.
