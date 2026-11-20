# Transformer-Patcher: One Mistake worth One Neuron

**Citation.** Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong and Zhang Xiong. The affiliations are Beihang University, Mila and the University of Montreal, and WeChat AI at Tencent. "Transformer-Patcher: One Mistake worth One Neuron". [arXiv:2301.09785](https://arxiv.org/abs/2301.09785), version 1 of 24 January 2023, the only version. The arXiv comment says "accepted in ICLR 2023". We didn't check that on OpenReview. The code is at [github.com/ZeroYuHuang/Transformer-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher). Funding: the State Key Laboratory of Software Development Environment of China, grant SKLSDE-2023ZX-16. The text gives no disclosure of tool use.

**Mismatch.** Appendix B computes the mistakes per folder from the accuracy, not the error rate, but the counts come out close. Section 5.1 says the runs were repeated "2020 times", and Appendix B says 20. The text gives GeLU(−3) as about 0.004, but it is about −0.004.

**Status.** Planned. #5220 names it as the closest published method to its arms G (a written unit) and I (a gated corrector). The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in two fetches: Sections 1 to 6, the references and Appendices A to C. The PDF came back as binary, so we didn't read it. We didn't see any figure as an image. So the success-rate curves of Figure 4 and the locality curves of Figure 7 appear here only as the text describes them. We didn't read the code.

## Summary

**Claim.** A model can be fixed one mistake at a time by adding one neuron to its last FFN layer. Each neuron fires on its mistake and stays silent elsewhere. Over a stream of thousands of mistakes, nearly every edit holds and the model keeps its accuracy.

**Method.**

- **The patch.** For each mistake, one new FFN neuron with a key $k_p$, a value $v_p$ and a bias $b_p$, in the last FFN layer. The output is $\mathrm{FFN}(q) + a_p v_p$, with $a_p = \mathrm{GeLU}(q \cdot k_p + b_p)$. All other weights stay frozen. In generation, each wrong target token has its own patch, up to 5 for each example.
- **The start.** The key starts at $q_e / \lVert q_e \rVert^2$, so the mistaken query's activation starts at 1. The value starts large, so it outweighs the existing values.
- **The losses.** The total is $l_e + a\, l_a + m\,(l_{m1} + l_{m2})$, with $a = 1$ and $m = 10$:
  - **Edit loss.** The task loss on the one edit example.
  - **Activation loss.** It raises the patch's activation on the mistaken query.
  - **Memory loss.** It keeps the patch silent on up to 40,000 stored queries from the training data. $l_{m1}$ pushes the patch's pre-activation below $\beta = -3$ on the top 1,000 of these queries. $l_{m2}$ makes the mistake's pre-activation exceed theirs by a margin $\gamma = 3$.
- **Models and tasks.** Fact checking on FEVER with BERT-base, and closed-book question answering on zsRE with BART-base. The edits are the models' real mistakes, not invented counterfactuals.
- **Protocol.** The edit set is split into 20 folders, each run as one sequence of edits. A single stream of all edits is also run.
- **Metrics.** SR is success on the edit, and GR on its paraphrases. ER counts the edits that still hold at the end. TrainR and TestR are accuracy on training and test data, as a ratio to the first model.
- **Baselines.** Fine-tuning with and without a KL term, MEND, KE and SERA, a variant of SERAC.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Folders, FEVER and zsRE (Table 1) | SR 1.00 and 1.00, GR 0.82 and 0.82, ER 1.00 and 0.99, TrainR 0.999 and 0.997, TestR 1.000 and 0.996 |
| Same, fine-tuning all weights (Table 1) | GR 0.74 and 0.68, ER 0.83 and 0.43, TestR 0.994 and 0.910 |
| Same, SERA (Table 1) | GR 0.89 and 0.90, TestR 0.916 and 0.901 |
| Single stream, zsRE, 2,308 edits (Table 2) | SR 0.99, GR 0.81, ER 0.97, TrainR 0.912, TestR 0.948 |
| Single stream, FEVER, 998 edits (Table 7) | SR 1.00, GR 0.82, ER 1.00 |
| Ablation, no memory loss (Table 3) | GR 0.94 and 0.94, ER 0.61 and 0.21, TrainR 0.737 and 0.069 |
| Mean absolute activation (Table 4) | past edits 0.00 on both tasks, random test mistakes 0.05 and 0.10 |
| Layer, zsRE (Section 5.3) | layers 0 and 1 make no effective edits. GR rises from about 0.74 at layer 2 to 0.81 at layer 5 |
| Model growth after the single stream | 1.4% larger (FEVER) and 4.5% larger (zsRE) |

## Where the claims go beyond the evidence

- **Paraphrase reach is measured, not trained, and it trades against silence.** No paraphrase enters the losses. GR is about 0.82 with the memory loss. Without it, GR is 0.94 but most edits and most of the model's accuracy are lost. The paper doesn't discuss this trade.
- **Interference between patches is measured only on average.** Table 4 gives mean absolute activations. Figure 5c shows a "clear diagonal" for each patch's activations. No table gives the worst case, or the share of patches that fire on other edits.
- **The memory is the training data.** Silence is enforced on queries from the same distribution as the test data. The paper doesn't test silence on text of another kind.
- **Two small models.** BERT-base and BART-base only, with no decoder-only model.
- **Means over folders, no intervals.** SR and GR are pooled with no deviation. Table 5 gives deviations for ER, TrainR and TestR only, and no seeds.
- **The single stream costs accuracy.** On zsRE, TrainR falls to 0.912 and TestR to 0.948, which the text still calls keeping performance.

The authors name some limits themselves. The edit and test data have different distributions. The patches may overfit to edit examples. And the model grows with each patch.

## Use in this repo

- **The issue's three questions, as the source answers them.**
  - *How the neuron stays silent:* a memory loss on up to 40,000 stored training queries, with a margin. Without it, the edits don't hold.
  - *How the key reaches paraphrases:* it isn't trained to. GR of about 0.82 comes from the key's start at the mistaken query and its geometry.
  - *How patches compose:* each one is independent, and the memory grows with the edits. Past-edit activations stay near 0 on average.
- **It agrees with the repo's gate result on depth.** The [copy-gate analysis](../analyses/copy-gate.md) for #5233 finds a usable gate only late. After the last layer, a probe trained with corpus text catches 58% of held-out wrong copies. It opens on 5 of 97 right copies and on none of 18,000 corpus tokens. At layers 2 and 3 the same kind of gate catches 12% to 21%. T-Patcher puts its neuron in the last FFN layer, and early layers make no effective edits. The two results point the same way.
- **Its memory loss is the corpus arm of that gate study.** The `logistic_corpus` probe adds corpus tokens as negatives, as T-Patcher's memory adds stored queries. The patch's key is a trained probe, with the bias as its threshold.
- **Its patch is a unit the registry could explain.** One added neuron with one key and one value is a pole in the registry's sense. A patch could have a registry record like any pole: an explanation, probes and a verdict. Our reading, not the paper's: that record would test whether the key fires on the mistake's meaning or on its surface form.
- **Its unit isn't a dormant slot.** Arm G of #5220 writes into a dormant pole (#5134), with no gradients. T-Patcher adds a new neuron and trains it. Its numbers bear on the gate and the value, not on writing with no gradients.
- **Questions for the maintainer, recorded on this page only.**
  - Should arms G and I report interference per added unit as a worst case, not a mean? Table 4's means hide the patches that misfire.
  - Should the silence check include corpus text of another kind, as the gate study does, and not only data like the edits?
- **Related pages.** The [editing survey](editing-llms-survey.md) reports T-Patcher on GPT-J and CounterFact, with locality 8.37. [GRACE](aging-with-grace.md) and [WISE](wise-lifelong-editing.md) are the other methods that add parameters for each edit. [`ffn-key-value-memories.md`](ffn-key-value-memories.md) is the key–value view of an FFN neuron that the patch uses.
