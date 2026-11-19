# A Modern Self-Referential Weight Matrix That Learns to Modify Itself

**Citation.** Kazuki Irie, Imanol Schlag, Róbert Csordás and Jürgen Schmidhuber. All four are at the Swiss AI Lab IDSIA (USI and SUPSI, Lugano). Schmidhuber is also at KAUST. "A Modern Self-Referential Weight Matrix That Learns to Modify Itself". arXiv 2202.05780 [cs.LG], v1 11 February 2022, v2 17 June 2022. We read v2. [Paper](https://arxiv.org/abs/2202.05780), [code](https://github.com/IDSIA/modern-srwm). Venue: ICML 2022, PMLR 162, as the PDF and issue #4840 both say. An earlier version was at the NeurIPS 2021 Deep RL Workshop. The paper uses public datasets (Omniglot, Mini-ImageNet, FC100 and ProcGen) and releases no new data. Funding: ERC Advanced grant 742870 (AlgoRNN) and Swiss National Science Foundation grant 200021 192356 (NEUSYM). NVIDIA and IBM donated hardware, and CSCS project d115 gave compute. A footnote says that an earlier preprint reported 57.1% on FC100 because of a bug in test-data shuffling. v2 reports 44.5% after the fix, and the authors ran the main experiments again.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it in its synthesis on weight-space programming. The paper gives no method that applies to a transformer's units. It does give a clear case where the weights at inference are not the trained weights. Its own interpretability section reads one statistic and stops there. That is evidence for #5144, the count that caps any weight-only check.

**Read.** We read the arXiv v2 PDF on 2026-11-19, as text that `pdftotext` took from it. We have the source's own text for the abstract, §1 to §7, the acknowledgements and the references. We also have Appendices A (model details), B (few-shot details and extra results) and C (extra RL results). The values of Tables 1 to 10 came through as text. We also read the arXiv abstract page for the version history. We did not see the figures as images. Figures 4 and 6 to 9 are accuracy curves, and only their axis labels came through. So no number from them is here. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** One weight matrix can learn to write new values into all of itself while it runs, with outer products and the delta rule. Gradient descent trains only its initial values. This self-referential weight matrix (SRWM) is practical. It is competitive on few-shot image classification and on multi-task reinforcement learning (RL) in ProcGen.

**Method.**

- **The base, DeltaNet (Eqs. 1–4).** Slow weights W_slow make a key, value, query and rate β from the input. The delta rule then writes a new fast matrix: W_t = W_{t−1} + σ(β_t)(v_t − v̄_t) ⊗ φ(k_t). Here v̄_t = W_{t−1}φ(k_t) is the value that the matrix held for that key. The output is y_t = W_tφ(q_t), and W_0 = 0.
- **The SRWM (Eqs. 5–8).** There is no separate slow net. The matrix W_{t−1} itself makes y_t, q_t, k_t and β_t from φ(x_t). It then makes the new value v_t = W_{t−1}φ(q_t). It updates itself by the same delta rule. The only weights that gradient descent trains are W_0.
- **Details (§3, Appendix A).** In practice, β_t has four parts, one for each sub-matrix that makes y, q, k and β. The SRWM is multi-head, so it has fewer parameters than a DeltaNet of the same size. It replaces self-attention in a Transformer block, and it runs as a custom CUDA kernel.
- **Few-shot learning (§4.1).** 5-way, 1-shot and 5-shot, on Omniglot and Mini-ImageNet, with a Conv-4 vision backend. Labels come with each support image, and the model predicts the query's label. The baselines are DeltaNet and figures quoted from MAML, fwCNN-Hebb, HyperTransformer and SNAIL.
- **Sequential multi-task (§4.2).** Each label comes one step after its image. At step 74, the stream switches from Omniglot to Mini-ImageNet, or the reverse. The model must adapt with no outside help. The baseline is DeltaNet.
- **Multi-task RL (§4.3).** Six ProcGen games in the easy distribution, with IMPALA and a 15-layer residual vision model. Training runs for 300 M steps on 200 levels. Three test splits of 200 levels each and three training runs give nine points. The baselines are feedforward, LSTM, DeltaNet and "Fake SR", which is the SRWM without self-modification.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Omniglot 5-way 1-shot, test accuracy (Table 1) | DeltaNet 97.2 ± 0.0, SRWM 97.4 ± 0.0, SNAIL* 99.1 ± 0.2, MAML* 98.7 ± 0.4 |
| Mini-ImageNet 1-shot / 5-shot (Table 1) | DeltaNet 47.0 / 62.7, SRWM 47.0 / 61.4, HyperTransformer* 53.8 / 67.1 |
| Intervals in Table 1 | 95% interval over five sets of test episodes, not over seeds. * marks numbers from other papers |
| FC100 5-way 5-shot (Table 5) | DeltaNet 43.2 ± 0.1, SRWM 44.5 ± 0.1 (Conv-4-32), SRWM 46.3 ± 0.1 (Res-12-256), TADAM* 56.1 ± 0.4 |
| Sequential, Omniglot then Mini-ImageNet, total accuracy (Table 2) | Omniglot: DeltaNet 92.2, SRWM 92.3. Mini-ImageNet: DeltaNet 50.4, SRWM 53.3 |
| Sequential, reverse order (Table 7) | Mini-ImageNet: DeltaNet 50.7, SRWM 53.5. Omniglot: DeltaNet 92.2, SRWM 92.9 |
| ProcGen, six games, normalised score × 100, train (Table 3) | FF 22.5, LSTM 28.3, Fake SR 27.0, DeltaNet 35.0, SRWM 34.6 |
| ProcGen, six games, test (Table 3) | FF 16.4, LSTM 15.7, Fake SR 15.3, DeltaNet 18.6 (1.7), SRWM 20.0 (1.8) |
| Trainable parameters, RL (Table 3) | FF 626 K, Fake SR 959 K, LSTM 1.2 M, DeltaNet 1.05 M, SRWM 968 K |
| Reset the weight updates every 50 steps (§4.3) | Train 28.5 (1.2), test 16.1 (2.2), close to the model with no self-modification |
| Bigfish, test, multi-task / expert (Table 4) | Fake SR 4.7 / 15.8, SRWM 9.0 / 14.2 |
| ProcGen memory distribution, four games (Table 8) | Train: DeltaNet 51.8, SR-DeltaNet 59.0. Test: 38.0 (4.1) and 38.5 (3.2) |
| Learning rates σ(β_t) in RL (§5) | All four parts stay between 0.50 and 0.65. The authors find it hard to read anything more from them |
| Speed on one P100 (§5) | About 3,500 steps per second for FF and LSTM, 2,300 for DeltaNet and 1,700 for SRWM |

## Where the claims go beyond the evidence

- **The recursion claim has no experiment.** The abstract says that such networks can "meta-learn to learn" and "meta-meta-learn" in principle. §6 says that the experiments only test "practical ability on well known" tasks. Self-improvement is future work. No test separates self-modification at one level from more levels.
- **The gain over DeltaNet is small and mixed.** In Table 1, the SRWM ties DeltaNet or is lower: 61.4 against 62.7 on 5-shot Mini-ImageNet. In Table 3, its RL train score is lower (34.6 against 35.0). Its test score is higher by 1.4, with standard deviations of 1.7 and 1.8 over nine points. The authors call this "slightly better", which is fair.
- **Table 1's intervals hide seed variance.** The ± values come from five sets of test episodes for one trained model. Appendix B.3 says that seed variance is "typically high" on these tasks. For some seeds, the model stays at the chance level of 20%. The paper reports no seed spread for few-shot learning.
- **Table 2 and Table 7 have no intervals.** The multi-task gain on Mini-ImageNet is about 3 points, with no seeds and no spread. Appendix B.5 says that bad configurations or seeds stay at a poor solution, so the selection of runs matters.
- **The authors searched hyperparameters for each model.** Appendix B.2 gives the best settings for each model, but doesn't name the split that picked them. For Omniglot, the text says that training checks the validation set every 1000 steps. So the search probably used validation data.
- **Table 9 disagrees with Table 3.** In Table 9's aggregated train row, the "True" SRWM has 27.0 and "Fake" has 34.6. Table 3 gives the reverse, and the test row in both tables agrees with Table 3. So one of the two tables has the train values in the wrong columns.
- **A name differs between sections.** Appendix C.1 and Table 8 call the memory-distribution model "SR-DeltaNet", and Table 10 calls it "SRM-Delta".
- **The tasks don't need memory of a sequence.** The authors say so in §5: the main tasks can be solved by feedforward nets. The memory-distribution result is "only a preliminary result". There, the SRWM's train gain does not carry to test (38.5 against 38.0).
- **Interpretation stops at one statistic.** The authors read only the range of σ(β_t). They say that β_t alone doesn't describe the update, which also depends on the key and query.

The authors state several limits themselves. These are the tasks that feedforward nets can solve, the preliminary memory result, the seed variance and the FC100 bug and its fix. They also note that self-modification may not keep the training objective in the long run.

## Use in this repo

- **In an SRWM, the weights at step t are not the trained weights.** Gradient descent trains only W_0. Each step then writes the whole matrix from the input so far. So no reading of trained weights alone says what the layer computes at step t. This is the strongest form of the limit that [`fast-weight-memories.md`](fast-weight-memories.md) and [`fast-weights-recent-past.md`](fast-weights-recent-past.md) describe. There, only the fast part depends on the input. Here, every weight does.
- **The registry's models are not like this.** The four configured models (`pyproject.toml`) are Qwen2.5-0.5B, Qwen2.5-1.5B, GPT-2 and Pythia-70m. Their MLP weights stay fixed at inference. The part that depends on the input is the attention pattern and the activations. So a weight-only reading of an MLP neuron is possible in principle, as ROTATE (#3446) proposes. It can see a rule that the fixed weights hold, such as a token set on the input side. It can't see a rule that depends on context.
- **The authors could not read their own weights.** They had full access to β_t, keys, queries and every weight. They read one range (0.50 to 0.65) and found nothing more to say. This agrees with the repo's own result for weight-only readings. In [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md), ROTATE's weight-only kurtosis doesn't predict VERIFIED on 518 test poles. Every AUC interval includes 0.5. In [`docs/analyses/copy-suppression-l4h1.md`](../analyses/copy-suppression-l4h1.md), Pythia-70m's L4/H1 scores 42.0% on the OV test from weights. On ordinary text, copy suppression explains 1.8% of its effect.
- **What the paper adds to #5144.** #5144 asks how many VERIFIED explanations are token sets and how many need context. That share caps any weight-only check. The SRWM is the case where the cap is zero: every rule depends on the input so far. For the registry's fixed-weight models the cap is above zero. #5144 measures how far. The repo's plants point the same way. In [`docs/analyses/ground-truth-calibration.md`](../analyses/ground-truth-calibration.md), the gate kept 7 of 12 plants, and none that needs context.
- **Delta-rule layers would matter if a model with them joins.** #3331 proposes Qwen3.5-0.8B. [`models-rewrite-own-weights.md`](models-rewrite-own-weights.md) reports that the Qwen3.5 family mixes Gated DeltaNet linear attention with full attention. A DeltaNet layer's state is a fast weight that the delta rule writes, as in this paper's Eqs. 2–3. A weight-only check could not read that state, and `harness/heads.py` needs an attention pattern that such a layer never forms. [`linear-attention-transformers-are-rnns.md`](linear-attention-transformers-are-rnns.md) says the same for linear attention. The registry adds no new model before every model meets its target, so this is not work for now.
- **Weight-space programming (#3381, #4471).** The SRWM writes its own weights with a learned rule, and nothing in the written weights is meant to be read. #4471 has an agent write weights as readable Python code. So the two sit at opposite ends: the SRWM's writes are fast and opaque, and #4471's are slow and readable. The repo's own weight writing is the ground-truth plants (`scripts/ground_truth.py`). They write one neuron's input weights, not a rule that runs at inference.
- **For the interference question of #4782.** Each delta-rule write first removes the value that the key held, v̄_t, before it adds the new one. So a write to one key also modifies what overlapping keys return. The paper doesn't measure that interference. The RL ablation only shows that a reset of the updates every 50 steps removes the gain.
- **No single-neuron content.** The paper explains no unit, probe or head. It reports no null and no random weights. Its only interpretation is the range of β_t. Nothing in it bears on a pole's verdict.
- **No new follow-up.** #5144 already asks for the count that this paper bears on. This page is evidence for it, and a comment on #5144 can link the page.
- **Question for the maintainer.** Could a weight-only reading ever count toward a verdict, for example to contest a token-set claim? [`fast-weights-recent-past.md`](fast-weights-recent-past.md) records the same question. That would modify the decision rule, so it would need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). We record it as a question only.
