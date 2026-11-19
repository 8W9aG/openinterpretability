# Progressive Neural Networks

**Citation.** Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu and Raia Hadsell (Google DeepMind, London). The first three authors contributed equally. "Progressive Neural Networks". [arXiv:1606.04671](https://arxiv.org/abs/1606.04671), v1 of 15 June 2016, v2 of 21 June 2016, v3 of 7 September 2016 and v4 of 22 October 2022. Preprint, no venue given. The text gives no code or data links, and no funding or tool statement. All authors are at DeepMind, which also built the A3C agent that the paper uses.

**Status.** Not applicable. The repo trains no networks and adds no capacity to a model, so it has nothing to build on the method. One idea carries over. It is a guarantee by construction, and no measurement in the repo can test it.

**Read.** On 2026-11-19 we downloaded the arXiv v4 PDF and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the extracted text in the source's own words. That is the abstract, sections 1 to 6 and the references. It is also the supplementary material: A perturbation analysis, B compressibility, C setup, D learning curves and E Labyrinth. It includes Table 1. We did not see the figures as images. The transfer matrices (Figures 4a, 6a and 8a) and the AFS plots (Figures 5, 7 and 10) came through only as captions. The text of the learning curves in Figures 11 to 13 came through as axis labels only. We did not compare v4 with the earlier versions.

## Summary

**Claim.** A network that adds a new column for each task, and freezes the old columns, can't forget old tasks. Lateral connections let each new column reuse the old columns' features. On reinforcement-learning tasks, this transfers as well as fine-tuning, or a little better.

**Method.**

- **The architecture.** The first column is a deep network trained to convergence on task 1. For task 2, its parameters are frozen, and a new column starts from random weights. Layer i of column k takes the layer i − 1 output of its own column and of every earlier column. Earlier columns receive no input from later ones, so their forward pass never changes.
- **Adapters.** The lateral inputs pass through a small MLP, after a learned scalar that starts small. Its hidden layer projects to the width of the target layer, so the lateral parameters grow with the first column's size. Convolutional layers use 1 × 1 convolutions for this.
- **Transfer analysis.** Average Fisher Sensitivity (AFS) is a diagonal Fisher of the policy, taken with respect to each layer's normalized activations, not its parameters. Each fully connected neuron or convolutional feature map has one value. The value is then normalized across columns, so it gives each column's share of the policy's sensitivity in a layer. Average Perturbation Sensitivity (APS) adds Gaussian noise to a whole layer of one column. It finds the noise level that halves the score, and normalizes its precision across columns.
- **Setup.** The agent is A3C with 16 workers. The network has 3 convolutional layers of 12 feature maps each, and a fully connected layer of 256 units. Learning rate, entropy cost, gradient clip and the lateral scalar come from small grids. Runs last at most 4 × 10⁷ agent steps.
- **Score.** Each result averages the top 3 of 25 jobs, each with its own seed and hyperparameters. The measure is the area under the learning curve. The transfer score is that area relative to baseline 1, a single column trained only on the target task.
- **Baselines.** Baseline 2 pretrains one column on the source task and fine-tunes the output layer only. Baseline 3 fine-tunes the whole column. Baseline 4 is a 2-column progressive network whose first column is random and frozen.
- **Domains.** Pong Soup: eight variants of Pong (Noisy, Black, White, Zoom, V-flip, H-flip, VH-flip). Atari: sources Pong, River Raid and Seaquest, with 12 target games. Labyrinth: eight 3D maze foraging levels.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Table 1, Pong Soup, mean / median transfer (%) | Baseline 2: 35 / 7. Baseline 3: 181 / 160. Baseline 4: 134 / 131. Progressive 2 col: 209 / 169. Progressive 3 col: 222 / 183 |
| Table 1, Atari, mean / median transfer (%) | Baseline 2: 41 / 21. Baseline 3: 133 / 110. Baseline 4: 96 / 95. Progressive 2 col: 132 / 112. 3 col: 140 / 111. 4 col: 141 / 116 |
| Table 1, Labyrinth, mean / median transfer (%) | Baseline 2: 88 / 85. Baseline 3: 235 / 112. Baseline 4: 185 / 108. Progressive 2 col: 491 / 115 |
| Atari, 12 target games | Progressive networks give positive transfer in 8 of 12, with "only two cases of negative transfer". Baseline 3 gives positive transfer in 5 of 12 |
| Pong to H-flip (Figure 5a) | The new column reuses both convolutional layers, but the fully connected layer "must be largely re-learned" |
| Pong to Zoom (Figure 5a) | The new column reuses low-level vision and learns new mid-level features |
| One-column Pong net, conv 1 (Figure 5b) | "Only one feature map is effectively used" of 8. The authors read it as a spatio-temporal filter with a large temporal DC component |
| Atari, 72 three-column nets (Figure 7b) | The most negative transfer comes with full reliance on the old columns' convolutional layers. Positive transfer sits at a "sweet spot" between old and new features |
| APS against AFS on Pong (Figure 9) | The two per-layer sensitivities are "highly similar" |
| AFS spectra, Atari, layer 2 (Figure 10) | With more columns, the net relies on a smaller share of the old columns' feature maps. The newest column's features matter less |

The text gives no intervals, and no numbers for Figures 4 to 10 except those above.

## Where the claims go beyond the evidence

- **"Immune to forgetting" is true by construction, and it has a cost the claim leaves out.** The old columns are frozen, so their outputs can't change. But to use the right column at test time, the agent must know the task label. The authors state this limit themselves, in section 2.
- **Selection of the best runs.** Each result is the mean of the top 3 of 25 jobs, ranked by the same measure that the tables give. The text gives no spread across the 25 jobs.
- **Small gaps with no test.** On Atari, baseline 3 has a mean of 133% and the 2-column network 132%. The median gap is 110% against 112%. The text calls the 3-column result a convincing win at 140% against 133%, with no interval.
- **Means driven by outliers.** On Labyrinth, the 2-column network's mean is 491% but its median is 115%. The authors note that the mean is more sensitive to outliers, and read that as a sign of better transfer when transfer is possible.
- **"Positive transfer increases with the number of columns" rests on small changes.** The Atari medians are 112%, 111% and 116% for 2, 3 and 4 columns.
- **"Significantly" with no statistics.** Section 5.2 says progressive networks "significantly outperform the baselines for particular game pairs". The text gives no test.
- **A single-unit reading with no test of it.** The Pong claim that one feature map does the work rests on its AFS value and on activation maps from four example states. Nothing checks it against other inputs.
- **Inconsistencies within the source.** Section 2 cites "Equation 2" for the progressive layer, which is Equation 1, and "Appendix 2", which is Appendix B. Section 5.1 cites "section 3 of the Appendix", which is Appendix C. Appendix D calls baseline 1 a column trained on the source game Seaquest, but section 5.1 defines it as trained on the target task. Figure 5b's Pong net has 8 feature maps in conv 1, but Appendix C gives 12 for the Atari net. The text doesn't say whether the Pong net differs.
- **Credit: the authors state the growth limit themselves.** Parameters grow quadratically with the number of columns. Appendix B reads the sparse AFS spectra as a sign that pruning could contain that growth, and doesn't claim that it did so.

## Use in this repo

- **The guarantee holds only when old units receive no input from new capacity.** In a progressive network, a frozen column's activations never change, so any explanation of one of its units stays true. A transformer is different. An adapter or LoRA at one layer writes into the residual stream, and every later layer reads it. So a frozen base doesn't keep the inputs of later neurons fixed. A VERIFIED explanation of a neuron below the change still holds exactly. One above the change has to be tested again.
- **What the repo measured under full fine-tuning.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED explanations again after instruction tuning. 38 of 43 keep on Qwen2.5-1.5B (88%), at a median activation correlation of 0.983. 252 of 311 keep on Qwen2.5-0.5B (81%), at a median of 0.974. On Qwen2.5-0.5B by depth, 104 of 112 keep in the first third. 63 of 92 keep in the middle third, and 85 of 107 in the last. The paper's Pong analysis also reuses early vision more than later layers. But that is one RL net and a different measure, so it is no evidence for the repo's pattern.
- **Units keep changing while the task stays solved.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) found that Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. At step 16,000 their median correlation with the final form is 0.62, and adjacent checkpoints correlate at a median of 0.78 to 0.82. Progressive networks avoid such drift only by freezing a unit and never training it again. Nothing in the paper says how far units drift when they are not frozen.
- **AFS is close to the repo's per-pole effect, not a new tool.** AFS scores each unit by a first-order sensitivity of the output. [`docs/analyses/pole-effects.md`](../analyses/pole-effects.md) scores each pole by a first-order estimate of mean ablation's effect on the loss. Its estimate ranks 60 checked poles at a Spearman of 0.97 on GPT-2 small and 0.85 on Qwen2.5-0.5B. The paper's APS check is coarser: whole layers, and "highly similar" with no number.
- **Few units carry the load, in both places.** The paper's AFS spectra say a growing network relies on a small share of its old feature maps. On Qwen2.5-0.5B, `pole-effects.md` finds the estimate's top 1% of poles raise the loss 12 times as much as a uniform pole. These are different models and measures, so this is a match in kind only.
- **Superposition is not addressed.** The lateral connections let a new column read old features, but the paper never asks whether one unit serves several tasks. The repo's superposition records (`openinterpretability/explain_groups.py`) and the [complementary-learning-systems](complementary-learning-systems.md) page cover that kind of interference. This paper has nothing to add there.
- **A question for the maintainer.** Should a weight fix for #4781 be judged by how many VERIFIED explanations above the changed layer still hold? A frozen base keeps only the ones below it exactly. That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **No follow-up issue.** The repo has no fine-tuning or weight-edit code ([`fixing-models.md`](fixing-models.md)), and the paper's measure is already covered by `pole-effects`.
