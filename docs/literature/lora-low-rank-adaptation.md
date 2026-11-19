# LoRA: Low-Rank Adaptation of Large Language Models

**Citation.** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang and Weizhu Chen, all of Microsoft Corporation. Hu and Shen made an equal contribution. Li also gives an address at Carnegie Mellon University. "LoRA: Low-Rank Adaptation of Large Language Models". [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), version 2 of 16 October 2021. Version 1 is of 17 June 2021. A note says that version 2 adds better baselines, GLUE experiments and more on adapter latency. Venue: ICLR 2022, as the issue states. The arXiv v2 text names no venue, so this page did not check the venue in the source. Code, the `loralib` package and checkpoints for RoBERTa, DeBERTa and GPT-2 are at [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA). The authors release no GPT-3 adapters. The text names no funding. All experiments ran on NVIDIA Tesla V100 GPUs, and the latency study on one Quadro RTX8000.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it in its synthesis page. #5133 reads the same kind of Δ that the paper's section 7 reads. The page proposes one follow-up that reads weights only. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v2 PDF (26 pages) and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own words the abstract, sections 1 to 8 and Appendices A to H. We read Tables 1 to 8 and 10 to 18 as text. We did not see the figures as images. So this page gives Figures 2 to 8 only through their captions and the numbers that the text states. The text of Table 9 (RoBERTa hyperparameters) and of section D.1 came through only in part. We did not read the references in full, the ICLR version or its OpenReview discussion.

## Summary

**Claim.** The weight change that adapts a pre-trained language model to a task has a low "intrinsic rank". So one can freeze each weight matrix W0 and train only a low-rank update ΔW = BA. On RoBERTa, DeBERTa, GPT-2 and GPT-3 175B, this matches or beats full fine-tuning with far fewer trainable parameters. The update merges into W0, so it adds no inference latency.

**Method.**

- **The update.** For a frozen W0 of size d × k, LoRA trains B (d × r) and A (r × k), with r much smaller than d and k. The forward pass is h = W0x + BAx, and BAx is scaled by α/r. A starts as random Gaussian and B as zero, so ΔW is zero at the start. The authors set α to the first r they try and do not tune it.
- **Merge.** At deployment one can store W = W0 + BA. To switch task, one subtracts BA and adds another B′A′.
- **Which weights.** All experiments adapt attention weights only, mostly Wq and Wv. The MLP modules are frozen. Section 4.2 leaves "adapting the MLP layers, LayerNorm layers, and biases to a future work."
- **Models and tasks.** RoBERTa base and large and DeBERTa XXL on GLUE. GPT-2 medium and large on E2E NLG, WebNLG and DART. GPT-3 175B on WikiSQL, MultiNLI-matched and SAMSum.
- **Baselines.** Full fine-tuning (FT), fine-tuning of the last two layers (FTTop2), bias-only (BitFit), prefix-embedding and prefix-layer tuning, and four adapter designs. The authors reuse many baseline numbers from prior papers, and mark them with an asterisk.
- **Statistics.** GPT-2 rows give the mean over 3 seeds, and each run takes its best epoch. GPT-3 rows give one "typical standard deviation" per task, not one per entry. GPT-3 rows report "the best validation performance from each run."
- **Section 7, the low-rank update.** On GPT-3 175B, the authors compare ΔW subspaces across ranks and seeds. The measure is a normalized subspace similarity φ in [0, 1], from the Grassmann distance. They compare ΔW with W by projecting W onto ΔW's top-r singular directions. They contrast that with W's own top-r directions and with a random matrix. They call ‖ΔW‖F divided by that projection's norm the "amplification factor".

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| GPT-3 175B, training memory and checkpoint (section 4.2) | VRAM falls from 1.2TB to 350GB. With r = 4 on Wq and Wv, the checkpoint falls from 350GB to 35MB |
| GPT-3 175B, throughput (footnote 5) | 32.5 tokens/s per V100 for FT, 43.1 for LoRA |
| GPT-2 medium latency, batch 1, sequence 128 (Table 1) | FT or LoRA 19.8±2.7 ms. AdapterL +20.7%, AdapterH +30.3% |
| GLUE mean, RoBERTa base (Table 2) | FT 86.4 (125.0M parameters, prior work). LoRA 87.2 (0.3M) |
| GLUE mean, DeBERTa XXL (Table 2) | FT 91.1 (1500.0M, prior work). LoRA 91.3 (4.7M) |
| E2E NLG BLEU, GPT-2 medium (Table 3) | FT 68.2 (354.92M, prior work). LoRA 70.4±.1 (0.35M) |
| GPT-3 175B (Table 4) | FT: WikiSQL 73.8, MNLI-m 89.5, SAMSum 52.0/28.0/44.5. LoRA 4.7M: 73.4, 91.7, 53.8/29.8/45.9. LoRA 37.7M: 74.0, 91.6, 53.4/29.2/45.1 |
| GPT-3, seed spread (Table 4 caption) | About ±0.5% on WikiSQL, ±0.1% on MNLI-m, ±0.2/±0.2/±0.1 on SAMSum |
| GPT-3, rank with Wq and Wv adapted (Table 6) | r = 1: WikiSQL 73.4, MNLI 91.3. r = 64: 73.5, 91.4 |
| GPT-3, rank with Wq only (Table 6) | r = 1: WikiSQL 68.8. r = 4: 70.5 |
| GPT-3, 18M budget across weight types (Table 5) | Wq alone 70.4 on WikiSQL. Wq and Wv 73.7. All four 73.7 |
| GPT-3, ranks 8 and 64, layer 48 (section 7.2) | The two share "a subspace of dimension 1 with normalized similarity > 0.5" in ΔWq and ΔWv |
| GPT-3 layer 48, ΔWq against Wq, r = 4 (Table 7) | ‖Uᵀ Wq Vᵀ‖F is 0.32 for ΔWq's directions, 21.67 for Wq's own top directions, 0.02 for random. ‖Wq‖F = 61.95, ‖ΔWq‖F = 6.91 |
| Same, r = 64 (Table 7) | 1.90 for ΔWq's directions, 37.71 for Wq's, 0.33 for random. ‖ΔWq‖F = 3.57 |
| Amplification factor (section 7.3, H.4) | About 21.5 for r = 4 (6.91/0.32), and "around 2" for r = 64 |
| Overlap of ΔWq with Wq's top directions (H.3) | The similarity between ΔW's top 4 directions and the top 10% of W's "barely exceeds 0.2" |
| GPT-2 medium, rank sweep on E2E (Table 18) | Validation loss is lowest (1.16) from r = 16. BLEU peaks at r = 4 (70.38) |
| GPT-3, MNLI with 100 examples (Table 16) | FT 60.2, PrefixEmbed 37.6, LoRA 63.8 |
| GPT-3, few-shot against FT (Table 8) | MNLI-m 40.6 against 89.5. RTE 69.0 against 85.4 |

From Table 7, the authors conclude three things. ΔW correlates with W more than a random matrix does. ΔW does not repeat W's top singular directions. And ΔW amplifies "features" that W holds but does not emphasize. In their words, the update "potentially amplifies the important features for specific downstream tasks". These are features "learned but not emphasized" in pre-training. H.4 adds that one should expect a different set of directions to be amplified for each task.

## Where the claims go beyond the evidence

- **The ΔW analysis rests on one model, one layer and one matrix.** Table 7 is GPT-3 layer 48, Wq only, one run, no seeds. H.1 extends the subspace plots (Figures 3 and 4) to layers 1, 32, 64 and 96, but not the Table 7 numbers. So "ΔW amplifies directions that W does not emphasize" is shown for one matrix of one model.
- **"Features" here are singular directions, not units.** The paper's word "features" means singular vectors of a weight matrix. It never looks at neurons, heads or any explained unit. The interpretation that these are "task-specific" features is not tested, for example by ablation of those directions.
- **The amplification factor mixes in the α/r scale.** This point is our inference. ‖ΔWq‖F is smaller at r = 64 (3.57) than at r = 4 (6.91), and ΔW is scaled by α/r. The authors read the lower factor at r = 64 as evidence for low intrinsic rank. Part of the difference may come from the scale and from the fixed α, which the paper does not separate.
- **No measure of what adaptation overwrites.** No experiment tests the merged model on its pre-training distribution or on other tasks. The adapter can be removed, so the design does not need it. But the paper gives no evidence on forgetting after a merge.
- **Selection on evaluation data.** GPT-2 runs take "the best epoch", and GPT-3 runs report "the best validation performance from each run". The GPT-3 learning rate is tuned per method and dataset on the same validation sets. So the GPT-3 margins over FT, often under 2 points, carry this selection.
- **Few intervals, and baselines from other setups.** GPT-3 rows give one typical standard deviation per task. Many FT and adapter rows come from prior papers. The authors say some baselines "might only appear in certain experiments".
- **Rank claims for two GPT-3 tasks.** "r = 1 suffices" rests on WikiSQL and MNLI. The authors state the limit themselves. Footnote 6 says a small r should not work for every task, for example a task in a new language. H.2 also says the link between model size and best rank is "still an open question".
- **Small inconsistencies.** Section 7.2 uses the right-singular matrices of A, but Appendix G speaks of left singular matrices. The projection is written Uᵀ W Vᵀ in section 7.3 and Uᵀ W V in H.4. Section 7.2 lists "Wc" where Wo is meant. None of these changes a result.

The authors include random baselines in Figure 4 and Table 7, a null for the subspace claims. They also state the main limit of the merge. Once A and B are inside W, one pass cannot easily batch inputs for different tasks.

## Use in this repo

- **The repo has no LoRA or fine-tuning code.** `git grep` finds no `peft`, `loralib` or adapter code in `openinterpretability/` or `scripts/`. [`fixing-models.md`](fixing-models.md) notes the same. None of the paper's models is a registry model. So its runs cannot be repeated here as they are.
- **A merge does not change the question.** This point is our inference from equation 3. W0 + BA computes the same function as W0 with the adapter beside it, up to rounding. So "does a VERIFIED unit keep its explanation after a merge?" is the same as "after this update". The merge only changes where the weights are stored.
- **The paper's adapters leave the registry's units in place.** Every experiment freezes the MLP modules. The registry's units are MLP neuron poles, so their own Wgate, Wup and Wdown do not move. Only the residual stream they read changes, through Wq and Wv. [`text-to-lora.md`](text-to-lora.md) makes the same point for T2L's adapters. A weight-only reading at a pole, as #5133 plans, would see zero change under such an adapter.
- **The base-instruct numbers are for a wider change.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests each VERIFIED pole of Qwen2.5 on its instruct model at the same index. On Qwen2.5-1.5B, 38 of 43 hypotheses keep (88%, 95% CI 76–95%), at median activation correlation 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%, 76–85%), at median 0.974. The analysis does not record how the instruct models were tuned. That tuning also changed MLP weights, so it is not a test of an attention-only adapter.
- **Section 7.3 gives a weight-level test that #5133 does not have.** #5133 reads the norm of Δ = instruct − base in each pole's own rows and columns, and Δ's rank per layer. The paper adds a different question: does Δ amplify directions that W already has but does not emphasize? For an MLP pole, that becomes whether Δ's top singular directions overlap the pole's input row or output column. If failing poles sit in the amplified directions, the paper's account would predict which explanations break. The follow-up below proposes this.
- **The amplified directions need not align with neurons.** This point is our inference. A rank-4 ΔWq changes every row of Wq inside one shared four-dimensional subspace. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) and [`intrinsic-dimensionality-fine-tuning.md`](intrinsic-dimensionality-fine-tuning.md) make the same point: a low-rank change is not a change to few neurons. It bears on superposition directions in `openinterpretability/explain_groups.py` more than on one neuron.
- **The subspace measure suits sister neurons less than activity does.** The paper compares two runs by the Grassmann similarity of their ΔW subspaces. [`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md) matches poles across models by activity over `corpus.jsonl`. Activity matching does not depend on a basis. The paper's Figure 4 shows two seeds share only some directions, so a weight-space match of units across runs would be unreliable.
- **Rank per layer connects to the depth pattern, as a question.** On Qwen2.5-0.5B, the middle third of poles keeps least: 63 of 92, against 104 of 112 and 85 of 107. The paper's Table 7 is for one middle layer only, so it says nothing on depth. #5133 reads Δ's rank per layer for this reason.
- **Pre-training shows the same gap between score and weights.** In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's 86 VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. That is after the loss settles. Adjacent checkpoints correlate at a median of 0.78 to 0.82. The LoRA paper also shows good task scores from a tiny update. Neither result shows what the update does to units.
- **Evidence for the fix campaign, not a design.** A fix for the #4781 " days" error would be a narrow update. The paper shows that a rank-1 attention update can move a task score a lot on GPT-3. It measures nothing on other skills. [`complementary-learning-systems.md`](complementary-learning-systems.md) and [`fixing-models.md`](fixing-models.md) explain why a narrow update needs a measure of what it damages.
- **Question for the maintainer.** Should #4782 treat an attention-only adapter as a separate case from a full fine-tune? With such an adapter, a pole's own weights do not change, and only its input moves. That would test whether a pole keeps when its input distribution shifts. The page records this and opens no issue.
- **Follow-up: the paper's ΔW-against-W test on the Qwen2.5 pairs.** For each Qwen2.5 pair, take Δ = instruct − base for each MLP matrix. Repeat Table 7: project W onto Δ's top-r singular directions, onto W's own top-r directions, and onto random directions. Then, for each VERIFIED pole, read the share of its Wgate and Wup rows and its Wdown column that lies in Δ's top-r directions. Test that share against the keep and fail rows, against a null of shuffled poles. It reads weights only, trains nothing, and can share rows with #5133 and #5154.
- **Leave LoRA variants to their own pages.** #4855 (AdaLoRA), #4856 (LoRA+), #4857 (QLoRA), #4858 (DoRA) and #4859 (LoRA against full fine-tuning) each have an issue. This page does not restate them.
