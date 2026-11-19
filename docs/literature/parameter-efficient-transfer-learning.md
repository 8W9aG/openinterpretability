# Parameter-Efficient Transfer Learning for NLP

**Citation.** Neil Houlsby, Andrei Giurgiu, Stanisław Jastrzębski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan and Sylvain Gelly. Giurgiu and Jastrzębski contributed equally. Jastrzębski is at Jagiellonian University, and the other authors are at Google Research. "Parameter-Efficient Transfer Learning for NLP". [arXiv:1902.00751](https://arxiv.org/abs/1902.00751), v1 of 2 February 2019 and v2 of 13 June 2019. Venue: Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, Long Beach. The paper gives code at [google-research/adapter-bert](https://github.com/google-research/adapter-bert). We did not open it. The text gives no funding statement. It thanks four people for comments. The training ran on Google Cloud TPUs, and the AutoML baseline is an earlier method by two of the authors (Wong et al., 2018).

**Status.** Not applicable. The repo trains no adapters and has no fine-tuning code, and the paper measures task accuracy, not units. One idea carries over: an adapter keeps every base weight fixed, so it splits weight drift from input drift.

**Read.** On 2026-11-19 we downloaded the arXiv v2 PDF (13 pages) and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the extracted text in the source's own words. That is the abstract, sections 1 to 4, the acknowledgements and the references. It is also the supplementary material: section A with Tables 3 to 7, and section B on the learning rate. Tables 1 and 2 came through as text. We did not see the figures as images. Figures 1 and 3 to 7 came through only as captions and axis labels, and Figure 2 as its caption and box labels. So we have the Figure 6 heatmap values only where the text states them. We did not compare v2 with v1.

## Summary

**Claim.** Small bottleneck modules, inserted into a frozen pre-trained Transformer and trained per task, come close to full fine-tuning. On GLUE they score within 0.4 of full fine-tuning, with 3.6% new parameters per task. Since the shared weights stay frozen, a new task never changes an old one.

**Method.**

- **The adapter.** Each adapter projects the d-dimensional features down to m dimensions, applies a nonlinearity, and projects back to d. It has an internal skip connection. It adds 2md + d + m parameters per layer, and in practice 0.5% to 8% of the model.
- **Placement.** Each Transformer layer has two adapters: one after the attention block's output projection, one after the feed-forward block. Each acts on the sub-layer's output before the residual add, and its output then goes into layer normalization.
- **What trains.** The adapters, new layer-normalization parameters for each task, and the classification head. The original weights stay frozen.
- **Near-identity start.** Adapter weights start from a zero-mean Gaussian with standard deviation 10⁻², truncated at two standard deviations. With near-zero projections, the module starts close to the identity.
- **Models.** BERT-Large (24 layers, 330M parameters) for GLUE. BERT-Base (12 layers) for the other tasks and the analyses.
- **Tasks.** GLUE without WNLI, 17 further public text-classification tasks, and SQuAD v1.1.
- **Training.** Adam, with the learning rate raised linearly over the first 10% of steps, then decayed linearly to zero. Batch size 32. A small sweep of learning rate, epochs and adapter size, scored on the validation set.
- **Baselines.** Full fine-tuning. "Variable fine-tuning", which tunes only the top n layers. Tuning only the layer-normalization parameters. A "no BERT" AutoML baseline that searches over 10,000 models per task.
- **Analyses.** Removal of trained adapters from spans of layers, with no retraining. A sweep of the initial scale from 10⁻⁷ to 1. A comparison across adapter sizes.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| GLUE test, BERT-Large, Table 1 | Full fine-tuning 80.4, with 9.0× parameters in total and 100% trained per task. Adapters of size 8 to 256: 80.0, with 1.3× and 3.6%. Adapters fixed at size 64: 79.6, with 1.2× and 2.1% |
| GLUE, chosen adapter size | 256 for MNLI, 8 for RTE, the smallest dataset |
| 17 further tasks, BERT-Base, Table 2, mean test accuracy | No BERT 72.7. Fine-tuning 73.7 (17×). Variable fine-tuning 74.0 (9.9×, 52.9% per task). Adapters 73.3 (1.19×, 1.14% per task) |
| SMS spam collection, Table 2 | Fine-tuning 99.3 ± 0.2. Adapters 95.1 ± 2.2 |
| MNLI matched, BERT-Base, validation | Top layer only: about 9M parameters, 77.8% ± 0.1%. Adapters of size 64: about 2M parameters, 83.7% ± 0.1%. Full fine-tuning: 84.4% ± 0.02% |
| Layer normalization only, BERT-Base | 40k parameters. About 3.5% lower on CoLA and 4% lower on MNLI |
| SQuAD v1.1, validation F1 | Adapters of size 64 (2% parameters): 90.4. Fine-tuning: 90.7. Adapters of size 2 (0.1%): 89.9 |
| Removal of one layer's adapters, MNLI and CoLA, size 64 | The largest drop is 2% |
| Removal of all adapters | 37% on MNLI and 69% on CoLA, the majority-class scores |
| Removal of adapters in layers 0 to 4, MNLI | "barely affects performance" |
| Mean validation accuracy over eight tasks, by adapter size | Size 8: 86.2%. Size 64: 85.8%. Size 256: 85.7% |
| Initial scale | Stable for standard deviations below 10⁻². Larger starts do worse, more on CoLA |

The paper also tried batch or layer normalization inside the adapter, more layers, tanh and other activations, adapters only in attention, and parallel adapters. In each case it saw performance similar to the plain bottleneck.

## Where the claims go beyond the evidence

- **"Perfect memory" is true by construction, and only with the adapter removed.** Section 1 says the model "has perfect memory of previous tasks". That holds because each task keeps its own adapters and layer-normalization parameters. It says nothing about the frozen network's own activations while an adapter is in place.
- **One model family and mostly classification.** Every run uses BERT, an encoder. The tasks are text classification and one extractive QA set. Nothing tests a generative decoder.
- **Best of five seeds.** On GLUE, the authors run each adapter setting five times "due to training instability", and keep the best on validation. The text doesn't say whether the BERT-Large row had the same selection.
- **No statistical test.** Table 2 gives a standard error, but not the number of seeds. Figures 4, 5 and 7 use three seeds. No gap has an interval or a test.
- **"Within 0.4%" is 0.4 points.** Table 1 shows 80.0 against 80.4 on the GLUE score, which is a difference in points, not a percentage.
- **Epochs chosen by eye.** For the 17 further tasks, the epochs come "manually, from inspection of the validation set learning curves".
- **Adapters lose clearly on some tasks.** On SMS spam, adapters reach 95.1 ± 2.2, against 99.3 ± 0.2 for fine-tuning. The average hides this.
- **The layer claim rests on removal without retraining.** The text says adapters "automatically prioritize higher layers". The evidence is removal of trained adapters from BERT-Base, on two tasks, at one size. It shows what the trained model leans on, not that lower adapters can't help.
- **Inconsistencies within the source.** Sections 1 and 3 say 3% task-specific parameters, and the abstract and Table 1 say 3.6%. Section 3.3 says variable fine-tuning trains 52% of the network, and Table 2 says 52.9%. Section 2.1 says "two serial adapters after each of these sub-layers". The Figure 2 caption says each layer has the adapter twice, one after each sub-layer.
- **Credit: the authors show their own limits.** They report the layer-normalization baseline that fails, the tasks where adapters lose, and the initial scales where training fails.

## Use in this repo

- **An adapter fixes every MLP weight, but not every MLP input.** This point is our inference from section 2.1. Each adapter adds its output to the residual stream, and every later block reads that stream. The per-task layer-normalization parameters also change what each block reads. So with an adapter in place, no MLP neuron above the first adapter sees its old input. Its explanation has to be tested again, although its own weights stay the same.
- **With the adapter removed, every explanation holds exactly.** The base weights stay as they were, so the base model runs unchanged when the adapter is off. Every VERIFIED explanation then holds by construction. [`progressive-neural-networks.md`](progressive-neural-networks.md) makes the same point for frozen columns. There the guarantee holds with the new column in place too, because old columns receive no input from new ones.
- **A bottleneck adapter can't be merged.** This is our inference, and the paper does not discuss merging. The adapter has a nonlinearity between its two projections, so it is not a change to any base matrix. #4782's question about a merged adapter applies to LoRA (#4854), which adds a low-rank change to the weights. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) and [`text-to-lora.md`](text-to-lora.md) review that design.
- **An adapter splits weight drift from input drift.** #5133 asks whether the weight change at a pole predicts whether its explanation keeps. #5118 asks the same of diagonal Fisher. Under adapter tuning, the pole's own weight change is zero. So any explanation that fails must fail through a changed input. #5125 fine-tunes in stages and could run an adapter arm beside full fine-tuning. The gap between the arms would give the share of failures due to the pole's own weights. That needs fine-tuning code that the repo lacks ([`fixing-models.md`](fixing-models.md)).
- **What the repo measured under full fine-tuning.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED explanations again after instruction tuning. 38 of 43 keep on Qwen2.5-1.5B (88%), at a median activation correlation of 0.983. 252 of 311 keep on Qwen2.5-0.5B (81%), at a median of 0.974. The analysis doesn't say how Qwen tuned the instruct models. A run with no change to the MLP weights would show how much of the 19% loss on Qwen2.5-0.5B comes from inputs alone.
- **The depth pattern doesn't match in a simple way.** The paper finds that adapters in layers 0 to 4 of BERT-Base matter little on MNLI. On Qwen2.5-0.5B, the first third keeps most: 104 of 112. But the middle third keeps least, 63 of 92, and the last third keeps 85 of 107. The models, the tuning and the measure all differ, so this is a match in kind at most.
- **Freezing stops the drift that the repo sees in training.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) finds that Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. Their median correlation with the final form is 0.62 at step 16,000. Adjacent checkpoints correlate at a median of 0.78 to 0.82. Full fine-tuning lets such drift go on, and an adapter stops it for the base weights only.
- **The adapter's hidden units are new units.** Each adapter has m hidden units after a nonlinearity, of the same kind as an MLP neuron. The repo's pole audit could explain them in principle. But the registry explains released models, and none of them has adapters. So this opens no issue.
- **The paper avoids interference between tasks, and doesn't study it.** Each task has its own adapters, so tasks never share parameters that train. The paper has nothing on superposition or on interference between units. [`complementary-learning-systems.md`](complementary-learning-systems.md) and `openinterpretability/explain_groups.py` cover that.
- **A question for the maintainer.** Should a weight fix for #4781 report the keep share of VERIFIED explanations both with the fix in place and with it removed? That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **No follow-up issue.** The adapter arm fits inside #5125, so it belongs on that issue as a comment, not in a new issue.
