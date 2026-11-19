# PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning

**Citation.** Arun Mallya and Svetlana Lazebnik (University of Illinois at Urbana-Champaign). "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning". [arXiv:1711.05769](https://arxiv.org/abs/1711.05769), v1 of 15 November 2017, v2 of 13 May 2018. Published at CVPR 2018, as the issue gives it. The arXiv v2 text names no venue, and we did not read the CVPR proceedings version. Code: [github.com/arunmallya/packnet](https://github.com/arunmallya/packnet). The text gives no separate data release, but every dataset is public. Funding: the National Science Foundation, grants 1563727 and 1718221. The acknowledgements thank Maxim Raginsky for the first idea and Greg Shakhnarovich for the name.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it, and this page proposes one follow-up that measures evidence. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v2 PDF and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the extracted text in the source's own words. That is the abstract, sections 1 to 6, Tables 1 to 6, the figure captions, the acknowledgements and the references. The paper has no appendix. We did not see the figures as images. For Figures 2 to 5 we have the values that the PDF prints on the plots, as text, but not the bars or error bars. We matched each value to its line or bar from the text layout, so the figure rows below carry that small risk. We did not run the code.

## Summary

**Claim.** One network can learn several tasks in sequence with no loss on the earlier tasks. After each task, weight-magnitude pruning frees part of the weights, and the next task trains only those. The weights of earlier tasks stay fixed, and a binary mask for each task selects its weights at test time.

**Method.**

- **Training.** The network first trains on task I. Then pruning sets a fixed share of the weights in each convolutional and fully connected layer to zero, by lowest absolute magnitude. The share is 50% or 75%, in one shot. The surviving weights train again briefly, and from then on they stay fixed. Task II trains the freed weights, and uses task I's fixed weights as well. Then pruning and training again follow in the same way, and so on.
- **What stays fixed.** Biases, and the batch normalisation gains, biases and running statistics, stay fixed after the first round of pruning. Each new task has its own output layer.
- **Inference.** For task K, the mask turns on the weights of tasks 1 to K. Every later task also uses a weight that task K uses first. So a mask needs at most log₂(N) bits for each parameter, for N tasks. One pass serves one task only, because a filter's response depends on which of its weights are on.
- **Models and data.** VGG-16 trained on ImageNet, then VGG-16 with batch normalisation, ResNet-50 and DenseNet-121. The added tasks are CUBS birds, Stanford Cars and Oxford Flowers (fine-grained classification), and Places365 (scenes).
- **Baselines.** *Classifier Only* trains a new last layer on fixed features. *Individual Networks* fine-tunes one full network for each task, the best case. *LwF* is the authors' own re-implementation of Learning without Forgetting. For Places365, a jointly trained network comes from the Places365 authors.
- **Runs.** The pruning results average 18 runs: 6 orders of the three fine-grained tasks, 3 runs each. LwF has 1 run for each order. Classifier Only and Individual Networks average 3 runs.

**Findings, as the source reports them.** All numbers are error rates in per cent, top-1 unless the row says top-5.

| Setting | Result |
| --- | --- |
| VGG-16 on ImageNet, after 50% and 75% pruning and training again (Table 2) | Top-1 rises from 28.42 to 29.33 and 30.87. Top-5 rises from 9.61 to 9.99 and 10.93 |
| Three tasks added to the 75%-pruned VGG-16, against Individual Networks (Table 2) | CUBS 24.95 against 22.57, Cars 15.75 against 13.97, Flowers 9.75 against 8.65 |
| The same tasks with LwF (Table 2) | CUBS 30.42, Cars 22.97, Flowers 15.21. ImageNet rises to 39.23 (top-5 16.94) |
| The same tasks with Classifier Only (Table 2) | CUBS 36.76, Cars 56.42, Flowers 20.50 |
| Storage (Table 2) | One PackNet model of 595 MB, against 4 individual networks of 2,173 MB. Classifier Only is 562 MB |
| Mask overhead on VGG-16, conv1_1 to fc7, 537 MB | About 17 MB for one added task and about 34 MB for three |
| Earlier tasks while later ones train (Figure 2) | With PackNet, the error on each earlier task stays the same. With LwF, ImageNet error goes from 28.42 to 31.36, 37.14 and 39.02 as tasks are added |
| Places365 added to 75%-pruned VGG-16 (Table 3) | 46.99 against 46.35 for an individual network. The jointly trained network has 33.49 on ImageNet, against 28.42 |
| Other networks, three tasks (Table 4) | ResNet-50: CUBS 21.13, Cars 13.75, Flowers 7.10, against 19.56, 12.99, 8.50. DenseNet-121: 21.84, 15.55, 7.71, against 19.72, 13.15, 8.02 |
| Order of tasks, Stanford Cars (Figure 3) | 16.00 when added first, 18.34 second, 19.91 third. Over the three datasets, about 3 points worse from first to third |
| Free parameters on VGG-16 | About 67 M free after the first 50% pruning. Task II uses 16.75 M, and task III about 13 M. About 37 M stay free for task IV |
| 90% one-shot pruning of a new task, then training again (Figure 4) | Cars 15.75 to 17.84, CUBS 24.13 to 24.72, Flowers 8.96 to 9.48 |
| Separate biases for each task (Table 5) | CUBS 25.62 against 25.72 shared, Cars 18.17 against 18.08, Flowers 10.11 against 10.09 |
| Only some layers train for a new task (Figure 5) | Cars: 54.63 with the classifier only, 34.61 with the fully connected layers too, 15.75 with all layers |
| Filter pruning in place of weight pruning (Table 6) | About 30% of filters could go (4,000 of 12,416), and 3,730 of those were in the fully connected layers. Cars 34.78 against 13.97 with weight pruning |

## Where the claims go beyond the evidence

- **No loss on old tasks is true by construction.** The old weights, biases and batch normalisation stay fixed, and the mask brings back each old network. So Figure 2's flat PackNet lines don't need a test. The cost is that the task must be known at test time, and one pass serves one task. The authors state this limit themselves.
- **The baselines differ in how many runs they had.** PackNet averages 18 runs and LwF has 6, one for each order. The text gives no spread for any table. Figure 3 has error bars over 6 runs, but we did not see their values.
- **The text doesn't show the rules came from held-out data.** The text names training and evaluation splits only. It doesn't say which split chose the pruning ratios, the epochs or the learning rates.
- **One kind of task.** All tasks are image classification, starting from an ImageNet network. The paper doesn't test tasks that share an output, tasks that conflict, or any language model.
- **The joint baseline is not like for like.** The jointly trained Places365 network came from other authors, and it trained for 60 to 90 epochs against PackNet's 10. The authors say so.
- **Wider wording than the tests.** Section 5.2 reads the small loss at 90% pruning as "effective transfer learning". That is an interpretation, and no experiment separates it from a task that needs few weights.
- **Credit: the authors give the filter-pruning result that limits their own idea.** Pruning whole filters would allow all tasks in one pass. They report that it could free only about 30% of filters, mostly in the top layers.

## Use in this repo

- **The repo already measures change without any freezing.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED explanations again after instruction tuning. 38 of 43 keep on Qwen2.5-1.5B (88%), with a median activation correlation of 0.983. 252 of 311 keep on Qwen2.5-0.5B (81%). PackNet would make that share 100% by construction, for the old task's mask only.
- **An explanation in PackNet describes a unit under one mask.** The paper says a filter's response changes with the weights that are on. So the same filter computes a different function for each task. [`papers/neuron-development/`](../../papers/neuron-development/README.md) makes the matching point in time. Pythia-70m's 86 VERIFIED explanations describe one checkpoint, with a median correlation of 0.62 to the final form at step 16,000. A record names a model and a revision, so it would also need to name the mask.
- **Sister neurons are trivial inside PackNet.** `open-interpretability sisters` pairs poles across models by activation correlation ([`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md)). 53 of 259 Qwen2.5-0.5B poles (20%) have a sister in Qwen2.5-1.5B. In PackNet, a unit keeps its position in each task, so the useful question is whether its function stays the same. The sisters search would not answer that question, because the position is already known.
- **Weight pruning and unit pruning give different answers.** Weight pruning freed 50% of VGG-16 for a rise of 0.91 points in ImageNet error. Pruning whole filters freed about 30%, nearly all in the fully connected layers. So in VGG-16 a task's spare capacity sits in weights spread across units, not in whole units. The repo explains whole units and sparse directions over them (`openinterpretability/explain_groups.py`). The paper tests no language model, so this is a caution, not evidence about our models.
- **A follow-up can test how much of a pole's weights its explanation needs.** PackNet's result suggests that many weights of a unit carry little. A test could zero the lowest-magnitude share of a VERIFIED pole's input weights, at 50%, 75% and 90%. Then it would run the record's own probes again, as `scripts/base_instruct.py` does for instruct models. That shows which explanations rest on a few weights. This page proposes that issue. It measures evidence only and saves no weights.
- **A question for the maintainer.** Should a weight fix for #4781 leave fixed the weights of VERIFIED units that the fix doesn't target? That is PackNet's rule applied to a fix. [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code. The choice belongs to the fix campaign, so this page records it as a question and opens no issue.
- **A word clash.** The paper calls a task's filter the "superposition" of the weights of all earlier tasks. That is a sum of masked weights. The repo's superposition records are sparse directions over a layer's neurons. The two meanings are not related.
- **No single-unit content.** The paper studies weights and task error. It says nothing about single neurons, explanations, probes or random baselines for them. Figure 1's filter is a drawing, not a measured unit.
