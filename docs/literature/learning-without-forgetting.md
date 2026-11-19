# Learning without Forgetting

**Citation.** Zhizhong Li and Derek Hoiem (Department of Computer Science, University of Illinois at Urbana-Champaign). "Learning without Forgetting". [arXiv:1606.09282](https://arxiv.org/abs/1606.09282), version 3 of 14 February 2017 (version 1 of 29 June 2016, version 2 of 7 August 2016). The conference version is in the *European Conference on Computer Vision* (ECCV) 2016, pages 614–629, which the paper cites as its own reference [10]. The arXiv comment says that version 3 is "updated with journal version". The issue names that journal as TPAMI 2018, but the text we read does not name it. The paper gives no code or data link. Funding: NSF Awards 14-46765 and 10-53768, and ONR MURI N000014-16-1-2007.

One mismatch matters for a reader. Version 3 is not the ECCV paper. The authors say it adds baselines, replaces Places2 with Places365-standard, and uses a stricter search for hyperparameters "which slightly changed our results". So numbers from the ECCV version can differ from the numbers on this page.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it, and the page adds evidence to the open issue #5118. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 3) and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the extracted text in the source's own words. That is the abstract, sections 1 to 5, the references, Appendix A (tracking with MD-Net) and Appendix B (the class splits). We read Tables 1, 2 and 3 as text. We did not see the figures as images. Figures 4, 5 and 7 are curves, so this page gives only their captions and what the text says about them. We did not read the ECCV version or the TPAMI version.

## Summary

**Claim.** A trained network can learn a new task from the new task's data alone, and keep most of its skill on the old tasks. Before training, it records its own outputs for the old tasks on the new images. During training, a distillation loss keeps those outputs close to the recorded ones. On the new task the method often does better than fine-tuning, and on the old task it does much better.

**Method.**

- **Parameters.** A network holds shared parameters θs, old-task output parameters θo and new-task output parameters θn. In the main experiments only the last layer (fc8) is task-specific.
- **The loss.** The method first records Yo, the old-task outputs of the original network on each new-task image. Training then minimises λo · Lold(Yo, Ŷo) + Lnew(Yn, Ŷn) + R. Lnew is the usual cross-entropy on the new labels. Lold is the knowledge distillation loss of Hinton et al., at temperature T = 2. Here R is a weight decay of 0.0005, and λo is 1 in most experiments.
- **Training.** A warm-up step first trains θn with θs and θo frozen. Then all parameters train together. The learning rate is 0.1 to 0.02 times the original rate.
- **Models and data.** AlexNet in most experiments, and 16-layer VGG in a few. The old task is ImageNet (ILSVRC 2012) or Places365-standard. The new tasks are PASCAL VOC 2012 (5,717 training images), CUB birds (5,994) and MIT indoor scenes (5,360). MNIST is a fourth new task, which the authors expect to fail.
- **Baselines.** Fine-tuning, fine-tuning of the fully connected layers only ("fine-tune FC"), and feature extraction. Less Forgetting Learning (LFL) and joint training, which uses the old data, are the others. Joint training is "an upper-bound" for the method. Ablations test an L2 penalty on weights, a lower learning rate, more task-specific layers, network expansion and other output losses.
- **Statistics.** Each experiment runs three times, and the tables report the mean. The metric is mean average precision for VOC and accuracy for all other tasks. For ImageNet, Places365 and VOC the numbers are on validation sets. Epochs and learning rates were chosen on a held-out set, by new-task performance only.

**Findings, as the source reports them.** The numbers are from Tables 1 to 3 and the text. Table 1 gives LwF's score and each other method's difference from it, in points. A row marked *derived* is our arithmetic on the table, not a number in the text.

| Setting | Result |
| --- | --- |
| AlexNet, 7 task pairs (Table 1(a)), new task | LwF beats fine-tuning on 4 of 7 pairs, by 0.3 to 1.7 points. Fine-tuning is higher on Places365→VOC (+0.1) and Places365→CUB (+1.0), and level on ImageNet→MNIST (0.0) |
| AlexNet, old task, fine-tuning against LwF | Fine-tuning is lower on all 7 pairs, by 0.9 to 4.6 points |
| AlexNet, old task, feature extraction against LwF | Feature extraction keeps the original old-task score, and is higher than LwF on all 7 pairs, by 0.8 to 7.3 points |
| AlexNet, joint training against LwF | Old task: joint training is higher on all 7 pairs, by 0.3 to 7.2. New task: within 1.5 points on all pairs |
| Places365→CUB, old-task loss (text) | 8.4% for fine-tuning, 3.8% for LwF and 1.5% for joint training |
| ImageNet→MNIST (Table 1(a)) | LwF keeps 49.8 on ImageNet. Feature extraction keeps 7.3 points more, so LwF loses about 7.3 points (*derived*) |
| Places365→VOC, test sets (Table 1(b)) | LwF 50.6 old, 73.7 new. Fine-tuning −2.1 old, +0.1 new |
| VGG (Table 1(c)) | Fine-tuning loses 9.9 (ImageNet→CUB) and 4.1 (ImageNet→Scenes) points of old task against LwF. Joint training is higher on both tasks of both pairs |
| No warm-up step (Table 2(b)) | ImageNet→CUB old task: fine-tuning falls from 50.9 to 42.5, and LwF from 54.7 to 53.5 |
| Fine-tuning at 10% of the θs learning rate (Table 2(a)) | Old task 52.2, 54.8 and 49.3 on three pairs, against 54.7, 55.9 and 50.6 for LwF. The new task is lower too |
| L2 penalty on weights (Figure 7, text) | LwF "outperforms this baseline". The penalty gives "a result between feature extraction" and fine-tuning |
| Other output losses (Figure 7, text) | Distillation "slightly outperforms" L1, L2 and cross-entropy, "although the advantage is not large" |
| Three new tasks in sequence (Figure 4, text) | LwF "underperforms joint training on the old task after more tasks are added" |
| Tracking, VOT 2015 (Table 3) | Expected average overlap 0.373 for MD-Net and 0.383 with LwF, 15 runs. The difference is not significant (p = 0.70, Student's t-test) |

## Where the claims go beyond the evidence

- **The "unexpected" gain on the new task is small.** On the new task, LwF beats fine-tuning by 0.3 to 1.7 points on 4 of 7 AlexNet pairs. Each number is a mean of three runs, and the tables give no spread or test. The authors themselves say that one or two more epochs can make the score fluctuate. The advice to replace fine-tuning with LwF "as the standard approach" rests on these differences.
- **The method needs related data, as the authors say.** The old-task loss is largest where the new images are far from the old ones: CUB from Places365, and MNIST from ImageNet. The authors state this, and they expected MNIST to fail. So the claim holds for related tasks, not in general.
- **Model selection favoured the new task.** The authors say they chose epochs and learning rates by new-task performance only, so their "selected hyperparameter favors the new task more". Most reported scores are on validation sets, and only one experiment reports test sets.
- **Image classification only.** All main results are image classifiers with a large pretraining set. The authors list detection, segmentation and "problems outside of computer vision" as future work. The one other use, tracking, gives no significant effect.
- **Small inconsistencies.** The text says LwF beats fine-tuning on the new task except on ImageNet→MNIST and Places365→CUB. Table 1(a) also shows fine-tuning 0.1 higher on Places365→VOC. The caption of Figure 4 says "Places2→VOC", but the experiment uses Places365. The affiliation note spells the second author as "Hoeim".

The authors give credit to the concurrent A-LTM method, which they call "nearly identical in method" with other experiments and conclusions. They state that joint training is the upper bound, and they report where LwF loses to it.

## Use in this repo

- **Nothing in the repo uses this method today.** A search of the code, docs, papers and skills finds no use of LwF or of a distillation loss in training. The synthesis in [`fixing-models.md`](fixing-models.md) notes that the repo "has no fine-tuning or weight-edit code".
- **LwF is the link that the pseudorehearsal page flagged, and it holds.** [`rehearsal-and-pseudorehearsal.md`](rehearsal-and-pseudorehearsal.md) (Robins, 1995) reads a pseudoitem as an input and the network's own output for it. LwF records the network's own outputs too, so it is the same idea with a distillation loss. The difference is the inputs. Robins samples inputs from the old items' distribution, but LwF reuses the new task's images. LwF's own results show the cost: unrelated images (MNIST) keep the old task poorly. The authors' future work, a store of unlabelled images "representative" of old tasks, moves LwF back toward Robins.
- **For a language model, the inputs should be real text from the old distribution.** The analogue of Yo is the model's next-token distributions before a change, on a set of prompts. LwF's MNIST and CUB results say those prompts should be like the text whose behaviour must keep. A set of only the #4781 prompts is narrow, like MNIST for ImageNet. A sample of `corpus.jsonl` is nearer to the pretraining text. This is our inference, and the paper does not test text.
- **Outputs, not weights, are what LwF keeps.** The L2 penalty on weights did worse than LwF, and the authors argue that "many small parameter changes could cause big changes in the outputs". [`elastic-weight-consolidation.md`](elastic-weight-consolidation.md) and [`synaptic-intelligence.md`](synaptic-intelligence.md) protect weights by importance instead. So the three #4782 methods sit at two levels, and none of them constrains a single unit.
- **A kept output need not keep a VERIFIED explanation.** LwF constrains θs only through the old-task outputs. Hidden units can change while those outputs stay close. [`papers/neuron-development/`](../../papers/neuron-development/README.md) shows such change in Pythia-70m. By step 16,000 the loss is within 1% of its final value. Yet the 86 VERIFIED poles correlate with their final form at a median of only 0.62. An output-matching loss would permit that kind of movement. This is our inference. The paper never looks at a hidden unit.
- **Output effect is a second predictor for #5118.** #5118 asks whether a pole's diagonal Fisher predicts whether its explanation keeps after instruction tuning. LwF's argument suggests a second, output-level predictor: how much ablation of the pole changes the next-token distribution. Each record already stores this KL, in `lookahead.kl_divergence` (`openinterpretability/harness/lookahead.py`). On Qwen2.5-0.5B, 252 of 311 poles keep and 59 fail ([`docs/analyses/base-instruct.md`](../analyses/base-instruct.md)). The test would compare the KL of the two groups. We propose it as a comment on #5118, not as a new issue.
- **Instruction tuning kept most poles without any such loss.** In [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md), 38 of 43 Qwen2.5-1.5B poles keep, at a median activation correlation of 0.983. On Qwen2.5-0.5B, 252 of 311 keep, at a median of 0.974. So a forgetting penalty is not needed for most poles in that change. Whether a narrow fix would behave the same way is not known.
- **The warm-up result does not carry over directly.** Training the new output layer first protected fine-tuning's old task (42.5 to 50.9 on ImageNet→CUB). A fix of the " days" error adds no new output layer, so the step has no clear analogue.
- **A question for the maintainer.** Should a weight fix for #4781 add a distillation loss to the model's own outputs, on real-text prompts and the right-copy prompts of `docs/analyses/hallucination-shortlist.jsonl`? That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **No single-unit content.** The paper measures task scores only. It says nothing about single neurons, explanations, probes or random baselines for them, so it has no bearing on our review protocol or our verdicts.
