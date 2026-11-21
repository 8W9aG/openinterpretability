# Interpretability Illusions in the Generalization of Simplified Models

**Citation.** Dan Friedman (Princeton University), Andrew Lampinen (Google DeepMind), Lucas Dixon (Google Research), Danqi Chen (Princeton University) and Asma Ghandeharioun (Google Research). "Interpretability Illusions in the Generalization of Simplified Models". *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235, pages 14035–14059. [PMLR](https://proceedings.mlr.press/v235/friedman24a.html). The preprint is [arXiv:2312.03656](https://arxiv.org/abs/2312.03656), version 1 of 6 December 2023, revised as version 2 on 5 June 2024. The paper names no release of its own code.

**Mismatch.** Section 2 names the main language "Dyck-(20,10)", and Appendix A.1 names it "Dyck-(10,20)". Both describe 20 bracket types with a maximum depth of 10.

**Status.** Not applicable. No code uses it. `papers/syntax-positive-controls` cites it as background on bracket depth in transformers.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the PMLR page, the arXiv abstract page and the HTML of version 2. In three fetches it had Sections 1 to 8 and Appendices A and B in the source's own words. It saw the figure captions but not the figures.

## Summary

**Claim.** Interpretability often studies a simplified proxy of a model, such as a low-rank projection or a clustering of its attention keys and queries. A proxy that is faithful on the training distribution can differ from the model on new data. So an explanation drawn from it may misjudge how the model generalises.

**Method.**

- **Simplifications.** Project keys and queries onto their top k singular vectors, or replace them with k-means cluster centres. Or replace softmax attention with one-hot attention.
- **Models.** All are trained from scratch. Two-layer transformers with one head per layer on balanced-bracket (Dyck) languages. Character-level transformers with four layers on Java functions from CodeSearchNet. Small models on SCAN in an appendix.
- **Shifts.** Unseen bracket structures, deeper nesting, and for code, deeper Java and unseen languages (JavaScript, Go, PHP).
- **Metric.** How often the simplified model makes the same prediction as the original.
- **Statistics.** Three seeds, with 95% intervals for Dyck and a paired t-test for code.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Dyck, seen structures | An SVD to "as few as four dimensions" keeps "nearly 100% prediction similarity" |
| Dyck, unseen structures | The original model "generalizes almost perfectly". The simplified model "deviates considerably" |
| Dyck, deeper nesting | The original models reach about 75% accuracy. One-hot attention "slightly out-performs the original model" |
| Code, one head simplified (Section 5) | Prediction similarity of 75% in domain, and "as low as 61%" on unseen languages, "statistically significant with p≤0.005" |
| SCAN, Add jump (Appendix B.7) | "no discernible generalization gap" |

## Where the claims go beyond the evidence

- **Small models trained from scratch.** The authors say the study "focused on small-scale models trained on a limited range of tasks". No pretrained language model is tested.
- **One component at a time.** The authors note that simplifying a whole circuit "might yield more dramatic distribution shifts", which they didn't test.
- **Gaps don't always appear.** The SCAN Add jump split and the deeper-Java split show no clear gap. The authors say so.
- **Both directions.** The simplified model can underrate the original's generalisation, and on Dyck depth it can overrate it. So the paper shows a risk, not a fixed bias.

## Use in this repo

- **Our citation checks out.** `papers/syntax-positive-controls` says the paper studies bracket depth "in small transformers trained on parenthesis languages, and in character-level models trained on code, not in GPT-2". Sections 2, 5 and Appendix B.6 match, and no pretrained model appears. The five authors, 2024 and ICML 2024 match the PMLR page. That paper is published, and needs no addendum.
- **The lesson matches the protocol's held-out probes.** An explanation that fits the scan's top activations can fail on new text. The reviewer writes hard negatives and positive holdouts in new contexts, so they test it off the scan's distribution. [`docs/PROTOCOL.md`](../PROTOCOL.md) gives the rule, in step 5.
- **Its unit isn't ours.** The paper simplifies attention keys and queries. It doesn't study neurons, and the repo doesn't simplify heads this way. Our reading: its warning applies to any explanation built from one dataset, including a neuron's.
- **No question for the maintainer.**
- **Related pages.** [`learning-transformer-programs.md`](learning-transformer-programs.md) reviews the first author's earlier work, whose programs also degrade on longer inputs. [`who-cares-about-brackets.md`](who-cares-about-brackets.md) reviews a study of brackets in GPT-2 small.
