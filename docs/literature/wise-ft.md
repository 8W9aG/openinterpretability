# Fine-Tuning Zero-Shot Models (WiSE-FT)

**Citation.** Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong and Ludwig Schmidt. The affiliations are the University of Washington, OpenAI, Columbia University, Google Research Brain Team, the Allen Institute for AI and the Toyota Research Institute. The HTML lost some of the footnote symbols. So the text that we read maps only Kim (OpenAI) and Kornblith, Roelofs and Gontijo-Lopes (Google) to their affiliations. The exact title is `Robust fine-tuning of zero-shot models`. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) [cs.CV, cs.LG]. Version 1 is of 4 September 2021, version 2 of 25 February 2022, and version 3, the one we read, of 21 June 2022. The arXiv comment gives the venue as CVPR 2022. The licence is arXiv's non-exclusive distribution licence. Code is at [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft). The funders are the NSF (IIS 1652052, IIS 17303166), DARPA (N66001-19-2-4031, W911NF-15-1-0543) and the Allen Institute for AI.

**Status.** Used. `papers/weight-average-keep/` takes its interpolation from this paper. That paper moves Qwen2.5-1.5B along the line from its base model to its instruct model and tests the explanations of single neurons at each step. The reference in that paper matches this source: the authors, the title, CVPR 2022 and the arXiv ID.

**Read.** On 2026-11-28 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read the source's own text for the abstract, Sections 1 to 7 and the acknowledgements. It also read Tables 1, 2 and 9 to 13, and Appendices C.7, C.8 and D.1 to D.3. Appendices A, B and C.1 to C.6 came back only as a summary. It did not read Appendices D.4, E or F, or Tables 3 to 8. Quotes were capped at 125 characters. We didn't see the figures or the PDF.

## Summary

**Claim.** Fine-tuning a zero-shot model such as CLIP raises its accuracy on the target distribution but can lower it under distribution shift. An element-wise weighted average of the zero-shot weights and the fine-tuned weights fixes this. Compared with standard fine-tuning, it "provides large accuracy improvements under distribution shift, while preserving high accuracy on the target distribution", at no extra cost.

**Method.** WiSE-FT fine-tunes the zero-shot model on the target distribution, then combines the two models as

$$
f\big(x, (1 - \alpha)\,\theta_0 + \alpha\,\theta_1\big), \quad \alpha \in [0, 1],
$$

where $\theta_0$ are the zero-shot weights and $\theta_1$ the fine-tuned ones. The paper recommends $\alpha$ = 0.5 when nothing is known about the target shift, and also reports the best $\alpha$ for each column. Fine-tuning is end-to-end, with the final layer started from CLIP's zero-shot classifier, or of a linear classifier alone. The main model is CLIP ViT-L/14 at 336 pixels, and others are ALIGN, BASIC and a ViT-H/14 trained on JFT-300M. The reference distribution is ImageNet, with five shifts: ImageNetV2, ImageNet-R, ImageNet-Sketch, ObjectNet and ImageNet-A. Six more shifts and seven transfer datasets test the method further. The metric is top-1 accuracy.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Zero-shot CLIP | ViT-L/14@336px (Table 1) | ImageNet 76.2, mean of the five shifts 73.3 |
| Standard end-to-end fine-tune | Same | ImageNet 86.2, shifts 68.6 |
| WiSE-FT, end-to-end, $\alpha$ = 0.5 | Same | ImageNet 86.8, shifts 76.9 |
| Gain under shift over prior work | Abstract | 4 to 6 points, with 1.6 points more on ImageNet |
| Gain under shift over the zero-shot model | Introduction | 1 to 9 points |
| Six further shifts | Against the fine-tuned model | +3.5, +6.2, +1.7, +2.1, +9.0 and +23.2 points, with at most 0.3 points lost on the reference |
| Seven transfer datasets | End-to-end, $\alpha$ = 0.5 against fine-tuning (Table 2) | +0.8 to +3.3 points. For example, DTD 81.9 → 84.6 |
| JFT ViT-H/14 | $\alpha$ = 0.8 | +2.2 points on the shifts, ImageNet within 0.2 points |

**Where the claims go beyond the evidence.**

- **Image classification only.** The authors say so: "our investigation here is limited to image classification", with detection and NLP left as future work.
- **The best $\alpha$ is chosen on the test data.** The "optimal $\alpha$" rows are tuned on each evaluation column. The fixed $\alpha$ = 0.5 rows are not, and they carry the headline.
- **Some rows don't add up.** In Table 1, the per-column values of two "optimal $\alpha$" rows don't average to the stated means. Table 13 gives a different mean for one of them.
- **No seeds.** The tables give no intervals or seed counts. The figures give Clopper–Pearson intervals for some points only.
- **Why it works is open.** The paper says that a linear path joins the two models, and that accuracy stays high along it. It leaves "the exact shape of the loss landscape" as an open problem.

## Use in this repo

- **The citing paper's claims, checked.** `papers/weight-average-keep/` said that WiSE-FT "keeps much of the zero-shot model's robustness". The source claims more. At $\alpha$ = 0.5 the average is more accurate under shift than both models: 76.9, against 73.3 for the zero-shot model and 68.6 for the fine-tune. It also keeps the fine-tune's ImageNet accuracy. The PR that adds this page rewords that sentence and the two related ones to match. Its other claims match. WiSE-FT averages the two models' weights and moves along the line between them by $\alpha$. It measures accuracy and robustness, not units.
- **What the repo's paper adds.** WiSE-FT shows that a model halfway along the line behaves well. `papers/weight-average-keep/` asks whether the explanations of single neurons also survive there. On Qwen2.5-1.5B the answer is mostly no. The full instruct change breaks 11 of 80 explanations. The average at $\alpha$ = 0.5 breaks 7, and it brings back only 5 of the 11.
- **No units in the source.** The paper measures accuracy, prediction diversity and CKA between the models. It looks at no neuron or feature. So it can't say what the average does inside the model, which is the repo's question.
- **The setting differs.** WiSE-FT interpolates between a zero-shot image model and a fine-tune on one task. The repo's paper interpolates between a base and an instruct language model, where the change is a broad post-training. The method carries over, but the source's robustness result doesn't have to.
