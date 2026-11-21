# The Platonic Representation Hypothesis

**Citation.** Minyoung Huh, Brian Cheung, Tongzhou Wang and Phillip Isola, of MIT, with equal contribution. "The Platonic Representation Hypothesis". [arXiv:2405.07987](https://arxiv.org/abs/2405.07987), version 1 of 13 May 2024, last revised as version 5 on 25 July 2024. It appeared at ICML 2024 as a position paper, "Position: The Platonic Representation Hypothesis". That is *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, pages 20617–20642 ([PMLR](https://proceedings.mlr.press/v235/huh24a.html)). Code is at [github.com/minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep), and the project page is [phillipi.github.io/prh](https://phillipi.github.io/prh/). Funders include a Packard Fellowship, a Sloan Research Fellowship, the MIT-IBM Watson AI Lab, ONR, NSF and DARPA.

**Mismatch.** Our reference called it a preprint, but it appeared at ICML 2024 with "Position:" in its title. This review fixes the reference in `papers/shared-concept-space`. We didn't read the PMLR PDF, so the match rests on the authors and the title. The HTML of Section 2.2 says "7878 vision models", which is likely a doubled 78.

**Status.** Planned. `papers/shared-concept-space` names the paper's mutual nearest-neighbour alignment as a next step, and #3379 holds the question for the registry.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the arXiv abstract page, the PMLR listing and the HTML of version 5. It had the abstract, Sections 1 to 6 and the acknowledgements in the source's own words. Appendices A to F came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** "Neural networks, trained with different objectives on different data and modalities, are converging to a shared statistical model of reality in their representation spaces." The authors argue that scale, many tasks and simplicity bias drive this.

**Method.**

- **The metric.** Each model gives a kernel, the inner products of its embeddings of the same inputs. The "mutual nearest-neighbor metric … measures the mean intersection of the k-nearest neighbor sets induced by two kernels, K1 and K2, normalized by k". CKA and SVCCA are named as other measures.
- **Vision.** 78 vision models, compared on Places-365 images and scored by how many VTAB tasks they solve.
- **Language and vision.** Language models and vision models compared on paired Wikipedia images and captions (WIT).
- **Competence.** Language models' alignment to a vision model set against Hellaswag and GSM8K scores.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Vision models (Figure 2) | "Models that solve more VTAB tasks tend to be more aligned with each other" |
| Language against vision (Figure 3) | "a linear relationship between language-vision alignment and language modeling score" |
| Best language–vision alignment (Section 6) | The score "only reaches a score of 0.16", where the maximum is 1 |
| Alignment against task scores (Figure 4) | Linear for Hellaswag, and an "'emergence'-esque trend" for GSM8K |

## Where the claims go beyond the evidence

- **A position, not a test.** The paper argues for a direction of travel. The highest alignment it measures between language and vision is 0.16 of a possible 1.
- **Whole representations only.** The paper compares kernels over inputs. It cites "Rosetta Neurons" in vision models as related work, but makes no claim of its own about single units.
- **Limits stated by the authors.** Their argument "only holds for intelligences that are optimized to perform well on many tasks". It also assumes "a bijective, deterministic function" from the world to observations, which "will not hold true" for lossy or stochastic ones.
- **No seeds or intervals** appear in the sections we read.

## Use in this repo

- **Our citations check out, but not the venue.** `papers/shared-concept-space` quotes "to a shared statistical model of reality", which is in the abstract. It describes the metric as "the average overlap of each input's nearest neighbours", which matches Section 2. The four authors and 2024 match. The reference now names ICML 2024 and the "Position:" title, fixed in place because the paper is a preprint.
- **The registry asks a narrower question.** The paper compares how two models arrange the same inputs. `openinterpretability/concept_regions.py` compares where two models' explained poles fall on one map, by histogram intersection of their region shares. Our reading: a high region overlap is consistent with the hypothesis but doesn't test it. It reads units, not inputs, and only the studied ones.
- **Sisters are the unit-level case.** `docs/analyses/sister-neurons.md` finds a sister in Qwen2.5-1.5B for a fifth of Qwen2.5-0.5B's VERIFIED poles, and fewer across families. Our reading: that is closer to the "Rosetta Neurons" the paper cites than to its own kernel claim. It also fits the paper's own heading, "Not all representations are presently converging".
- **The next step is planned.** #3379 asks whether the hypothesis holds for the registry's models once they are well understood. The paper's metric, run on the four configured models, would give a measure to set beside the map's region overlap.
- **No question for the maintainer.**
- **Related pages.** [`convergent-learning.md`](convergent-learning.md) and [`universal-neurons.md`](universal-neurons.md) review unit-level convergence across seeds.
