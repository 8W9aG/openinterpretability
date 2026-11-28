# Unified Factored Representations Resist Catastrophic Forgetting

**Citation.** Joel Lehman and Kenneth O. Stanley. "Unified Factored Representations Resist Catastrophic Forgetting". [arXiv:2611.11020](https://arxiv.org/abs/2611.11020) [cs.LG, cs.AI], version 1 of 7 November 2026, the only version. Preprint, no venue given. The licence is arXiv's non-exclusive distribution licence. The paper names no affiliations, links no code and gives no acknowledgements or funding in the text that we read. It builds on Kumar, Clune, Lehman and Stanley (2025), *Questioning representational optimism in deep learning* (arXiv:2505.11581), which #7339 will review.

**Status.** Planned. #7340 asks whether this repo's layers with a higher pass rate keep more of their explanations under fine-tuning. That carries the paper's idea to units that the review has tested. No code, skill or protocol here depends on the paper.

**Read.** On 2026-11-28 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). The source's own text covered the abstract, Sections 1 to 6, most of Section 7, and Appendices F to J with their tables. Appendices A to E and the references came back only as a summary. The subagent also saved the PDF, and we extracted its text with `pdftotext`. From it we checked the abstract, the header and Appendix D in the source's own words, including the CLIP numbers below. We saw no figure images, so values shown only in plots are unknown to us.

## Summary

**Claim.** The *representational forgetting hypothesis*: "networks with unified, factored representation (UFR) can resist catastrophic forgetting without any modification to the underlying fine-tuning algorithm or network architecture". In a UFR, each regularity is encoded once and reused, and independent factors are kept apart. SGD tends to produce *fractured entangled representations* (FER), which forget more. A change on part of the input then disrupts features that the whole input shares.

**Method.**

- **Cyclic fine-tuning.** A network is fine-tuned on part of its input range toward a variant of its target, then back to the target, many times. Forgetting is the loss on the held-out part of the input after each cycle. The variant is *compatible* when it changes the target along a factor that the UFR network already has.
- **The illustrative domain.** 1D regression of $\lvert x \rvert$ and two- and three-level variants, trained on $x \ge 0$ and measured on $x < 0$. The UFR networks are 4-layer ReLU MLPs of 20 units a layer, built by hand to capture the symmetry. The baseline is the same shape trained by SGD. A sawtooth wave is the incompatible control.
- **The image domain.** Picbreeder networks, CPPNs that people evolved into images of a skull, a butterfly and an apple, with 147 to 918 trainable weights. The paper calls them UFR because their units often represent coherent parts of the image. The variants come from sweeping single weights of the CPPN. The SGD baseline is the best of a search over depth, width, optimiser and activation, an 8-layer GELU MLP of 29,120 parameters. Skull (X), an X painted on one side, is the incompatible control. Each image is scored by mean squared error and by CLIP distance on the held-out region.
- **Inducing UFR in SGD.** A symmetry loss and a representation-distillation loss, both used only in the first training. Distilled GELU students of the Picbreeder networks are a third treatment.
- **Statistics.** 20 seeds per treatment, mean ± 1 SEM, and Wilcoxon or Mann–Whitney tests. We saw no correction for the many comparisons.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| UFR against SGD, CLIP distance | Four compatible Picbreeder tasks, cycle 30 (Appendix D) | 3.4–7.4× lower for UFR ($p \le 10^{-7}$). By validation loss, 2.9–12× |
| The incompatible control | Skull (X), cycle 30 | Forgetting is high for all treatments. UFR forgets slightly more than SGD by validation loss ($p$ = 0.002) and less by CLIP distance ($p = 10^{-4}$) |
| Symmetry loss against SGD | Butterfly (fly), 10 cycles, $\lambda$ from 0.003 to 0.3 (Appendix I) | Final forgetting falls by 11.4% to 46.8% as $\lambda$ grows. At 0.3 it falls by 46.8% ($p < 10^{-4}$) |
| The wrong symmetry | Same task, horizontal axis, 20 seeds | Forgetting rises, by up to 22.8% |
| Representation distillation | Four compatible Picbreeder tasks | Forgetting is significantly lower than SGD's in three of the four |
| Distilled GELU students against SGD | Appendix H | 2.2× less forgetting on Skull (mouth), 9.6× on Butterfly (fly), 12× on Butterfly (color), and 1.16× on Apple (stem), $p$ = 0.41 |
| Free capacity added to the UFR network | 1, 5 or 10 units a layer, 10 cycles | "Forgetting increases monotonically with free capacity added in every task" |
| Growing a UFR network to a harder target | 2-level to 3-level, 20 seeds (Appendix J) | Only 6 of 20 seeds fit it |

**Where the claims go beyond the evidence.**

- **The UFR networks are not trained.** They are built by hand or evolved by people. The authors say that "in general a method to construct UFR networks is lacking". So the result compares two very different ways of making a network, not two representations that one method could produce.
- **The networks are tiny, and the tasks few.** The largest network is a 453,823-parameter student. There are three images and one 1D function family. Nothing tests a language model.
- **Compatibility is chosen to fit.** UFR helps only when the variant changes the target along a factor that the network already has. The image variants come from sweeping its own weights. On the incompatible tasks UFR does not help, and the paper reports this.
- **The 47% is the best of a sweep.** It is the largest of five values of $\lambda$, on the same runs, after 10 cycles rather than the main 30.
- **Many tests, no correction.** Each table reports several $p$ values, with no correction for the number of comparisons.

The authors give most of these limits themselves, and they call the setting "idealized".

## Use in this repo

- **The registry's units are the closest thing here to a factored representation.** A neuron whose explanation passes the review encodes one regularity that the review can test. If the paper's hypothesis carries over, a layer with more such units should lose fewer of them under fine-tuning. That step is ours, not the paper's, and passing explanations are not the paper's notion of UFR.
- **Existing data hints at it, weakly.** In `papers/sequential-tuning/` Table 3, the middle third of Qwen2.5-0.5B has the fewest passing poles (101, against 133 and 128). It also keeps the fewest after three narrow fine-tunes (55–56%, against 72–86%). That is three points, and that paper offers other readings: the early layers' explanations are more often about single tokens. #7340 tests the relation across all 24 layers, with the token-set label as a covariate, on data that the repo already has.
- **The forgetting here is not cyclic.** The repo's fine-tunes run once, on the whole input, with no return to the first task. The paper's mechanism needs a change on part of the input that spreads to the rest. So the comparison holds only loosely.
- **Weight sweeps that map to one factor.** The image variants come from sweeping one CPPN weight, which changes one part of the image. In a trained language model, a ROME edit is a rank-one change. But in pythia-70m it moved 25 of 870 pole readings above its layer (`papers/rome-keep/`). So one weight change doesn't map to one factor there.
- **Other forgetting pages.** `catastrophic-forgetting-connectionist.md` and `anatomy-of-catastrophic-forgetting.md` explain forgetting by overlap in the representation. This paper says that the right kind of sharing, one unit for each regularity, reduces it.
