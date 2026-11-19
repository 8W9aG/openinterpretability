# Using Fast Weights to Deblur Old Memories

**Citation.** Geoffrey E. Hinton and David C. Plaut (Computer Science Department, Carnegie-Mellon University). "Using Fast Weights to Deblur Old Memories". *Proceedings of the 9th Annual Conference of the Cognitive Science Society*, 1987, pages 177–186. Hillsdale, NJ: Lawrence Erlbaum Associates. [Index page](https://www.cnbc.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html) and [scanned PDF](https://www.cnbc.cmu.edu/~plaut/papers/pdf/HintonPlaut87CogSciConf.fastWeights.pdf) on David Plaut's CMU pages. The paper has no code or data, and it names no funding. One mismatch: the index page says that the PDF has 12 pages, but the PDF has 10, numbered 177 to 186. The page range in the citation also gives 10 pages. The index page warns that some figures are missing from the scan.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it. The paper gives no method to read a unit from its weights, so ROTATE (#3446) and #4471 can't use it directly. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we read the scanned PDF from the CMU page. The plain-text copy that the index page links returned "403 Forbidden". The PDF is page images with no text layer. So we rendered each page, ran OCR with `tesseract`, and read the OCR text of all 10 pages. A model (Claude) did this reading. To check the OCR, we viewed these page regions as images:

- page 177, with the abstract and the introduction
- the part of page 180 with the simulation settings
- page 181, with Figure 2 and its caption
- the part of page 185 with equation 2 and the 0.943 ratio

The OCR damaged equations 1 and 3, on pages 184 and 185. We did not view those equations as images, so we report their meaning from the text around them only. We saw Figures 1, 3 and 4 only at low resolution. We read Figure 2's curves, but the axis has no labels between 0 and 0.5. So this page gives no number from it.

## Summary

**Claim.** Give each connection two weights. A slow weight holds long-term knowledge, and a fast weight learns quickly and decays toward zero. The effective weight is the sum of the two. Suppose that later learning blurs a set of old associations. Then retraining on a few of them restores all of them, and the fast weights let this happen without permanent damage to the new associations. The authors explain the transfer with geometry and a derivation for a linear network.

**Method.**

- **Learning rule.** Back-propagation on a layered feed-forward network. The fast and slow weights share the same error derivative. They differ only in their learning rate ε and momentum α, and in a decay. Each fast weight shrinks by a fraction *h* after each weight change.
- **Simulation.** The task maps random binary 10-bit inputs to random binary 10-bit outputs. There are 100 associations, with the inputs drawn without replacement from all 2¹⁰ vectors. The network has 10 input units, 100 hidden units and 10 output units, with biases on the hidden and output units. The weights change once after each sweep through the training set.
- **Schedule.** First, the network learns the 100 associations. Then it learns 5 new random associations with no rehearsal of the old ones. This continues until the slow weights hold the new knowledge. Last, it retrains on a subset of the old 100, and the authors compare the retrained subset with the rest.
- **Analysis.** A network with no hidden units and one linear output unit knows two associations, A and B. Then the authors add independent Gaussian noise to each weight. The measure of transfer is the improvement on B from retraining on A, divided by the improvement from retraining on B itself. For small steps of fixed size, this ratio is the cosine of the angle between the two gradients.
- **Baselines.** The comparison is between retrained and unretrained associations. The paper reports no seeds, no error bars and no run without fast weights.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| First training: 100 associations, 1,300 sweeps, ε = .02, α = .9 | "the network knew the associations perfectly", with the fast weights near zero |
| Fast-weight decay during first training | 1% after each weight update |
| Interference: 5 new associations | 400 sweeps, until the new knowledge was in the slow weights |
| Retraining on 50 of the 100 (Figure 2, left) | Early in retraining, the unretrained associations improve "very nearly as" much as the retrained ones |
| Retraining on 10 of the 100 (Figure 2, right) | The result "held even if only 10%" were retrained |
| Linear network, expected cosine of the two gradients (equation 2) | n_ab² / (n_a n_b), where n_a and n_b count the 1s in each input and n_ab counts the shared 1s |
| 0/1 inputs, each bit 1 with probability 0.5 | Expected cosine 0.25 |
| Linear network, 100 associations, noise, retrain on 50 | Expected ratio of improvement, unretrained to retrained, at the start: 50 × .25 / (1 + 49 × .25) = .943. The authors say it "agrees well" with their simulations, and give no simulated value |
| Inputs of 1, 0 and −1 | Expected transfer between one pair falls in proportion to 1/n, for input dimension n. If the number of associations grows with n, the overall ratio doesn't depend on n |
| During retraining | The transfer falls as retraining continues (geometric argument, Figure 4) |

## Where the claims go beyond the evidence

- **The simulation doesn't separate fast weights from slow ones.** The paper gives no fast-weight learning rate or momentum. It doesn't report how the fast and slow weights moved during retraining. The Conclusions say that the analysis "applies just as well if there are no fast weights". So the paper shows the transfer, but it argues the part that fast weights play and doesn't measure it.
- **No measurement that the new associations come back.** The abstract says that the fast weights decay and the new knowledge returns. No figure or number shows the error on the 5 new associations, during retraining or after the decay.
- **"All" associations are not fully restored.** The abstract says that *all* the old associations can be deblurred. In Figure 2, the unretrained curve stays above the retrained curve after 20 sweeps. The gap is larger when the authors retrain 10 than when they retrain 50.
- **One run, with no spread.** Figure 2 shows one curve for each condition. The paper reports no seeds or intervals. "Agrees well with simulations" has no number.
- **The analysis is for a linear network with no hidden units.** The authors say that multilayer networks are "hard to analyze", and offer the linear case as a guide. The 10-100-10 simulation is the only test with hidden units.
- **Random associations only.** The tasks have no structure. The authors expect structure to make the transfer larger, but they don't test it.
- **Nobody tested the human prediction.** The authors state two problems with it themselves. Networks that learn long enough use different units for unrelated items. Also, response competition could hide the effect. They also say that they know of no well-controlled study.

## Use in this repo

- **No method to read units from weights.** The paper never reads a weight to say what a unit does. Its transfer measure is the cosine between two gradients. In equation 2, the expected value depends on the overlap of the two inputs only. So it needs the inputs, which come from forward passes in a deep model. It gives nothing to ROTATE (#3446) or to a check of an explanation without forward passes.
- **A reading of stored weights reads only the slow part.** In the paper's networks, the function is the slow weights plus a temporary overlay. #4839 will review the claim that linear attention is a fast-weight memory of this kind. If that claim holds, stored weights alone miss the part of the function that depends on context. The repo's own weight-only test is consistent with a weak signal. In [`weight-kurtosis.md`](../analyses/weight-kurtosis.md), ROTATE's kurtosis predicted VERIFIED at an AUC of 0.51 (0.45–0.57) for the output side. The input side scored 0.54 (0.48–0.60). Adding it to the predictor gave no gain. This paper doesn't test that question, so it is context, not evidence.
- **Relearning brings back knowledge that was not trained.** The paper gives a mechanism: shared weights move back toward the old solution, so untrained associations improve too. The synthesis [`fixing-models.md`](fixing-models.md) lists "relearning and in-context attacks" as a test that a fix removed a behaviour and did not hide it. Hinton and Plaut's result predicts that a small relearn can restore much more than it trains on. That supports the test, from an older and simpler setting.
- **Transfer and interference need shared units.** The paper's prediction holds only if the associations share units. The authors note that networks that learn long enough separate unrelated items. Sparse, clean neurons are the separated case. So a VERIFIED pole that serves one concept would show little transfer and little interference. A pole in a superposition (`explain_groups.py`) is the shared case. The paper doesn't measure this in any model.
- **Whether layer inputs are zero-mean matters.** The paper finds that transfer between random items fades with dimension when inputs have a mean of zero. With 0/1 inputs, it doesn't fade. In GPT-2 and Pythia, the MLP's GELU outputs are mostly non-negative. So their down-projections read inputs whose mean isn't zero. Our reading is that the 0/1 case may fit those weights better. Qwen's gated MLPs can give outputs of either sign. Nobody measured this here.
- **A decaying overlay is a reversible edit.** Fast weights are the oldest case in #4782's weight-space strand of a function written into weights for a short time. While the overlay is present, a VERIFIED explanation describes slow plus fast weights. After the decay, it describes the slow weights again. An unmerged LoRA adapter is a modern version that doesn't decay. Whether the campaign uses edits of this kind is the maintainer's question.
- **No single-unit content.** The paper studies whole networks and errors averaged over output units. Nothing in it bears on a pole's explanation or on our verdicts.
