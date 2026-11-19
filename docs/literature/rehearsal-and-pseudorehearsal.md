# Catastrophic Forgetting, Rehearsal and Pseudorehearsal

**Citation.** Anthony Robins (Computer Science Department, University of Otago, Dunedin, New Zealand). "Catastrophic Forgetting, Rehearsal and Pseudorehearsal". *Connection Science*, 1995, volume 7, number 2, pages 123–146. [doi:10.1080/09540099550039318](https://doi.org/10.1080/09540099550039318). We read the author's pre-publication draft, from [CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.3078&rep=rep1&type=pdf). The draft's header asks readers to cite the published version. We did not compare the draft with the published text, so small changes in wording or figures are possible. The paper gives no code or data, and names no funding. It thanks research assistants: Andrew Gillies helped to develop sweep rehearsal, and Nick Mein made the Appendix A graphs. It says that Noel Sharkey's laboratory developed sweep rehearsal independently.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we read the 37-page draft as a PDF. The publisher gives no open copy, and the typeset.io and scispace.com copies refused the download. We extracted the text with `pdftotext` and read every page: the abstract, Sections 1 to 7, the references and Appendix A. The extraction lost the figure curves. So we rendered Figures 1, 2, 5 to 8, 12 and 13 as images and read the curves. The word "about" marks a value that is our reading of a plot. We did not view Figures 3, 4, 9 to 11, A1 or A2 as images. For those, we have the captions and the text only.

## Summary

**Claim.** In a back-propagation network, learning new items one at a time disrupts what it learned before. Rehearsing old items with each new item reduces the damage. "Sweep" rehearsal almost removes it. "Pseudorehearsal" keeps much of the benefit with no access to the old items. It rehearses on random inputs, with the network's own outputs for those inputs as targets.

**Method.**

- **Network.** A 32-16-32 back-propagation network, from Ratcliff (1990). Weights update after each pattern, with learning rate 0.3 and momentum 0.5. A population counts as learned when every output unit is within 6% of its target.
- **Items.** Random binary input and output vectors, each element 0 or 1 with equal chance. The main runs are heteroassociative, with targets independent of the inputs. Autoassociative runs, with target equal to input, are in Section 5 and Appendix A.
- **Schedule.** The network learns a base population of 20 items. Then it learns 10 intervening items one at a time, each to the criterion.
- **Measure.** "Goodness": the mean normalised dot product of target and output, rescaled so that 1 is a perfect match and 0 is chance. Each curve averages 50 replications, each with a new random population.
- **Regimes.** Each new item trains in a buffer with three previously learned items:
  - *recency*: the three most recent items, as in Ratcliff (1990).
  - *random*: three old items at random, trained to the criterion. Choosing the three most or least similar items gave the same results.
  - *sweep*: three old items drawn again at random every epoch, until the new item reaches the criterion. No old item is trained to the criterion.
  - *pseudorehearsal*: random or sweep, with "pseudoitems" in place of the old items. A pseudoitem is a new random input and the output that the network gives for it. The method builds a new pseudopopulation for each intervening item, with 8, 32 or 128 pseudoitems.
- **Appendix A.** Sweep rehearsal over 70 intervening items. A test population of 70 novel random items measures discriminability d′. That is the difference between the mean goodness of old and novel items, divided by the standard deviation of the novel items' goodness.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| No rehearsal, base populations of 1, 4 and 20 | After a single intervening item, the network's ability to reproduce the base population's outputs is "significantly disrupted" (Section 2.2). Groups of items resist more. Figure 1: after 10 items, about 0.33 for 20 items, about 0.25 for 4 and near 0 for 1 |
| Recency rehearsal | The base population falls to "a goodness of around 0.3 after 10 intervening items" (Section 2.3). Figure 2: about 0.39 for a base of 20 |
| Random rehearsal, base of 20 | The base population is still disrupted, at a better goodness than with recency. Figure 7: about 0.68 after 10 items |
| Sweep rehearsal, base of 20 | The base population stays at "a very high level of accuracy" and rises slightly over the 10 items. Figure 7: about 0.96 throughout |
| Sweep rehearsal, bases of 10 to 50, 20 intervening items | It performs well "over all conditions" (Figure 12). From the plot, every curve stays at about 0.9 or higher |
| Random pseudorehearsal, 8, 32 and 128 pseudoitems | "Only moderately effective". The pseudopopulation's size has no effect, since a buffer holds only three. Figure 5: about 0.45 after 10 items for all three sizes |
| Sweep pseudorehearsal, 8, 32 and 128 pseudoitems | More effective, and better with more pseudoitems. Figure 6: about 0.50, 0.60 and 0.70 after 10 items |
| Sweep pseudorehearsal, bases of 10 to 50, 20 intervening items | Performance "falls off with increasing base population size" (Figure 13). From the plot, about 0.6 for a base of 10 and about 0.44 for a base of 50, after 20 items |
| The buffer's share of old items | "at best 15%" of the previously learned items in the standard runs. Initial runs with larger buffers gave a higher goodness |
| Sweep rehearsal against training again from the start | Only a "slight advantage", in training speed, when all old items are available (Section 4.1) |
| Autoassociative sweep rehearsal, 70 intervening items | Goodness is "still approximately 0.9", but discriminability is low from the start and falls. The network passes through any input, so novel items score high too (Appendix A) |
| Serial position, all regimes (Figure 8) | Every regime shows a drop for the first intervening items. Most regimes improve for the last ones |

The author interprets these results with function approximation (Section 5.2). Training fits a function to the items. Without rehearsal, a new item can bend the function away from the old points. Pseudoitems are samples of the old function, so rehearsing them keeps its shape. The author predicts that pseudorehearsal works when the network fits a "compact" function that interpolates well, and fails when it fits a "noisy" one.

## Where the claims go beyond the evidence

- **One small network and random data.** Every result is from one 32-16-32 network on random binary vectors. The author states this limit in Section 6.3, and names tests on real data as "the obvious next step". The function approximation account is an argument, and the author says it has no formal analysis.
- **The account's main prediction is not tested.** The paper predicts that pseudorehearsal works for compact functions and fails for noisy ones. Its random populations have no structure to generalise from, so the experiments can't test this. The paper says the method "may well be capable of more effective performance" than shown, without data.
- **Sweep pseudorehearsal beats random rehearsal by very little.** Section 5.1 says that sweep pseudorehearsal outperforms the other regimes except sweep. In Figure 7 it ends at about 0.70 and random rehearsal at about 0.68. In Figure 8 the two curves overlap for most serial positions. No test or interval supports the order.
- **"Significant" has no test.** The paper uses "significantly" many times, but reports no statistical test. Each curve averages 50 populations, and the figures show no spread after the first point.
- **Fixed settings.** The buffer holds three old items, the base 20 items and the run 10 or 20 new items. Section 6.2 says that larger buffers help, from "initial" and "preliminary" experiments that the paper does not show.
- **Good scores on old items can hide a failure.** The author shows this in Appendix A, to his credit. An autoassociative network keeps a goodness of about 0.9 on old items, because it passes through any input. Only a test on novel items shows the loss of discrimination.

## Use in this repo

- **No rehearsal method in the repo.** A search for rehearsal, pseudoitems and pseudodata in the code, analyses, papers and skills found nothing. The repo has never trained a model, so nothing in it applies this method today.
- **Pseudorehearsal is self-distillation on sampled inputs.** A pseudoitem is an input and the network's own output for it. For a language model, the analogue is a set of prompts and the model's own next-token distributions before a change. Learning without Forgetting (#4814) appears to use a related idea, which its own review should check. Robins samples inputs from the same distribution as the training items. Random tokens are far from the text that a language model reads. So the analogue needs real-text inputs, such as `corpus.jsonl`, not random ones. This is our inference, not a claim of the paper.
- **The function approximation account fits a pole's explanation only in part.** Pseudoitems keep the old input-output map where they sample it. A VERIFIED pole's explanation is a claim about one unit, inside the map. Keeping the outputs on sampled prompts need not keep the same unit doing the same work. [`papers/neuron-development/`](../../papers/neuron-development/README.md) shows that units move while the loss holds still. At step 16,000 the loss is within 1% of its final value. But the 86 poles' median correlation with their final form is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. So output-level rehearsal is no evidence that an explanation survives. A check would have to run the pole's recorded probes again.
- **Appendix A supports the review's hard negatives.** Good scores on old items hid a network that answered every input the same way. The repo's review guards against the same failure. Two thirds of the 30 probes are hard negatives (`docs/PROTOCOL.md` §3). The `not_default_on` gate (§4) rejects a unit active on half its scan's tokens or more. Any check that an explanation survives a weight change should keep both, not only the positive holdouts.
- **Broad and shallow beat narrow and deep.** Sweep rehearsal touched many old items for one or a few updates each. Random rehearsal trained three old items fully, and was worse. [`complementary-learning-systems.md`](complementary-learning-systems.md) reached the same view from another side: interleave new items with the old ones. For a fix of the " days" error (#4781, #5015), the old items at risk are the prompts that copying answers right. Whether a fix rehearses them, or samples them as pseudoitems, is the maintainer's question. [`fixing-models.md`](fixing-models.md) has the editing-side tests.
- **Overlap sets the limit.** Appendix A ends with the claim that representational overlap in a finite network sets an upper limit on any rehearsal. That is the same trade-off as superposition, where many features share neurons (`openinterpretability/explain_groups.py`, the superposition records). The paper does not measure overlap, so this is a pointer for the other #4782 reviews, not evidence.
- **Second-hand evidence for an unread source.** The paper replicates Ratcliff (1990), whose page (#4807) still needs its own source. Its account of Ratcliff's recency rehearsal is second-hand evidence only.
- **No single-unit content.** The paper measures outputs only. It never looks at a hidden unit, so nothing in it bears directly on a pole's explanation or on our verdicts.
- **A question for the maintainer.** Should the fix campaign measure the VERIFIED poles' recorded probes before and after a change, in addition to the model's outputs? This paper and the neuron-development results suggest that outputs alone are not enough. We record the question and open no issue.
