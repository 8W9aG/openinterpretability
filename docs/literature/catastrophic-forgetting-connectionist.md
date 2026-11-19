# Catastrophic Forgetting in Connectionist Networks

**Citation.** Robert M. French (Quantitative Psychology and Cognitive Science Unit, Department of Psychology, University of Liège, Belgium). "Catastrophic forgetting in connectionist networks". *Trends in Cognitive Sciences*, April 1999, volume 3, number 4, pages 128–135. [doi:10.1016/S1364-6613(99)01294-2](https://doi.org/10.1016/S1364-6613(99)01294-2). It is a review article, so it has no code or data. Funding: research grant IUAP P4/19 from the Belgian government. Box 1 reports unpublished rat-timing work by André Ferrara, credited in the acknowledgements.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we read the PDF from the LEAD laboratory pages at the Université de Bourgogne, French's later lab. Its PDF title is the article's PII, S1364-6613(99)01294-2, and its pages carry the journal's headers. So it is this 1999 *Trends in Cognitive Sciences* review, not one of French's other papers. We extracted the text with `pdftotext` and read all eight pages in the source's own text. That covers the abstract, every section, Boxes 1–4, the outstanding questions and the reference list. The extraction lost the plus signs, so "111" in the text is the sum 1+1. We did not see Figure I in Box 2 as an image, only its caption. The reference list is partly out of order in the extracted text, and reference 22 is missing from it.

## Summary

**Claim.** Distributed networks with one set of shared weights can forget old knowledge suddenly and completely when they learn new patterns. The cause is the same thing that lets them generalize: overlapping representations in shared weights. The review covers the measures, the remedies of the 1990s, and the proposal that the brain avoids the problem with two separate learning systems.

**Method.** A narrative review of about fifty works, with no new experiments. It focuses on networks with "a single set of shared (or partially shared) multiplicative weights", mostly back-propagation networks. It groups the remedies into these classes:

- reducing the overlap between representations: sparse or semi-distributed hidden layers, orthogonal input codes, and pre-training.
- localist or modular designs: CALM, ART and ALCOVE.
- rehearsal of old items, and Robins's pseudopatterns when the old items are gone.
- dual-network models, where one network learns fast and passes what it learned to a long-term network.
- other techniques, such as fast and slow weights.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| McCloskey and Cohen (1989): a back-propagation network learns the 17 "ones" addition facts, then the 17 "twos" facts | Right answers on the ones facts fall from 100% to 20% within 1–5 trials of the twos. They reach 1% after five more trials, and none are right by 15 trials |
| Barnes and Underwood (1959): humans learn A–B paired associates, then A–C | Forgetting of A–B is "relatively gradual" |
| Exact-recognition measure (McCloskey and Cohen, Ratcliff) | A pattern counts as forgotten if any output node is not within 0.5 of its target |
| Savings measure (Hetherington and Seidenberg, 1989) | A network that seems to forget everything can often relearn it very quickly. Later work showed that not all forgetting is this "shallow" |
| Activation sharpening (French, 1991 and 1992) | Sparser hidden layers reduced forgetting "as long as there were not too many patterns to be learned" |
| Pre-training on a structured domain (McRae and Hetherington, 1993) | Catastrophic interference disappears from later learning |
| Sparse coding in combinatorial domains | Less interference, at the price of poorer generalization and poorer discrimination |
| Rehearsal of old items | Every method of choosing the old items studied made forgetting less severe |
| Pseudopatterns (Robins, 1995) | Random inputs and the network's outputs to them stand in for the old patterns. They reduce forgetting, though less well than the old patterns |
| Weight cliffs (Kolen and Pollack, 1990), Box 2 | A 2-2-1 network learning XOR: small changes in two of nine initial weights change convergence time a great deal. Each panel tests 40,000 weight pairs |
| Rats, Box 1 | Sequential learning of a 40 s and then an 8 s interval leaves no savings for 40 s. Concurrent learning of both lets the rat shift back quickly |

## Where the claims go beyond the evidence

- **A review, with no new tests.** Each number above comes from a cited study, and the review gives no seeds, runs or spread for any of them. We did not read the cited studies, so we can't check how French reports them.
- **Small networks and toy tasks only.** The evidence is addition facts, paired associates, XOR and auto-associators with few units. The review says little about how forgetting scales with network size or with the number of patterns. Its own note on sharpening says the gain holds only for few patterns.
- **The review argues that overlap is the single cause, but does not show it.** The claim that almost all remedies work by reducing overlap is a classification of the literature. The review asks itself whether every model with gradual forgetting needs some separation, and leaves it open.
- **The weight-cliff link is an inference.** Kolen and Pollack measured sensitivity to initial weights, not forgetting. Box 2 calls the link to forgetting "immediately obvious", but doesn't test it.
- **The rat study is a thesis, not a paper.** Box 1 rests on a 1998 PhD thesis, and the box gives no numbers.
- **The author reviews his own work.** French's sharpening and pseudo-recurrent networks take a prominent place. The conclusion favours dual networks, but French adds that this is "far from obvious" as the only way, which is a fair limit.

## Use in this repo

- **Late change in Pythia is gradual, not catastrophic, so far.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) followed 86 VERIFIED poles of Pythia-70m. Adjacent checkpoints after step 16,000 correlate at a median of 0.78 to 0.82. That is steady movement, not a sudden loss. Pythia's training is one interleaved pass over its data. The review's account says interleaving turns catastrophic forgetting into gradual forgetting, which fits these numbers.
- **Some explanations hold and then fail.** The same paper found that 51 of the 86 explanations pass at step 16,000, and 13 of those fail again at a later checkpoint. That is the review's exact-recognition measure applied to a pole. The review's point is that this measure alone can overstate the loss. A cheap check in the spirit of savings is whether those poles pass again at later checkpoints. We propose it as a follow-up issue below.
- **Two measures for any change to a model.** For the fix of #4781 (the " days" error, [`fixing-models.md`](fixing-models.md)), the review's lesson is to measure damage two ways. One is whether the neighbouring prompts still answer right, such as " legs" and " wheels" on the shortlist. The other is how fast the old behaviour comes back. How the campaign measures damage is the maintainer's question.
- **Overlap is what lets units share, and what lets them interfere.** The review's trade-off, generalization against interference, is the superposition trade-off. Superposition directions in `explain_groups.py` are sparse directions over many neurons, and so are overlapping representations in the review's sense. A pole that serves several features is where a later weight change could interfere. The review gives no single-unit test of this.
- **Semi-distributed coding matches the repo's working bet.** French argues for representations sparse enough to overlap little, but still distributed. The repo's premise that single MLP neurons can carry VERIFIED explanations is a bet on such partly sparse codes. The review supports the idea, but it gives no measure that the repo could use.
- **Second-hand evidence for two unread sources.** This review reports McCloskey and Cohen (1989) and Ratcliff (1990), the subjects of #4806 and #4807. Its numbers for the addition-facts experiment are second-hand, and those pages still need their own sources. Robins (1995) is #4809.
- **No single-unit content.** The review's units are hidden units in small networks, studied as whole patterns. Nothing in it bears on a pole's explanation or on our verdicts.

## Proposed follow-up

- **Do explanations that fail at a later checkpoint come back?** For the 13 poles in `papers/neuron-development/` that pass at step 16,000 and fail later, record the later checkpoints at which they pass again. This would separate a lost explanation from one that only moves away for a time. Area: analysis. Priority: low.
