# Universal Neurons in GPT2 Language Models

**Citation.** Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda and Dimitris Bertsimas. Most of the authors are at MIT, and Horsley is at the University of Cambridge. Nanda is the senior author. "Universal Neurons in GPT2 Language Models". [arXiv:2401.12181](https://arxiv.org/abs/2401.12181), version 1 of 22 January 2024, the only version. Our reference gives the venue as *Transactions on Machine Learning Research* ([OpenReview](https://openreview.net/forum?id=ZeI104QZ8I)). We couldn't open that page, so the venue isn't checked here. The text says its code and data are public, but the link didn't come through.

**Mismatch.** None found.

**Status.** Used. The registry's sister search ([`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md)) follows its method: the maximum correlation across models, against a random-rotation baseline. The `neuron-family-triage` skill uses its neuron families. Two papers cite it.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 7 in the source's own words, in two fetches. Appendix A, most of Appendix B and the end of the references came back only as a summary, and this page uses nothing from them. OpenReview showed a browser check, which stopped the subagent.

## Summary

**Claim.** Some neurons recur across models trained from different random seeds: "1-5% of neurons are universal". Universal neurons are more often interpretable, and they fall into a few families and functional roles.

**Method.**

- **Models and data.** GPT-2 small and GPT-2 medium, five seeds each ("GPT2-{small, medium}-[a-e]"), with Pythia-160M and larger Pythia models as support. Activations come from "100 million tokens from the Pile test set".
- **Matching.** For each neuron of model a, the Pearson correlation of its activations with every neuron of each other seed, keeping the maximum.
- **Baseline.** The same maximum against "a random (approximate) rotation of a layer of neurons". The *excess correlation* is the real maximum minus the baseline maximum, averaged over the other seeds. A neuron is universal when it exceeds 0.5. The authors say there is "no principled threshold".
- **Choice of setting.** Seeds of one architecture on the same data give "the most favorable experimental conditions for measuring universality", as a rough bound for larger changes.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Universal share (Section 4.1) | GPT2-medium-a 1.23% (1,253 of 98,304), Pythia-160M 1.26%, GPT2-small-a 4.16% |
| Depth (Section 4.1, Figure 2c) | "significant depth specialization": a neuron's best match tends to be at a similar depth |
| Families (Section 4.3) | Unigram, alphabet, previous-token, position, syntax and semantic neurons |
| Output roles (Section 5.1) | Prediction neurons in later layers, then many suppression neurons at the very end. One suppression neuron in GPT2-medium-a lowers "tokens containing an open parenthesis" |
| Entropy neurons (Section 5.2) | L23.945 in GPT-2 medium writes almost nothing to the logits and modulates confidence through the final layer norm |

## Where the claims go beyond the evidence

- **Small models, one kind of universality.** The authors say so themselves: hundreds of millions of parameters, and "neuron universality over random seeds within the same model family".
- **The threshold is chosen.** The 0.5 cut-off is a convention, and the share changes with it.
- **Interpretation is partly by hand.** The authors say many interpretations "rely on manual analysis or algorithmic supervision".
- **Why the shares differ is open.** GPT2-small's 4.16% against GPT2-medium's 1.23% may reflect size. The authors call it "an open question" without models trained from more seeds.

## Use in this repo

- **Our citations check out.**
  - `papers/explanations-across-models` (preprint): five seeds, the random-rotation baseline, "1–5%" and the quoted "most favorable experimental conditions" all match. Its step from seeds to models of other families is marked as its own.
  - `papers/syntax-positive-controls` (published): "a suppression neuron for tokens with an open parenthesis in a GPT-2 medium model" matches Figure 5's GPT2-medium-a. No addendum is needed.
  - `docs/analyses/sister-neurons.md` says that sisters sitting at the same relative depth "is how Gurnee et al. (2024) found universal neurons". The source reports depth specialization across seeds of one architecture, which fits. Our sisters cross models with different layer counts, so relative depth is the right comparison there.
  - The `neuron-family-triage` skill lists "unigram, alphabet, position, syntax and semantic". The source also has previous-token neurons, which the skill leaves out. That is a gap in a summary, not an error.
- **The sister search is this method across model families.** The registry correlates poles between different models, not seeds, with the same rotated baseline. So its sisters are a harder case than the source's universal neurons. `papers/explanations-across-models` makes that point.
- **Entropy and suppression neurons bear on the triage skill.** They are families whose explanation is about output effect, not input pattern. The skill already covers entropy neurons, from Stolfo et al. (2024).
- **No question for the maintainer.**
- **Related pages.** [`neurons-as-an-interpretable-basis.md`](neurons-as-an-interpretable-basis.md) and [`intriguing-properties.md`](intriguing-properties.md) take the two sides of whether neurons are a natural basis.
