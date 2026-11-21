# Confidence Regulation Neurons in Language Models

**Citation.** Alessandro Stolfo (ETH Zürich) and Ben Wu (University of Sheffield), equal contribution. Then Wes Gurnee (MIT), Yonatan Belinkov (Technion), Xingyi Song (University of Sheffield), Mrinmaya Sachan (ETH Zürich) and Neel Nanda. "Confidence Regulation Neurons in Language Models". [arXiv:2406.16254](https://arxiv.org/abs/2406.16254), version 1 of 24 June 2024, revised as version 2 on 8 November 2024. The arXiv comment says "NeurIPS 2024". Code is at [github.com/bpwu1/confidence-regulation-neurons](https://github.com/bpwu1/confidence-regulation-neurons). The work was part of the MATS programme. The funders include the Israel Science Foundation, the Long-Term Future Fund, Open Philanthropy and EU Horizon Europe.

**Mismatch.** The source disagrees with itself in a few counts. Section 4 shows "5 selected" token frequency neurons, and Appendix H.2 says six. Section 6.1 says 100 sequences, and Appendix D says 500. Appendix F puts the LayerNorm share in Pythia and Gemma at 30–40%, and Appendix A at about 40%.

**Status.** Used. The `neuron-family-triage` skill lists entropy and token frequency neurons as families, from this paper. The `activation-patching-practice` skill says how to test them. `papers/why-neurons-rest` builds its account of the final LayerNorm on the two families.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). In three fetches it had Sections 1 to 8 and Appendices A to I in the source's own words. It saw the figure captions but not the figures.

## Summary

**Claim.** Two kinds of final-layer neuron regulate a model's confidence. Entropy neurons write into an effective null space of the unembedding and scale the logits through the final LayerNorm. Token frequency neurons move the output toward or away from the unigram distribution.

**Method.**

- **Entropy neurons.** Neurons with "an unusually high weight norm" and little direct effect on the logits. The unembedding has an "effective null space": in GPT-2 Small its singular values drop to near 0 around index 755 of 768. The share of a neuron's output weight in the bottom singular vectors measures how much it writes there.
- **Their mechanism.** Mean-ablate a neuron with the final LayerNorm's scale held fixed (the direct effect), and compare with the total effect. The gap is the effect through the LayerNorm.
- **Token frequency neurons.** Mean-ablate each final-layer neuron while holding the logits fixed along v_freq, the centred log frequency of each token. A neuron whose effect goes mostly along v_freq is a token frequency neuron.
- **Models.** GPT-2 Small and LLaMA2 7B in the main text. GPT-2 Medium, Pythia 410M and 1B, Phi-2 and Gemma 2B in Appendix F.
- **Samples.** 256-token inputs, 30 to 100 sequences per model, and 500 for induction.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Entropy neurons, effect through the final LayerNorm (Appendix F) | About 80% of the total effect in GPT-2 Small, LLaMA2 7B and Phi-2. 30–40% in Pythia and Gemma |
| Random neurons (Section 3.2) | "no significant difference between their total and direct effects" |
| Neuron 11.2378 in GPT-2 Small, repeated text (Section 6.1) | "up to a 70% reduction in entropy" when ablated, so it "counter-balances overconfident predictions" |
| Neuron 23.417 in Pythia 410M (Section 5) | A token frequency neuron, "suppressing common tokens and promoting rare ones" |

## Where the claims go beyond the evidence

- **Final layer only.** The authors say the study "focuses exclusively on two particular types of final-layer components".
- **Confidence is a proxy.** The authors note "the lack of a precise definition of confidence", and use entropy and distance from the unigram distribution.
- **Thresholds by eye.** No cutoff on weight norm, null-space share or frequency effect is given. Families are read from scatter plots.
- **No seeds and few tests.** "Significant" is used with no test named. The spread across models is put down to architecture and training, which the authors "do not explore".

## Use in this repo

- **Our citations check out.** `papers/why-neurons-rest` says entropy neurons change the final LayerNorm's scale, and token frequency neurons move the output toward or away from the unigram distribution. Both match the abstract and Sections 3 and 4. The seven authors, 2024 and NeurIPS 2024 match the arXiv listing.
- **The constant offset is our step.** `papers/why-neurons-rest` finds that a constant offset along the MLPs' mean output does both things at once. The source studies single neurons, not a shared offset. Our reading: the step fits the source's own preprocessing. It centres output weights, because components along the all-ones direction "have no effect". It also cites earlier work on a "constant bias to the residual stream".
- **The triage rules are proxies.** `neuron-family-triage` finds entropy neurons by a large output norm and a low share in the unembedding's span. It finds token frequency neurons by a correlation between W_U·w_out and log frequency. The source finds both by causal effects, with weight statistics as the first screen. Our reading: the skill's weight tests are a fair screen, and a neuron that passes still needs the source's ablation test, as `activation-patching-practice` says.
- **Top-token metrics miss them.** The source shows their effects go through confidence, not the top token. The triage skill already warns that a top-1 logit metric fails these neurons, and the protocol's next-token KL sees them.
- **No question for the maintainer.**
- **Related pages.** [`universal-neurons.md`](universal-neurons.md) reviews the earlier report of entropy neurons in GPT-2 Medium.
