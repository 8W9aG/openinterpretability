# Gaussian Error Linear Units (GELUs)

**Citation.** Dan Hendrycks, of the University of California, Berkeley, and Kevin Gimpel, of the Toyota Technological Institute at Chicago. A footnote says that Hendrycks did the work while at TTIC. "Gaussian Error Linear Units (GELUs)". [arXiv:1606.08415](https://arxiv.org/abs/1606.08415), version 1 of 27 June 2016, last revised as version 5 on 6 June 2023. The arXiv comment reads "Trimmed version of 2016 draft". Preprint, no venue given. Appendix B says the authors submitted it to ICLR in 2016, with no acceptance named. Code is at [github.com/hendrycks/GELUs](https://github.com/hendrycks/GELUs). The authors thank NVIDIA for donated GPUs.

**Mismatch.** None in the source. Section 3.2 says GELU "significantly outperforms" the other functions, but it gives no number and no test.

**Status.** Used. GPT-2's MLPs use the paper's tanh approximation of GELU, so every GPT-2 unit in the registry runs through it. `papers/default-on-poles`, `papers/spd-input-projections` and `papers/why-neurons-rest` rest arguments on its shape.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 5 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words: Sections 1 to 5, the references and Appendices A and B. Figures 1 to 7 are images, and the subagent read only their captions. We didn't read versions 1 to 4 or the PDF.

## Summary

**Claim.** GELU is x times the standard normal distribution function Φ(x). As an activation function, it matches or beats ReLU and ELU on vision, speech and language tasks.

**Method.**

- **Definition.** GELU(x) = xΦ(x), with Φ the distribution function of N(0, 1). It is the expectation of a stochastic mask that keeps x with probability Φ(x). Section 2 says that this combines ideas from dropout, zoneout and ReLU.
- **Approximations.** Section 2 gives 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)]) and xσ(1.702x). Section 4 says the experiments all use the tanh form.
- **Variants.** Other distribution functions give other units. The logistic one gives xσ(x), which the paper names the SiLU. The paper fixes μ = 0 and σ = 1.
- **Experiments.** MNIST classification and an autoencoder, Twitter part-of-speech tagging, TIMIT frame classification, CIFAR-10 and CIFAR-100. Each compares GELU with ReLU and ELU. Each reports the median of three or five runs.

**Findings, as the source reports them.**

| Setting | Median test error |
| --- | --- |
| Twitter part-of-speech tagging, two layers of 256 (Section 3.3) | GELU 12.57%, ReLU 12.67%, ELU 12.91% |
| TIMIT frames, five layers of 2,048 (Section 3.4) | GELU 29.3%, ReLU 29.5%, ELU 29.6% |
| CIFAR-10, 9-layer network (Section 3.5) | GELU 7.89%, ReLU 8.16%, ELU 8.41% |
| CIFAR-100, 40-layer wide residual network (Section 3.5) | GELU 20.74%, ReLU 21.77%, ELU 22.98% |

The MNIST results are in figures only.

## Where the claims go beyond the evidence

- **The margins are small, with no spread.** The gaps are 0.1 to 1 point, from medians of three or five runs. The paper gives no standard deviation, interval or test.
- **"Significantly" has no test.** Section 3.2 says GELU "significantly outperforms" the others on the MNIST autoencoder, from a figure alone.
- **Small networks, a 2016 baseline set.** The models are small by today's standard, and the comparison leaves out functions that came later. The authors can't be blamed for that.
- **No word on units.** The paper says nothing about single neurons, their resting levels or interpretability.

## Use in this repo

- **Our citations check out, with one fix.** The reference gives the right authors and the year of version 1.
  - `papers/default-on-poles` cites GELU = xΦ(x), and the tanh approximation that GPT-2 uses. Both are in Section 2.
  - `papers/why-neurons-rest` calls GELU not monotone. Section 4 says "non-monotonic".
  - `papers/spd-input-projections` says GELU acts on each neuron separately. The paper defines it on one neuron's input, so this is fair.
- **The minimum near −0.75 is ours, not the source's.** The paper never states where GELU's minimum is. `papers/default-on-poles` put the citation on that value, so this PR moves it to GELU itself. The minima are −0.7518 for the exact form and −0.7525 for the tanh form, by our computation.
- **The dip is the reason for default-on poles.** GELU is negative for inputs below zero, and the paper says only that GELU "can be both negative and positive". The registry's handling of default-on negative poles (`papers/default-on-poles`) follows from that shape, not from anything the paper tests.
- **The paper's motivation doesn't foresee the rest.** Section 2 motivates the normal distribution because "neuron inputs tend to follow a normal distribution". `papers/why-neurons-rest` finds that trained neurons rest near −1, below the dip, which that motivation doesn't foresee. Our reading, not the source's.
- **No question for the maintainer.**
- **Related pages.** [`neurons-speak-in-ranges.md`](neurons-speak-in-ranges.md) reads neurons by their activation ranges, which GELU's shape bounds below.
