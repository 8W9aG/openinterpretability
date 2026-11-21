# Incorporating Second-Order Functional Knowledge for Better Option Pricing

**Citation.** Charles Dugas, Yoshua Bengio, François Bélisle, Claude Nadeau and René Garcia, of CIRANO, Montréal. "Incorporating Second-Order Functional Knowledge for Better Option Pricing". *Advances in Neural Information Processing Systems 13 (NIPS 2000)*, edited by T. K. Leen, T. G. Dietterich and V. Tresp, MIT Press, 2001, pages 472–478. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2000/hash/44968aece94f667e4095002d140b5896-Abstract.html). The paper links no code or data.

**Mismatch.** Section 4 says the third quarter of 1988 is for validation and quarters "4 to 20" are test sets. Table 1's caption says the fourth quarter of 1988 is the test set and the first quarter of 1989 is a second test set.

**Status.** Used. `papers/why-neurons-rest` trains a softplus condition (`scripts/train_resting.py`), and credits the function to this paper.

**Read.** On 2026-11-21 a `web-fetch` subagent read the proceedings page, but couldn't extract the PDF's text, a scan. So this page's author, a model (Claude Opus 5.5), read all seven pages as images, figure included.

## Summary

**Claim.** A network can be built so that its output is positive, increasing in its inputs and convex in some of them. Such a network can still approximate any function of that kind. Building in that prior knowledge helps the model predict option prices over time.

**Method.**

- **Softplus.** Section 2 defines ζ(s) = log(1 + eˢ), "which we call *softplus*". It is the primitive of the sigmoid, so it is convex with derivative 1/(1 + e⁻ˢ).
- **The model.** Each hidden unit is a product of softplus terms over the convex inputs and sigmoid terms over the others. Exponentiated weights keep the derivatives non-negative.
- **The proof.** A sketch for two inputs shows that the class is a universal approximator of such functions.
- **Data.** 43,518 prices of European call options on the S&P 500, 1988 to 1993. The inputs are moneyness and time to maturity.
- **Baselines.** A one-hidden-layer perceptron, and a network shaped like the Black-Scholes formula, each with free or constrained weights.
- **Training.** 1,000 random starts per model, the best trained further, repeated 10 times. Results average the best of these.

**Findings, as the source reports them.**

| Setting (Tables 1 and 2, mean squared error ×10⁻⁴, best validation model) | Test 1 | Test 2 |
| --- | --- | --- |
| Perceptron, free weights | 1.27 | 30.56 |
| Perceptron, constrained weights | 3.08 | 3.81 |
| Products of softplus and sigmoid, constrained | 1.97 | 4.19 |
| Sums of softplus and sigmoid, constrained | 2.00 | 4.21 |

The free-weight networks do best on the first test set and fail badly on the second. Figure 1 shows the constrained models holding up through 1993.

## Where the claims go beyond the evidence

- **One market, one period.** The data are S&P 500 options from 1988 to 1993.
- **A sketch of a proof.** The authors give the proof "only for the case n = 2 and c = 1", for lack of space.
- **The proposed model doesn't win on every test.** On the second test set the constrained perceptron and Black-Scholes networks score lower than the proposed one. The authors' claim rests on stability over the later quarters in Figure 1.
- **No intervals.** The tables give single averages, with no spread.

## Use in this repo

- **Our citations check out.** `papers/why-neurons-rest` credits softplus to this paper and writes it as $\log(1 + e^h)$, which is Equation 5. It calls softplus "a smooth curve with no dip". The source shows it is convex with a positive derivative everywhere, so it has no dip. The five authors, NIPS 2000, 2001 and pages 472–478 match the PDF.
- **Softplus was a tool here, not the subject.** The source uses softplus to make a function convex, not as a hidden-layer activation for general networks. Our reading: using it as one, as `papers/why-neurons-rest` does, is a later step. The paper is still the right citation for the function and its name.
- **No question for the maintainer.**
- **Related pages.** [`gelus.md`](gelus.md) reviews GELU, the baseline activation in the same paper.
