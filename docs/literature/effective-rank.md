# The effective rank: A measure of effective dimensionality

**Citation.** Olivier Roy (EPFL) and Martin Vetterli (EPFL and UC Berkeley). "The effective rank: A measure of effective dimensionality". 15th European Signal Processing Conference (EUSIPCO 2007), Poznań, Poland, 3–7 September 2007, pages 606–610. [Paper](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf). The PDF prints no DOI. The Swiss National Science Foundation funded it, through NCCR-MICS (grant 5005-67322).

**Status.** Used. `scripts/dormant_rank.py` computes each MLP layer's effective rank with this paper's definition, and `papers/neuron-development/` reports it (Figure 4 and Table 4, #5134).

**Read.** On 2026-11-27 a subagent (Claude) downloaded the PDF from EURASIP and turned it into text. It read all five pages verbatim: the abstract, Sections 1 to 4, both figure captions and the references. The display equations came out of the two-column layout in pieces, and the subagent put them back together. So the equations on this page are rebuilt, and the prose quotes are verbatim. It did not see the two figures as images. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A matrix's rank counts its nonzero singular values, and it says nothing about their sizes. The effective rank is a real number from 1 to the rank. It is the exponential of the entropy of the normalised singular values, so it can be minimised with standard numerical methods.

**Method.** A definition and four proved properties, one worked example and one application.

- **Definition (Section 2.1).** For an M × N matrix with singular values σ_1 ≥ … ≥ σ_Q ≥ 0, where Q = min(M, N), let p_k = σ_k / ‖σ‖₁ (Eq. 1). This is the ℓ1 norm of the singular values, not their squares. Then erank = exp(−Σ p_k log p_k). "All logarithms are to the base e and we adopt the convention that 0 log 0 = 0."
- **Properties (Section 2.2).**
  - **1:** 1 ≤ erank ≤ rank ≤ Q. erank equals 1 when one singular value carries everything. It equals the rank when the nonzero singular values are equal.
  - **2:** erank doesn't change under transpose, conjugate or scaling by c ≠ 0.
  - **3:** a unitary transform doesn't change it.
  - **4:** erank(A + B) ≤ erank(A) + erank(B), proved for positive semidefinite Hermitian matrices only.
- **Other norms.** With minor changes, Properties 1 to 3 hold for an ℓp norm in place of ℓ1. With the ℓ0 norm, the effective rank becomes the rank.
- **Meaning (Section 3).** Through Campbell's coefficient rate, the effective rank is "the average number of significant dimensions in the range of A", in the limit of many samples.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| erank is largest at ρ = 0, where it equals the rank, 4, and it falls toward 1 as \|ρ\| grows, while the rank stays 4 | A 4 × 4 circulant matrix with correlation ρ (Section 2.3, Figure 1) |
| Both methods find α_opt ≃ 1.04 without noise. With noise, the effective rank "clearly outperforms" the smallest singular value, which "basically provides no insight" | Fitting N = 2 diffusive sources from tomographic measurements, at an SNR of 50 dB (Section 4, Figure 2) |

## Where the claims go beyond the evidence

- **One application, shown as plots.** The case against the smallest singular value rests on one problem with N = 2, in one noisy case. The paper gives no number of noise trials and no measure of spread.
- **The meaning is asymptotic.** The reading "average number of significant dimensions" holds as the number of samples, or the matrix size, goes to infinity. The paper says nothing about estimating the effective rank from a finite sample, or about its bias.
- **No word on centring.** The paper treats the effective rank of a given matrix. It never says whether a data matrix should be centred first. Section 3 assumes zero-mean random vectors.
- **Property 4 is partial.** The authors say so themselves: "It is not clear whether Property 4 still holds for arbitrary M × N matrices".

## Use in this repo

- **The repo's code matches the definition.** `scripts/dormant_rank.effective_rank` takes `p = σ / Σσ`, the ℓ1 normalisation of Eq. 1. It uses the natural log, and it drops zero values, which is the paper's `0 log 0 = 0`. Its doctests check Property 1's two ends: the 4 × 4 identity gives 4, and a rank-one matrix gives 1. Dohare et al. (2024) use the same definition ([Loss of plasticity](loss-of-plasticity.md)).
- **Compare only at one sample size.** Property 1 caps the effective rank at min(M, N). A layer of Pythia-70m has 2,048 neurons, so a sample of 2,000 tokens caps it at 2,000, and all 18,782 tokens at 2,048. The paper's Table 4 compares checkpoints on the same seeded 2,000-token sample, from 433 to 1,306, so each comparison is like with like. `erank_all`, on all tokens, is a separate scale. It shouldn't be compared with the 2,000-token values, and the paper doesn't compare them.
- **The activations aren't centred.** The script's docstring says so. The paper is silent on this, so the choice isn't wrong. But a large mean direction can carry much of a layer's first singular value. GELU outputs have a positive mean, so an uncentred effective rank partly measures that mean, not only how the tokens spread around it. This doesn't change Table 4's comparisons, because every checkpoint is measured the same way. It does mean that the numbers aren't comparable with a centred measure elsewhere. A centred variant would be a one-line change. Neither the paper nor any open issue asks for one, so no issue is opened.
- **Scale.** Property 2 makes the measure blind to a layer's overall scale. So the late fall in rank in Table 4 is a change in the shape of the spectrum, not in the size of the activations.
