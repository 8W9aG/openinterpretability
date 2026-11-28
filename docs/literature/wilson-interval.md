# Probable inference, the law of succession, and statistical inference

**Citation.** Edwin B. Wilson (Harvard School of Public Health). "Probable inference, the law of succession, and statistical inference". *Journal of the American Statistical Association*, 22(158), 209–212, June 1927. A short note in the journal's *Notes* section. [doi:10.2307/2276774](https://doi.org/10.2307/2276774), [JSTOR](http://www.jstor.org/stable/2276774). It has no data and no funding statement.

**Status.** Used. `papers/token-set-ceiling/` cites it for its 95% intervals. The same interval is in six functions, for example `openinterpretability/interference/keep.py::wilson`. Many papers and analyses report it, among them `docs/analyses/registry-stats.md`, `null-calibration.md` and `sva-circuit.md`.

**Read.** On 2026-11-28 a model (Claude) read the note in full, pages 209 to 212, from a JSTOR scan that the maintainer supplied (#6793). JSTOR and OpenAlex list no open copy. The formulas on this page are copied from the scan by eye.

## Summary

**Claim.** The usual statement of probable inference puts limits p₀ ± λσ₀ around the observed rate. The note calls this "elliptical", because the true rate p is either inside a range or not. It is better to ask how probable an observation as bad as p₀ is, for each possible p. Solving that for p gives limits that are "the proper form of probable inference".

**Method.** Algebra on the normal approximation, with worked cases and no data. Let p₀ be the observed rate in n trials, q₀ = 1 − p₀, and t = λ²/n. Wilson solves (p₀ − p)² = λ²pq/n for p. The chance of an observation as bad as p₀ is at most P_λ if the true p lies outside:

(p₀ + t/2)/(1 + t) ± √(p₀q₀t + t²/4)/(1 + t).

With S successes in n trials, the centre is (S + λ²/2)/(n + λ²).

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| The range is centred at (p₀ + t/2)/(1 + t), not at p₀ | It moves toward 1/2 by t(1/2 − p₀)/(1 + t) |
| Its half-width R = √(p₀q₀/n + λ²/4n²)/(1 + λ²/n) | Less than λσ₀ when p₀ is between 0.067 and 0.933, for t small compared with 2 |
| The forecast of p is (S + λ²/2)/(n + λ²), "not (S + 1)/(n + 2), nor S/n" | λ = 0 gives S/n, and λ² = 2 gives Laplace's law of succession |
| The usual allowance of 2σ gives (S + 2)/(n + 4) | λ = 2 |
| With no successes (p₀ = 0), p lies below λ²/(n + λ²) | Where p₀ ± λσ₀ gives the range 0 to 0 |
| A choice of λ for each case | λ very small for the sun failing to rise, about 2/3 for leprosy deaths in Massachusetts in 1924 (n = 4,000,000), and 2 for paratyphoid fever |
| With overdispersion (Lexian ratio L > 1), widen λ | L about 5 gives λ = 10, where a Bernoullian phenomenon gets λ = 2 |

The note also says that the law of succession rests on the assumption that all probabilities are equally likely a priori. It calls the centre of an asymmetric range "perhaps unfair" as the best estimate. But it notes that practice uses it.

## Where the claims go beyond the evidence

- **No check of coverage.** The note derives the interval and gives cases. It doesn't measure how often the interval holds the true p for small n. That came from later work, which this page didn't read.
- **The normal approximation stays.** The interval inverts a normal test of p. The note doesn't say how small n can be before the approximation fails.
- **λ is a judgement.** The note ties λ to "our readiness to gamble on the typicalness of our realized experience". The leprosy value of 2/3 and the Lexian λ = 10 are examples, not rules.

## Use in this repo

- **The repo's code is Wilson's interval.** Each of the six functions computes the centre (p + z²/2n)/(1 + z²/n) and the half-width z√(pq/n + z²/4n²)/(1 + z²/n). With z = λ, these are Wilson's expressions. They differ only at the edges, in clamping, rounding and n = 0. #7417 keeps one of them.
- **The token-set paper's intervals check out.** `keep.wilson` gives each row of its Table 1 again. For example, 147 of 826 is 17.8%, with limits 15.3% to 20.6%. Qwen2.5-1.5B's 7 of 75 is 9.3%, with limits 4.6% to 18.0%.
- **The total-failure case is used, and correctly.** `null-calibration.md` reports 0 of 255 swapped hypotheses VERIFIED, with a false-positive rate below 1.5%. Wilson's bound λ²/(n + λ²) gives 1.48%.
- **The repo's rates are not always Bernoullian.** Wilson widens λ when the phenomenon varies more than chance (L > 1). The repo's units cluster. Poles of one neuron, layers of one model, and labels from one agent's batch share causes. The token-set paper's per-model rates, 9.3% to 27.5%, differ more than one binomial would allow. On its Table 1, χ² = 28.8 with 3 degrees of freedom, p < 0.00001. So a pooled interval such as 15.3%–20.6% is likely too narrow. A cluster bootstrap over neurons or layers would show by how much. This is a question for each analysis, so no issue covers it yet.
- **λ = 1.96 is a fixed choice.** Wilson lets λ follow the stakes. The repo always uses λ = 1.96, the 95% level. That matches current practice, and it keeps intervals comparable across analyses.
