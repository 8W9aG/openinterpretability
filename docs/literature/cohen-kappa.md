# A coefficient of agreement for nominal scales

**Citation.** Jacob Cohen (New York University). "A coefficient of agreement for nominal scales". *Educational and Psychological Measurement*, 20(1), 37–46, 1960. [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104). The work came from the Psychiatric Evaluation Project of the Psychology Service at the Veterans Administration Hospital in Montrose, New York (footnote 1). The paper has no code or data beyond its two worked tables.

**Status.** Used. `papers/token-set-ceiling/` reports Cohen's kappa for its two raters and cites this paper. `scripts/token_set_ceiling.py` and `scripts/reviewer_crossover.py` each compute kappa as Equation 1 defines it. `docs/analyses/reviewer-crossover.md` and `docs/analyses/registry-stats.md` report it.

**Read.** On 2026-11-28 a model (Claude) read all ten pages, 37 to 46, from a scanned PDF that the maintainer supplied (#6792). It read the text, Tables 1 and 2, Equations 1 to 11, the summary and the references. The publisher's page answered 403, and no open copy was found. The equations on this page are copied from the scan by eye.

## Summary

**Claim.** Two judges sort the same units into unordered categories. Their raw share of agreement, and χ² over their table, are poor measures of how well they agree. The paper proposes κ, the share of agreement after chance agreement is removed, with its limits, its maximum, its standard error and tests.

**Method.** A definition, its properties and one worked example. Let p_o be the share of units on which the judges agree. Let p_e be the share expected by chance, from the products of each judge's marginal shares. Then κ = (p_o − p_e) / (1 − p_e) (Equation 1). The paper takes three conditions as the assumptions of κ:

1. The units are independent.
2. The categories are independent, mutually exclusive and exhaustive.
3. The judges operate independently.

It also assumes no criterion of correctness, judges equally competent a priori, and no restriction on each judge's distribution over the categories. Every disagreement counts the same, because the categories have no order.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Raw agreement p_o = 0.29 against chance p_e = 0.35 | Table 1, three categories. χ² = 64.59 with 4 df at N = 200 is "highly significant", and C = 0.49 |
| χ² and C are "indefensible" as measures of agreement | They test association, so disagreement beyond chance inflates them too. In Table 1 the largest χ² cell (27.00) is a disagreement cell |
| κ = 0 at chance, 1 only at perfect agreement, negative below chance | The lower limit depends on the marginals: −1/(k − 1) when their correlation is 0 (Equations 3–5) |
| κ_M = (p_oM − p_e) / (1 − p_e) | Equation 6. The maximum κ that the marginals allow. p_oM sums the smaller of each pair of marginal shares. Table 1: κ_M = 0.85 |
| κ = φ when both judges have the same marginals | Two categories. With marginals within 0.10 to 0.20 of each other, κ is within a few hundredths of φ |
| σ_κ = √(p_o(1 − p_o) / (N(1 − p_e)²)) | Equation 7, an approximation. The normal approximation should hold with N large, "≥ 100". 95% limits are κ ± 1.96 σ_κ |
| σ_κ0 = √(p_e / (N(1 − p_e))) | Equation 10, the standard error under κ = 0, for a test against chance |
| κ = 0.492, κ_M = 0.831, σ_κ = 0.055, 95% limits 0.384 to 0.600, z = 8.34 | Table 2, three categories, N = 200 |

The paper also says that a test of κ against zero is of little value. Knowing that κ is beyond chance is trivial, since much more is expected of reliability.

## Where the claims go beyond the evidence

- **One invented example.** Both tables are illustrations, not data. The paper proves its properties, so no wider test is needed. But the claim that σ_κ is adequate for N ≥ 100 is stated, not checked.
- **σ_κ treats p_e as a constant.** The paper says so itself: the formula is an approximation that ignores the sampling error of the marginals. Later work gave exact large-sample variances. This page didn't read that work, so it doesn't report its numbers.
- **Two fixed judges.** Every formula is for one pair of judges. The paper says "two (or more) judges" once, on page 37, but gives no method for more than two.

## Use in this repo

- **The repo's two functions match Equation 1.** `scripts/token_set_ceiling.py::kappa` computes p_e from each rater's own marginals, over all labels, as Cohen does. `scripts/reviewer_crossover.py::kappa` does the same for yes/no calls. Both return 1.0 when p_e is 1, where Equation 1 is undefined. Then both raters gave one label to every item, so they agree fully, and 1.0 is a fair convention.
- **The token-set paper's numbers check out.** From its Table 3 (165 poles, three labels), p_o = 142/165 = 0.861 and p_e = 0.392. So κ = 0.771, as the paper reports. Token-set against the rest gives κ = 0.884, reported as 0.88. By Equation 6, κ_M is 0.900 and 0.922. So the raters' different marginals cost little of the possible agreement.
- **Each kappa lacks the standard error that Cohen gives with it.** By Equation 7, the token-set kappas have 95% limits of about 0.68–0.86 and 0.79–0.98. `registry-stats.md`'s κ = 0.37 on 231 pairs has limits of about 0.24–0.49. `reviewer-crossover.md` reports κ 0.38–0.52 on 30 submissions, below Cohen's N ≥ 100. There σ_κ is about 0.17 for κ 0.43, so the six arms' kappas can't be told apart. #7412 adds the standard error and κ_M from one shared helper.
- **The token-set paper's "raters" are two pools of agents.** Eight agents gave the first labels and two others the second. Cohen's κ is defined for two judges, each with their own marginals. Here each marginal pools several agents. So κ measures agreement between two labelling procedures, not two individuals. The paper's wording, "a second rater", should be read that way.
- **The three assumptions mostly hold, with one caveat.** The second agents labelled without the first labels, so the judges were independent. The labels are exclusive and exhaustive. But a neuron's positive and negative poles can both be in the sample, and they may not be independent units. Most of the token-set paper's disagreements are mixed against context (17 of 23). Cohen counts those the same as token-set against context. The paper's binary split, token-set against the rest, is the one that suits its question.
- **Fleiss' κ is a different source.** `docs/analyses/verdict-stability.md` reports Fleiss' κ for three readings. This paper doesn't define it, and this page doesn't review it.
