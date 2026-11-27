# Harness-Delta Attribution: Dissecting Gains From Harness Evolution

**Citation.** Wenxuan Ding (Amazon and New York University), Qin Lu, Changlong Yu, Sha Li, Shuowei Jin, Xin Liu and Zhenyu Shi (Amazon). Greg Durrett is at New York University. "Harness-Delta Attribution: Dissecting Gains From Harness Evolution". [arXiv:2610.22466](https://arxiv.org/abs/2610.22466) [cs.AI], v1 of 15 October 2026, the only version. Preprint, no venue given. The paper is under CC BY 4.0. The [code](https://github.com/Wenwen-D/HarnessDeltaAttribution) holds the method, the evolved harnesses and the data splits, but it has no licence file. A [blog post](https://wenwen-d.github.io/blog/harness-delta-attribution/) of 6 August 2026 gives earlier numbers that differ from the paper's. This page uses the paper's. The Sloan Foundation, the NSF and the Simons Foundation funded the work. Amazon is not named as a funder. Claude Code (Opus 4.8) is the proposer in every run. The paper doesn't name the model behind its analysis agent.

**Status.** Not applicable. The repo runs no harness search, and no open issue would use the method. Its main point does bear on the reviews: fresh held-out data can share the artifact that a fit exploits. `docs/analyses/retries.md` (#3204) already measures part of this, and *Use in this repo* says what is left.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the PDF and turned it into text. It read the whole paper with Appendices A to G, and checked the HTML against it. It also read the code's README, its skill file and its significance test, and the blog. It saw the figures only as text that the extraction scattered. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A gain from harness search, where an agent rewrites the code and prompts around a frozen model, can be split into three parts. These are overfitting to the search set, extra test-time compute, and a residual that bounds the gain that can generalise. In most settings, overfitting and compute explain most of the gain.

**Method.**

- **The split.** T is the gain that the base harness gets when it is given the same compute, for example pass@3 in place of pass@2. O is the score lost when each detected shortcut in the evolved harness is disabled. G is the rest. All four scores are on the search set. O is a lower bound on overfitting and G an upper bound on the gain that generalises.
- **Who finds the shortcuts.** An agent proposes the compute match and the disabled shortcuts, and a person approves each one.
- **Settings.** 11 executor models, from Qwen3.5-0.8B to Claude Opus 4.8. There are three search frameworks: Meta-Harness, SkillOpt and AHE. The benchmarks are ALFWorld, LiveMath, CREATE and SWE-bench Verified. That gives 25 settings, each with search, validation and test splits.
- **Significance.** A paired bootstrap on the search set decides whether a gain is real. The code shows a two-sided 95% interval from 10,000 resamples.
- **Validation selection.** One variant shows the proposer only the aggregate validation score, and keeps the harness with the best validation score.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Overfitting explains 54% of the gain on average, and extra compute 14% | The 21 of 25 settings with a significant gain. Each average is a mean of capped shares |
| Overfitting and compute together exceed half the gain in 14 settings | The same 21 |
| The 0.8B harness grew from 66 to 503 lines, and became a controller that makes no model calls | Meta-Harness on ALFWorld. It reached 100% on the search set and 93.3% on held-out games |
| One answer option, "a stronger result can be proven", is always right. It appears in 21 search questions and 53 test questions | LiveMath. All three frameworks found this shortcut |
| Test accuracy 53 of 53 on questions with that option, and 22 of 71 on the rest | The Meta-Harness 0.8B harness on LiveMath |
| The held-out gain is smaller than the search gain in 13 of 15 settings, and below the baseline in 4 | All settings outside ALFWorld |
| Validation selection improves the held-out score in 10 of 16 settings and lowers it in 6. In 14 of 16, the held-out gain stays below the search gain | Meta-Harness only |

## Where the claims go beyond the evidence

- **One run per setting.** Each setting has one evolution run, with seed 42. The paper gives no intervals for O, T or G, and no correction across its 25 tests. The search sets are small: 35 to 120 items, and an SWE-bench gain of +8.3% is 4 items.
- **The test runs on the set that chose the winner.** The bootstrap test runs on the search set, which also picks the best harness. The code's own skill file calls this "anti-conservative", a winner's curse. The paper doesn't widen the test for it.
- **The split depends on judgement.** An agent and a person decide which mechanisms are shortcuts and how to disable them. The paper doesn't measure how often two reviewers would agree. Disabling too much would raise O, as the skill file warns.
- **Capped shares, not raw values.** The table gives O, T and G as capped percentages. In one CREATE setting, compute alone beats the evolved harness, so the raw G is negative but shows as 0.
- **No limitations section.** The limits are spread through the text. O is a lower bound and G is an upper bound. Only shortcuts that can be found and removed are counted. The authors say these themselves.
- **A citation that may be wrong.** The paper cites LiveMath as Mahdavi et al. (2025). The blog and the code's data path point to LiveMathematicianBench (He et al., 2026).

## Use in this repo

- **A revision faces fresh probes, as a harness faces its held-out split.** Under `docs/PROTOCOL.md` §3, a revision is a new submission, and a new reviewer writes new probes for it. The proposer sees the earlier failed probes, as the proposer here sees its search-set results. So a revision can fit the failed probes, but it must still pass probes it has not seen. That is the paper's held-out split, and the paper shows two ways it can fail to catch a fit.
- **Fresh probes can share an artifact.** On LiveMath the shortcut was in the test set too, so held-out accuracy rose with it. The repo's analogue is a reviewer whose probes share a habit, such as positives that always quote the explanation's own words. A new reviewer of the same model may share the habit. `docs/analyses/reviewer-crossover.md` found that the reviewer model shifts the VERIFIED rate by about a factor of two. #7136 reviews a paper on the same risk from weak attackers. This page adds that a fit to the reviewer's habits can pass fresh probes. It opens no issue of its own.
- **The repo already measures the cost of retries.** `docs/analyses/retries.md` found that 81% of VERIFIED units took one try. Units that took more tries score lower on held-out detection, by +0.043 (95% interval +0.008 to +0.080). The analysis can't tell a selection effect from a near-miss pass. The paper's split suggests a way to tell them apart. Disable the part of a revised explanation that answers the failed probes, then score it again. That needs a rule for what "the part that answers the probes" is. The maintainer chose on 2026-11-27 not to do it.
- **A shortcut has a counterpart in the registry.** The ALFWorld controller replaced the model with rules. The closest analogue is an explanation that only lists the tokens of the top examples. It passes probes built from those tokens but explains nothing more. `papers/token-set-ceiling/` counts such token-set rules among VERIFIED poles: 147 of 826, or 17.8%. That paper treats them as real rules that a weight-only check can test, not as overfitting. The two readings differ in whether the tokens generalise to new text. The held-out detection score tests exactly that.
- **Compute matching has no analogue here.** A unit gets the same review whatever its number of tries. The retries analysis's union bound already charges each extra try. So this page opens no issue for T.
- **This repo's own loop.** The repo's agents edit the skills that drive them, which is a harness search ([CoEvoBench](coevobench.md), [Harness Evolution Hits a Ceiling](harness-evolution-ceiling.md)). The paper's advice is to name the cost limits before the search, and to audit an accepted change by disabling it and scoring again. The repo doesn't score its loop, so neither applies yet.
