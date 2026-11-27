# Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents

**Citation.** Seven authors. Xing Zhang, Guanghui Wang and Yanwei Cui are at AWS Forward Deployed Engineering, and Ziyuan Li at HSBC Technology Center, China. Then Wei Qiu, Bing Zhu and Peiyang He, the corresponding author. The text that we read doesn't make the affiliations of Qiu and Zhu clear. "Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents". [arXiv:2610.11464](https://arxiv.org/abs/2610.11464) [cs.AI, cs.CL, cs.LG], version 1 of 8 October 2026, the only version. The arXiv comment says "Accepted at the NeurIPS 2026 Workshop: Who Verifies the Agents? Toward Reliable Agent Development". The licence is CC BY-NC-SA 4.0. The text links no code. It gives no funding statement. Every model role in its loops is a Claude model. Don't confuse it with the same group's *Who Grades the Grader?* (arXiv:2607.12790).

**Status.** Planned. The repo doesn't evolve its graders: its verdict rule is fixed code, and only a versioned protocol change can change it. But the paper's main lesson applies to the review's model roles. A grader needs a reference set that it never reads, and a rising pass rate can't stand in for one. The blind human audit (#4034) and the injection-recovery tests (#3565) would give the repo that reference.

**Read.** On 2026-11-27 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It had the source's own text for the abstract, Sections 1 to 6, Algorithm 1 and Appendices A to E. It also had Tables 1 to 3, the captions of Figures 1 to 3, and the start of Appendix F. The rest of Appendix F, all of Appendix G and the end of the reference list came back only as a summary. The subagent did not see Tables 4 and 5, Figure 4 or the TeX source. Quotes were capped at 125 characters.

## Summary

**Claim.** A verifier can evolve with the agent that it grades, and still stay valid. To do that, it must be inspectable and kept against a small held-out reference. The agent's own score can't certify the verifier: "Downstream task score cannot certify a self-evolved verifier".

**Method.**

- **The loop.** A verifier evolves while a frozen solver writes outputs. The solver is "a single direct LLM call with no tools, iteration, or scaffolding". The verifier phases alternate with the phases of a skill loop that improves the agent. The paper calls the pair the Double Ratchet, and adopts the skill loop (arXiv:2605.22148) unchanged.
- **The verifier.** It is "an inspectable expression over small, mostly deterministic drawback detectors". The ops come in three cost tiers: static, execution and model-judge.
- **New detectors.** A synthesizer writes one op for each recurring cluster of failures. An op is born when it "fires on ≥ half its cluster and clean on known-good" (Algorithm 1).
- **Composition and selection.** A model composer joins the ops with a small grammar: and, or, not, K-of-k votes and weighted votes. It selects by agreement with a ten-item reference set of "five clear failures and five clean positives", and by weighted consensus on unlabelled outputs.
- **The guards.** A candidate with no usable opinion on the reference set can't be selected. A validity gate drops candidates that "pass, fail, or abstain on everything".
- **Tasks.** MBPP+, where hidden unit tests are the ground truth. Spider 2.0-Snow, where execution match is. Report generation, which has no accurate verifier, so a rubric scores it for reference only.
- **Models and statistics.** Every model role in the loops is Claude Opus 4.7. Claude Opus 4.8 is the outer judge of reports, outside every loop. Each configuration ran with three seeds, on locked test sets of 40, 40 and 48 items.
- **Ablations.** "Naive" turns off the reference-set guards and the skill rollback. "No-lifecycle" turns off the birth gate, the shadow ops and retirement.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Held-out agreement of the evolved verifier | MBPP+, 3 seeds (§5.1) | 0.625 ± 0.050, against 0.417 ± 0.058 for the hand-written seed (+0.21) |
| The same for the bare model judge | MBPP+ (§5.1) | 0.55 ± 0.04 |
| Paired test of fixed against broken verdicts | MBPP+, 120 matched verdicts, sign-flip by task (§5.1) | 61 against 36 at the peak (p = 0.12). 36 against 20 at the end (p = 0.21) |
| Held-out agreement | Spider 2.0-Snow (§5.1) | 0.500 ± 0.026, "near chance" |
| Retention of the reference loop's lift, as the paper defines it | MBPP+, Table 1 | 106%. The held-out task score is 0.717, against 0.700 for the ground-truth loop |
| The same | Spider, Table 1 | 110%. The task score is 0.458, against 0.483 for the ground-truth loop |
| The same | Reports, Table 1 | 88%. The task score is 0.812, against 0.850 for the rubric loop. The gap is −0.04, with a 95% interval of −0.05 to −0.03 |
| The naive arm's verifier | Reports, Table 3 | It collapses to always-pass on every seed, passing 0.97 to 1.0 of the items |
| The naive arm's task score | Table 1 | As high as the guarded loop. On MBPP+ it is 0.742 against 0.717 |
| Gamed report tags | Reports, §5.4 | About 30% of tags at the peak had no value beside them. After a fix, about 1% |
| Win rate over the baseline, task-aware judge | Reports, Table 2 | 0.515 before the fix, 0.770 after |

The text that we read doesn't give the formula for retention, so this page doesn't compute it from the task scores.

**Where the claims go beyond the evidence.**

- **The lift is not significant.** The authors say so: "The paired test makes the lift consistent but not significant". With the tasks as the unit, p is 0.12 at the peak and 0.21 at the end. Only counting each verdict as independent gives 0.014 and 0.044, which the authors call anti-conservative.
- **One of three validation tasks shows no lift.** On Spider the verifier's agreement is 0.500, at chance. So the headline retention of 88% to 110% is mostly a claim about the skill loop, which a weak verifier can still steer.
- **Small sets and one model family.** The authors state "three task families, one model family", and "40 locked tasks and three seeds". Their own words for what that supports are "a consistent direction, not a calibrated rejection". Every model role is a Claude model, with no human study.
- **The reference set is ten items.** The anchor that keeps the verifier valid has five failures and five passes. Appendix F replays it at four to eight items, but no larger set is tested.
- **No code.** The runs need credentialed warehouse access for Spider, and no repository is linked. So nobody can rerun the study as published.

The authors give these limits themselves, and they call the work "a mechanism study, not a scaling result".

## Use in this repo

- **The verdict rule is not a grader that evolves.** A verdict comes from measured activations, through the review's code (`review_metrics` and `decide`), under a numbered protocol (`docs/PROTOCOL.md` §6). A change to the rule needs the maintainer and a version bump. So the paper's failure, a grader that drifts to always-pass while scores rise, can't happen to the rule itself.
- **The model roles are graders, and they have fixed checks.**
  - The model that writes the test prompts decides much of a verdict (`docs/analyses/reviewer-crossover.md`). The explanation judge settles contested verdicts.
  - The null calibration measures how often the review passes a claim that should fail. Shuffled labels pass 1.34% of GPT-2 records and 2.16% of Qwen records (`docs/analyses/null-calibration.md`).
  - Planted neurons give a reference with a known answer. 5 of 7 plants passed, and none of the 5 passed with a wrong explanation, although that interval reaches 43% (`docs/analyses/ground-truth-calibration.md`).
  - The null-explanation control and the injection tests check the judge against empty, swapped and hostile explanations (`null-explanation-control.md`, `judge-injection.md`).
- **These references were each measured once.** The paper's lesson is that a reference only guards a grader if it is measured again after every change to that grader. In the repo, the prompts, the skills and the models of these roles change often. The checks above don't rerun when they do. A rising VERIFIED share therefore can't show that the review is sound, for the reason the paper gives.
- **What would change.**
  - The blind human audit (#4034) would be the repo's human-checked reference set, as the paper's ten items are.
  - The injection-recovery tests (#3565) would be a reference that no proposer reads.
  - I commented on both issues with this paper's argument for rerunning them after each change to a model role.
- **This project's own loop.** Agents change the skills and scripts that drive them, which makes the repo a harness that improves round by round. Its guard is the one that the paper asks for: CI tests that agents don't grade, and a maintainer who decides the rules. *CoEvoBench* (`coevobench.md`) and *The Winner's Curse* (`winners-curse-self-improvement.md`) make the same point.
- **A question for the maintainer.** Should a change to a model role's prompt or model rerun a fixed reference before it lands? Examples are the prompt writer's skill and the judge's model. This page records the question and opens no issue, because it changes how the protocol is kept.
