# Retries: how many tries a VERIFIED unit took

Run on 2026-11-14 for issue #3204, with `scripts/retries.py`. The numbers are in `retries.json`. Reported, not gated.

The question: every follow-up on a unit is another chance to reach VERIFIED. The site's false-discovery estimate (`tree.false_discovery`) sums 1 − P(useful) over the current VERIFIED claims, as if each claim took one try. How many tries did the claims take? What happens to the VERIFIED share and to that estimate when the tries count? And do units VERIFIED after several tries hold up on held-out text as well as units VERIFIED on the first?

## Method

- **Units.** Every native neuron pole and attention head with an experiment: 1,909 units and 2,550 records. Dictionary latents are a baseline, so they are left out, as the site leaves them out. The script reads the records and the errata as the site does (`build_tree`). So it drops experiments that an erratum sets aside, and it marks re-measurements the same way.
- **Attempts.** A unit's experiments are put in verdict order, and counted up to and including its first VERIFIED. A unit that was never VERIFIED counts all its experiments. *Independent* attempts leave out re-measurements of an earlier probe set (`Experiment.reuses`), which judge the same probes again under a newer rule.
- **Shares.** Each comes with a 95% Wilson interval:
  - *per record*: VERIFIED records over records.
  - *per unit*: units ever VERIFIED over units.
  - *per attempt*: units ever VERIFIED over the attempts spent on all units.
- **Union bound.** Take each current VERIFIED claim. P is its weakest agreeing review's P(useful), as on the site. k is its independent attempts to the first VERIFIED. The bound charges the claim min(1, k × (1 − P)). That bounds the chance that any of its k tries passed by error. It needs no model of how the tries depend on each other. It can only overstate.
- **Held-out text.** The comparison uses the detection and fuzzing balanced accuracy (`openinterp score`) of the record that first VERIFIED each unit. It compares units VERIFIED at the first independent attempt with units VERIFIED later, within each judge, pooled with weights n₁n₂/(n₁+n₂). The intervals come from 2,000 bootstrap samples drawn within each judge (`scripts/verified_vs_detection.py`).

## Result

**Most VERIFIED units took one try. One in five took more.**

| Independent attempts to the first VERIFIED | Units |
| --- | --- |
| 1 | 463 (81%) |
| 2 | 93 (16%) |
| 3 | 14 (2%) |

Counting re-measurements too, one unit took 4 experiments. Of the 570 units ever VERIFIED, 512 were random draws and 57 were chosen by hand. 1 was chosen by attribution.

**Counting the tries lowers the VERIFIED share a little.**

| Share | Value (95% Wilson) |
| --- | --- |
| Per record (708 of 2,550) | 27.8% (26.1–29.5%) |
| Per unit (570 of 1,909) | 29.9% (27.8–32.0%) |
| Per attempt (570 units over 2,355 attempts) | 24.2% (22.5–26.0%) |
| Per independent attempt (over 2,275) | 25.1% (23.3–26.9%) |

For random draws only, the shares are 30.0% per unit and 24.3% per attempt. So about one try in four ends a unit's search at VERIFIED. The per-unit share is higher because the search stops at the first success. The per-unit share answers "what fraction of studied units has a verified explanation?". The per-attempt share is the success rate of one more try.

**Charging the tries raises the expected false discoveries by at most a fifth.**

| Estimate | Claims | Expected false | Rate |
| --- | --- | --- | --- |
| Today's (`tree.false_discovery`) | 522 | 22.1 | 4.2% |
| Union bound over tries | 522 | 26.1 | 5.0% |

The bound is an upper limit. Most claims took one try, and P(useful) is high for most claims, so the charge for a second or third try is small. The site's 4.2% understates the rate, but by less than one percentage point at most.

**Units VERIFIED after several tries do a little worse on held-out text.**

| Score of the first VERIFIED record | First try | Later try | Within judge, first − later (95% CI) |
| --- | --- | --- | --- |
| Detection | 0.655 (n = 431) | 0.601 (n = 104) | +0.043 (+0.008 to +0.080) |
| Fuzzing | 0.626 (n = 368) | 0.580 (n = 91) | +0.032 (−0.001 to +0.068) |

Detection's interval excludes zero, and fuzzing's just includes it. So an explanation that took several tries predicts held-out activity a little less well. That fits a selection effect: a unit that is hard to explain gets more tries, and its final explanation is less complete. Or a retry may pass the review on a near-miss. This data can't tell the two apart.

## What this means

- **The site's false-discovery number is close.** Counting retries moves it from 4.2% to at most 5.0%. Showing the bound next to it would be honest and cheap. That is a change to the site, so it is left for a separate decision.
- **Quote the per-unit share for coverage, and the per-attempt share for effort.** "30% of studied units have a VERIFIED explanation" is right. "A try verifies 30% of the time" is not: it is 24%.
- **Weight retried claims a little less.** Units VERIFIED after a retry are a fifth of the claims, and they score about 0.04 lower on held-out detection. A reader who wants the strongest claims should start with first-try ones.

## Limits

- **The union bound ignores how the tries were made.** A revision that answers the failed probes is not a blind retry, so the bound overstates its risk. A re-review of an unchanged hypothesis is closer to a blind retry.
- **Later tries differ in more than their number.** They come later, often under newer protocol versions and on harder units. The held-out comparison controls for the judge only.
- **Held-out scores exist for 535 of the 570 first VERIFIED records.** The rest predate `openinterp score` or were not scored.

## Reproduce

```bash
uv run python scripts/retries.py   # writes docs/analyses/retries.json
```
