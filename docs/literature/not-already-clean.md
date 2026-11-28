# Neural Networks Are Not Already Clean

**Citation.** Leo Gao. "Neural networks are not already clean". *LessWrong*, 17 October 2026. [Post](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean). It is a blog post on the Frontpage, with no peer review and no edit date shown. It had 372 karma and 24 comments when we read it. It links no code or data, and it states no licence, funding or disclosure. It replies to Nick Cammarata's *Neural Networks Are Already Clean* (Goodfire, 15 October 2026, [already-clean.goodfire.com](https://already-clean.goodfire.com/)), which the `native-neuron-audit` skill follows.

**Status.** Planned. #3210 would score an explanation by what happens downstream when its simulated activations replace the real ones, which is the test that Gao proposes. #7327 measures how much of a passing pole's effect lies below the review's firing line, and #7328 scores explanations below that line. Gao's two other points are tests that the review already runs.

**Read.** On 2026-11-28 a `web-fetch` subagent read the post and its comments for this page's author, a model (Claude Opus 5.5). It returned the whole body and all seven footnotes in one fetch, but quotes were capped at 125 characters, so the rest came back paraphrased. The comments came back as summaries. One comment's body did not render, and neither did the crossed-out text in another. We did not read Cammarata's post for this page. Its claims here come from `.claude/skills/native-neuron-audit/SKILL.md` and `neuron-basis-objections.md`.

## Summary

**Claim.** Cammarata shows that people called neurons polysemantic too quickly, and Gao agrees: "we should update on it". But a simple pattern that fits a neuron's top examples is "nowhere near a sufficient argument" that the neuron is understood. Gao's best guess is still that concepts lie in superposition and that MLPs compute on them in superposition.

**Method.** The post is an argument with illustrations. It reports no experiments and no numbers of its own. It makes four points.

- **Weak activations can mean something else.** Gao's illustration is a neuron whose top examples are dogs riding bicycles. At the 20th percentile the examples are bicycles with a dog nearby, and the weakest are dogs alone. He asks whether "dogs riding bicycles" then explains the neuron, and answers no.
- **Fit is not the converse.** A pattern that fits every activating example doesn't show that the neuron fires on every instance of the pattern. A neuron that fires only on dogs on bicycles on sunny days also fits every example. Footnote 2 puts it as "X matches Y is not commutative!" He adds that forcing such a neuron to fire on every instance of the pattern would degrade the model.
- **Weak activations matter causally.** Gao expects that "ablating small neuron activations quite substantially degrades model performance". He calls this his best guess from results that he has seen. Footnote 4 says he hopes to share something concrete, and that his result differs from Cammarata's.
- **Simulation scores are badly built, not too demanding.** Cammarata says that simulation asks too much of an explanation. Gao says the problem is "the kinda terrible correlation-based simulation scores" of earlier automated interpretability. He would turn an explanation into a rubric and simulate the neuron's activations from it. The simulated activations then replace the real ones in the model, and the score is the downstream KL on a large dataset. Footnote 5 prefers KL to the change in loss, because errors of several neurons can offset each other in the loss. Footnote 6 says that one neuron's KL is tiny, so neurons should be compared with each other or ablated in batches.

**Findings, as the source reports them.** The post has no findings to tabulate. Each point above is an argument or an expectation. The comments add the following, which we did not check against their sources.

| Commenter's claim | Source given |
| --- | --- |
| About 87% of a Qwen3-32B neuron's activations above half its maximum are about Colorado. "Tornado" reaches about 8.6% of the maximum | Julian Bradshaw, from Goodfire's visualiser data |
| In a ConvNeXt, zeroing the weakest 75% of activations in each block costs 1 point of accuracy. The weakest 95% cost 19 points, and the weakest 99% cost 68 | ShivM, from an appendix of Cammarata's post |
| Tries at a protocol that trains explainers with downstream KL as the reward mostly failed at Qwen 14B scale | Gonçalo Paulo |

A commenter asks whether the same point about fit applies to SAE latents and transcoder features. Another commenter, not Gao, replies "Yes."

**Where the claims go beyond the evidence.**

- **No data.** The central causal claim, that ablating weak activations hurts the model, is an expectation from results that the post doesn't show. Gao says so, and notes that it disagrees with Cammarata's own ablations.
- **The main example is made up.** The dogs-and-bicycles neuron is an illustration, not a measured neuron. It shows what could go wrong, not how often it does.
- **The KL score is a sketch.** The post doesn't give the dataset, the model, how the rubric becomes activations, or what score would pass. Gao admits that KL is hard to compare across models of different sizes.
- **Superposition is asserted.** The post says that circuits between neurons are still "relatively large and complicated", but it cites no work for this.

Gao states the limits of his evidence himself, and frames the post as a call for rigour.

## Use in this repo

- **The converse check already runs.** Gao's second point is that inputs which fit the explanation must also fire the neuron. The review's testing model writes held-out positives for that purpose, beside its hard negatives. In `papers/lora-units/` each audit had 20 hard negatives and 10 held-out positives, and the review checks that the positives fire. The `interpretability-illusions-review` skill tells the reviewer to write them. So this point is a test in the review, not a gap.
- **The scores read weak activations, but only above the line.** `docs/analyses/score-by-range.md` bins each pole's active windows by strength. Four fifths of them peak between 0.25 and 0.44 of the maximum. The judge's detection recall rises from 0.38 in the weakest bin to 0.51 in the strongest. So the scores are not drawn from the top examples, and they lose some recall as activations weaken. That is the direction that Gao predicts. But every bin is at or above the line at 0.25 of the maximum, so no score reads Gao's weakest examples. #7328 scores the windows below the line.
- **The causal weight of weak activations is not measured here.** `docs/analyses/pole-effects.md` mean-ablates whole poles, not bands of activation. Tigges et al. (2026) ablate by percentile band on another model, and the middle half of the activations costs little loss (`neurons-as-an-interpretable-basis.md`). Gao expects the opposite. #7327 measures the share of a passing pole's effect below the line, which tells whether the line leaves out much of what a pole does.
- **The line already has a known cost.** `papers/conformal-line/` finds that the fixed line fires on a median 23.1% of held-out text in pythia-70m and 0.7% in Qwen2.5-1.5B. So "below the line" covers very different parts of the range in different models, and #7327 should report each model apart.
- **Scoring by downstream KL is #3210.** That issue replaces a pole's activation with its value on another token that the explanation labels the same, and checks that the loss barely changes. Gao's proposal is the same test, with KL in place of loss. We commented on #3210 with his argument for KL.
- **Goodfire's reply is still untested.** `neuron-basis-objections.md` records Goodfire's answer to Gao: a weak activation adds less of the same concept. The answer is stated as a hypothesis. #7327 and #7328 would test it on this repo's poles.
