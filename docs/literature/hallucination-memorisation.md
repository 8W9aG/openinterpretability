# LLMs Memorize Their Own Hallucinations During Post-Training

**Citation.** Anirudh Atmakuru and Tanya Goyal (Cornell University). "LLMs Memorize Their Own Hallucinations During Post-Training". arXiv:2611.26863, v1, 19 November 2026, cs.CL with cs.LG, CC BY 4.0. [Paper](https://arxiv.org/abs/2611.26863). Preprint, no venue given. The paper links no code, checkpoints or data. NSF grant IIS-2433072 and the Empire AI Consortium funded it. Its AI-use statement says that generative AI detected the hallucinations and wrote the evaluation questions, the distractors and the SFT traces. The tools named are GPT-4.1-mini and OpenAI's web search. It also says that AI edited parts of the text.

**Status.** Not applicable. The paper measures memorisation by behaviour only, in six Qwen3 and Olmo models of 1 to 30 billion parameters, none of which the registry holds. It tests no fix of a memorised fact beyond more training, a KL penalty and PPO. Its probe of recall before and after an update is one way to test a fix. But the fix campaign already tests each fix on held-out and retain prompts (#5220). So no code, skill or protocol here would use it, and no open issue needs it.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the abstract page and the full HTML of v1, and turned them into plain text. It read the main text, Sections 1 to 7, and Appendices A.1 to A.5 verbatim, with the numbers of Tables 1 to 14. It read only the totals and the first rows of the per-checkpoint Tables 15 to 18. It did not see Figures 1, 2, 5 and 6, which are images. So the log-probability curves, the difficulty trend and the retention plot are known from their captions only. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** RL with outcome rewards makes a model memorise the false facts that it writes in its own reasoning. A rewarded rollout can reach the right answer through a false statement. The update then raises the model's recall of that statement, and lowers its chance of abstaining on it.

**Method.**

- **Models and training.** The models are Qwen3-4B, 8B, 14B and 30B-A3B, Olmo-3-7B-Think and OLMoE-1B-7B.
  - **QuALITY:** each base model gets continued pretraining on the stories, then an SFT cold start, then one epoch of GRPO with no KL penalty.
  - **TriviaQA:** GRPO continues from the developers' post-trained checkpoints.
  - **Rewards:** 1 for a right answer in the right format, and 0.2 for the right format only. Each question gets 8 rollouts.
  - **Checkpoints:** 8 saved at equal intervals, plus the start.
- **Self-generated hallucinations.** For each training question with some right and some wrong rollouts, one right rollout is drawn. GPT-4.1-mini takes one false fact from its reasoning. For TriviaQA, three candidates are checked by web search, and one false fact is kept. HalluSet-train holds 15,795 facts over the 12 model and dataset cells, 552 to 2,435 each.
- **Recall probe.** Each fact becomes a four-option multiple-choice question. The hallucination is the "gold" option, and GPT-4.1-mini writes three distractors.
- **Memorisation score** (Eq. 1). The score is the mean recall over the checkpoints after the update that rewarded the fact, minus the recall just before it. The unit is percentage points.
- **Control.** HalluSet-heldout takes hallucinations, extracted the same way, from 1,000 test questions that no checkpoint trains on. It separates the reward's effect from a general sharpening of the model.
- **Abstention.** One distractor becomes "None of the above". The measure is how often the model picks it before and after the update.
- **Other checks.** The paper also tests these:
  - backtracked hallucinations, which the model takes back later in its reasoning.
  - hallucinations from wrong rollouts.
  - a second epoch, a KL penalty of 0.001, and PPO in place of GRPO.
  - shuffled options and new distractors.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Memorisation score 4.83 points, the mean of Table 1's Train column | 12 model and dataset cells. QuALITY averages 5.66 and TriviaQA 4.01 |
| 0.17 points, the mean of the Heldout column | The same cells, for facts that got no reward |
| Abstention falls by 7.23 points on average, for example 27.41% → 18.07% for Qwen3-4B on QuALITY | "None of the above" on HalluSet-train, before against after the update (Table 2) |
| An average change of 0.07 points in abstention | HalluSet-heldout (Table 10) |
| Backtracked and non-backtracked facts differ by at most 0.18 points, with p > 0.05 for every model (paired permutation test) | TriviaQA only |
| The KL penalty lowers the score by 0.52 points on average | QuALITY only, coefficient 0.001 |
| Hallucinations from wrong rollouts score below zero in general | Table 11 |
| 25–55% of the sentences in right rollouts hold a hallucinated fact | 200 right rollouts per model, labelled by an LLM |
| PPO gives scores close to GRPO's, for example 5.90 against 5.85 for Qwen3-4B on QuALITY | Table 14 |

**What the headline is.** The abstract writes "4.83%". Table 1's caption gives it as "4.83 points", the mean of the Train column. It is not the gap between train and held-out. That gap averages about 4.67 points, a number the paper does not quote. 0.17 and 7.23 are also percentage points.

## Where the claims go beyond the evidence

- **One run each, with no intervals.** No seed is reported, and no table gives an interval or a standard error. The only significance test is for backtracking, where it finds no difference. The central comparison, train against held-out, has no test. With 552 to 2,435 facts per cell, a binary score and nine checkpoints, the gap is probably real, but the paper doesn't show it.
- **An LLM writes most of the measure.** GPT-4.1-mini finds the hallucinations, writes the questions and writes the distractors. The authors checked parts of this by hand:
  - one author checked 200 TriviaQA facts against the web, with a precision of 98%.
  - backtracking labels were 87% and 91% accurate.
  - in 39 of 200 TriviaQA questions, a distractor was the true fact.

  §2.2 says GPT-4.1-mini labels the sentences, but Figure 18's caption names GPT-5.1-mini.
- **The control has less headroom.** Before the update, recall is higher on the held-out facts than on the train facts, for example 77.97% against 69.83% for Qwen3-4B on QuALITY. The authors read the sets as "broadly comparable". A set that is already recalled more has less room to rise, and the paper doesn't discuss this.
- **Multiple choice is not generation.** The authors state this limit themselves. Picking the hallucination in a multiple-choice question "does not imply that the model will also surface the hallucination under free-form generation". They add log-probability curves as a closer proxy, and this page saw only their caption.
- **No limitations section, and no release.** The paper releases no code, no data and no checkpoints, so HalluSet can't be checked.
- **Scale.** The models have 1 to 30 billion parameters, and there are two datasets and one epoch of RL. The claim about "post-training" in general rests on GRPO and PPO with outcome rewards.

## Use in this repo

- **It measures memorisation by behaviour, in models outside the registry.** The paper never looks inside a model. It has no neurons, no activations and no edits, and it locates nothing. Its six models are not configured here, and no new model joins until each configured model reaches `coverage_target` (`pyproject.toml`). So the issue's question has no input here: could a pole track a hallucinated fact across checkpoints? The nearest work is `papers/neuron-development/`. It follows 86 VERIFIED Pythia-70m neurons across 20 pretraining checkpoints. That is pretraining, not RL, and on facts the model wasn't rewarded for.
- **Its probe and a fix's evidence.** The memorisation score is recall after an update minus recall just before it. A fix could be scored the same way: recall of the false fact before and after the patch. But the fix campaign already asks more of a fix (#5220 §9):
  - the failure prompts it fixes, on held-out items.
  - retain prompts that it must not break.
  - an interference report, at the level of each unit and each route.

  A four-option recall question is a weaker test than those. It also needs an LLM to write the options. The paper itself warns that the option chosen is not what the model would generate. So the probe adds nothing that #5220 needs.
- **Where it touches the sequential-tuning paper.** `papers/sequential-tuning/` (#5125) measures what narrow fine-tunes break: 74% of the base model's VERIFIED explanations keep after three stages. This paper measures what RL writes in: false facts that the reward happened to pass. The two point the same way. An update changes more than its target, and a check on the target alone misses the rest.
- **A design point for fixes, as framing only.** GRPO here rewards a right answer reached through a false step. A fix that is scored only by its final answer has the same weakness. That is one more reason why the fix campaign explains every prompt a fix uses, route and all, before it trusts the fix (#5220 §9.6). It changes nothing here, and no follow-up issue is needed.
