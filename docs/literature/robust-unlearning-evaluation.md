# Eight Methods to Evaluate Unlearning in LLMs

**Citation.** The authors are Aengus Lynch (MATS, University College London), Phillip Guo (MATS, University of Maryland) and Aidan Ewart (MATS, University of Bristol). Also Stephen Casper and Dylan Hadfield-Menell (both MIT CSAIL). The exact title is `Eight Methods to Evaluate Robust Unlearning in LLMs`. The heading of this page uses a short form. arXiv 2402.16835, version 1, 26 February 2024. [Paper](https://arxiv.org/abs/2402.16835). The abstract page lists no later version, no code link and no arXiv comment. Thus the venue is "Preprint, no venue given". We saw no funding statement in the text that we read.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it. The paper lists tests that a fix must pass. Only then can we say that the fix removed a behaviour, and did not only hide it.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v1 HTML for us, in three passes. It returned short verbatim quotes from the abstract, the affiliations, the method, the trivia, latent-knowledge, side-effect, jailbreak, in-context and trivial-prompt sections, and the discussion. The tool limits each quote to 125 characters. So the eight-method list, the related work, the probe setup and most result details came back only as its summary. We did not see the figures, the PDF or the full appendices. Most of the paper's numbers are in its figures, so this page gives few numbers.

## Summary

**Claim.** Unlearning evaluations often use one ad-hoc metric. The paper tests one unlearned model with eight more methods. It finds that the knowledge stays in the model and comes back with little effort. In the authors' words, ad-hoc metrics "may be misleading about overall effectiveness".

**Method.**

- **Model under test.** The "Who's Harry Potter" (WHP) model of Eldan and Russinovich (2023). Those authors "fine-tune Llama-2-7B-Chat" "to unlearn knowledge of the Harry Potter universe". Their method fine-tunes on text in which generic content replaces the Harry Potter content (our paraphrase of the source).
- **Baselines.** The original Llama-2-7B-Chat, and a trivial prompt that tells the model to act as if it does not know Harry Potter.
- **Base metric.** Eldan and Russinovich's Familiarity score, from a GPT-4 judge, "using “gpt-4-turbo-preview” at seed=42 and temperature=0, with max tokens=252".
- **The eight methods (summary).** Other languages (Spanish and Russian, by GPT-4 translation), jailbreak prompts and in-context relearning. Also relearning by fine-tuning, downstream trivia tasks and latent knowledge by probes. The last two are the trivial prompting baseline and side effects on similar domains.
- **Relearning by fine-tuning (summary).** Rank-8 LoRA on five-sentence excerpts from the first three books. The runs use 800 or 8,000 sentences, on one A6000 GPU.
- **Trivia.** "The trivia dataset consists of 1239 questions", made with GPT-4. There are short-answer questions and binary-choice questions, split into easy and hard.
- **Latent knowledge.** Supervised linear probes and unsupervised Contrast-Consistent Search (CCS) probes on the residual stream. There is "one probe per layer at the last token position, on both the Llama-2 and WHP models".
- **Side effects (summary).** Familiarity question sets for English mythology, Dungeons and Dragons, Harry Potter film production, Lord of the Rings and the Wizard of Oz. Each set has 45 to 50 questions.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Familiarity, WHP against Llama-2 | WHP "obtains a Familiarity 77% lower than Llama-2's" |
| Any of the extraction methods | "higher-than-baseline amounts of knowledge can reliably be extracted" |
| Jailbreak prompts (two prompts) | "modest increases in the WHP model’s Familiarity both absolutely and relative to the original model" |
| In-context relearning, Book 1 excerpts or summaries | "these examples and summaries increase the WHP model’s absolute Familiarity" |
| Relearning by fine-tuning, 8,000 sentences | "fine-tuning on 8,000 sentences brings the WHP model's performance close to the original Llama-2 baseline" |
| Trivia, 1239 questions | The Familiarity gap reverses for short-answer questions, and becomes much smaller for binary-choice questions (our paraphrase). WHP "even performs better than the original model on short-answer trivia questions" |
| Latent knowledge, easy questions | Probes find the right answer in WHP with the same accuracy as in Llama-2 (our paraphrase of one sentence) |
| Latent knowledge, hard questions | "Llama-2 vs WHP supervised probes have a noticeable but relatively small gap in performance". "WHP unsupervised probes seem to be less stable across layers" |
| Side effects, related domains | "WHP loses significant Familiarity in related domains, including English Mythology". The authors "found Familiarity drop in four of them and no difference in the fifth" |
| Trivial prompting baseline | It "barely affects WHP Familiarity and reduces Llama-2 Familiarity, but not to the level of WHP" |

The authors give a reason for the trivia result: "the model does not need to output any Harry Potter-specific tokens to perform the task". Their conclusion is that "it will be important to implement adversarial evaluations". They also ask future work to give unlearning methods that adversarial evaluations cannot defeat (our paraphrase).

## Where the claims go beyond the evidence

**One unlearned model, one domain, one base model.** All results are for WHP, on Harry Potter, on Llama-2-7B-Chat. The title names eight methods for unlearning in general. The paper does not show how the methods rank other unlearning methods, or other domains.

**The judge is one model, at one seed.** Familiarity comes from one GPT-4 judge at seed 42 and temperature 0. The text reports no variation across judge runs, prompts or fine-tuning seeds. We found no confidence intervals or error bars. Sample sizes appear only as dataset counts.

**The trivia questions come from the same judge family.** GPT-4 wrote the 1239 questions and graded the short answers. GPT-4 also gives the Familiarity score. So the metric that the paper criticises and the test that it proposes share one source of error.

**The cross-lingual test cannot separate the two models.** The authors state this limit. Llama-2 itself does badly in Spanish and Russian. So the two models give low Familiarity in those languages for different reasons.

**An inconsistency in the side-effect domains.** The text says "These 5 domains were British mythology, Harry Potter film production, Lord of the Rings, and Wizard of Oz". That list names four domains. The dataset list also has Dungeons and Dragons, which is probably the fifth.

**No mechanism.** The probes show that a direction for the answer is still present. They do not show which units keep the knowledge, or why the fine-tuning left them. The paper does not study neurons.

**Credit.** The authors say that Familiarity may favour WHP. High scores need the Harry Potter terms that the WHP training removes. They also give a trivial prompting baseline. That control shows that WHP is more than a prompt-level effect.

## Use in this repo

**A fix in this repo will face the same question.** A fix guided by an explanation changes the units on a VERIFIED route. The paper shows that a target metric can drop while the knowledge stays. For #4781, the target metric is the probability of " days" on one prompt. On that prompt " days" has probability 0.73 and " months" is at rank 4 (`docs/analyses/hallucination-shortlist.jsonl`). A fix that moves that one number would be the analogue of the 77% Familiarity drop.

**Most of the eight methods have a version here.** These are the ones that carry over:

- **Paraphrases and other prompts.** The shortlist kept 6 of 35 prompts, and all 6 copy the first clause's unit. A fix that works on one prompt and not on the other five is like a fix that fails in Spanish. #5015 already asks whether the route generalises to them.
- **In-context relearning.** One extra example such as "a year has twelve months" in front of the prompt is a cheap test. If the wrong copy returns or the fix stops working, the fix is shallow.
- **Relearning by fine-tuning.** The paper uses rank-8 LoRA. #4782 covers LoRA and interference, and the same small adapter could measure how fast a fix comes undone.
- **Latent knowledge.** `open-interpretability probe-compare` (`openinterpretability/cli_probing.py`, `openinterpretability/core/probing.py`) already trains logistic and mass-mean probes. It uses the MLP hidden layer and the residual stream. Train a probe for the copied unit before and after a fix. It would show whether the fix removed the copy, or only hid it at the output.
- **Side effects.** Registry coverage (`openinterpretability/coverage.py`) and the sister tool (`openinterpretability/sisters.py`) can measure what a fix changes in other VERIFIED units. That is the analogue of the four related domains.

**Where the argument does not apply.** The paper studies unlearning for safety, against an adversary. A fix for #4781 adds a right answer, " months", and does not have to resist an attack. So jailbreak prompts have no clear use here. The useful part is the principle: test a fix with methods that the fix did not optimise.

**What the paper predicts for the #4781 copying failure.** The verified route copies " days" through layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. A fix that damps those heads on this prompt is like WHP. The paper predicts that the first-clause unit would still be readable in the residual stream at the answer position. It also predicts that one in-context example could undo the fix. This is a prediction from the paper, not a measurement.

**Question for the maintainer.** Should a fix experiment report more checks beside its target metric? Examples are paraphrase prompts, an in-context example, a probe and a side-effect measure. The fix campaign is the maintainer's design. So this page records the question and opens no issue.
