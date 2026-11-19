# Understanding Catastrophic Forgetting in Language Models via Implicit Inference

**Citation.** Suhas Kotha, Jacob Mitchell Springer and Aditi Raghunathan (Carnegie Mellon University). "Understanding Catastrophic Forgetting in Language Models via Implicit Inference". arXiv 2309.10105. Versions: v1 of 18 September 2023 and v2 of 14 April 2024. [Paper](https://arxiv.org/abs/2309.10105). The PDF header says "Published as a conference paper at ICLR 2024", and the arXiv comment says "ICLR 2024". We did not check the proceedings page. Code: [github.com/kothasuhas/understanding-forgetting](https://github.com/kothasuhas/understanding-forgetting), which we did not open. The regression code builds on the code of Garg et al. (2023). The authors thank Apple, the AI2050 program at Schmidt Futures and an NSF Graduate Research Fellowship (grant DGE2140739). Huan Zhang gave compute for the regression runs.

**Status.** Not applicable. The repo does no fine-tuning, and EleutherAI/pythia-70m is a base model, so the paper's mechanism can't cause the #4781 copy failure. Its split of an answer into task inference and capability is a reading of that failure, which `papers/copy-first-unit/README.md` already tests in its own terms.

**Read.** On 2026-11-19 a model (Claude) read the arXiv v2 PDF for this page. It downloaded the PDF, turned it into text with `pdftotext`, and read the text directly. It read the main text (Sections 1 to 8) and Appendices A to F in the source's own words, with every table. It saw the figure captions but not the figures, so no number below comes from a plot. It read the reference list only in part. It did not read the harmful-output examples of Appendix F.2. It did not open the code, the OpenReview page or the HTML version.

## Summary

**Claim.** Fine-tuning raises performance on its own distribution and lowers it on other tasks. The authors propose that a language model infers the task of a prompt before it solves it. In their view, fine-tuning moves that inference toward the fine-tuning tasks more than it changes the model's capabilities. So a prompt transformed to look less like the fine-tuning data, but with the same answer, recovers part of the pretrained behaviour. They call this *conjugate prompting*.

**Method.**

- **Synthetic setup (Section 2).** A GPT-2 style transformer of 22.4 million parameters (256 dimensions, 12 layers, 8 heads) learns linear regression in context. Prompts hold 0 to 40 exemplars, with σ = 1 noise and learning rate 0.0001.
  - *D_cont*: weights from N(0, I), where ridge regression is Bayes-optimal.
  - *D_disc*: 64 fixed weight vectors, where a posterior-weighted sum over the 64 ("discrete regression") is Bayes-optimal.
  - *D_mix*: α·D_disc + (1 − α)·D_cont. Its Bayes-optimal answer weights the two solutions by g(X, y), the posterior that the prompt comes from D_disc.
  - Models pretrain on D_mix for 5,000 steps and then fine-tune on D_disc for 400 steps.
- **The hypothesis (Section 2.7).** The authors write the model's answer as g_θ times the discrete solution plus (1 − g_θ) times the ridge solution. The authors call g_θ "task inference" and the two solutions "capabilities". They state that, for lack of mechanistic understanding, they can't test whether models compute this way.
- **Conjugate prompting (Sections 2.8 and 3).** A transform s makes the prompt less likely under the fine-tuning distribution, and an inverse maps the answer back. In regression, s scales the labels by γ and the inverse divides the output by γ.
- **Language models (Section 4).**
  - *Instruction tuning*: LLaMa-7b against Alpaca and Vicuna-7b, and OPT-1.3b against OPT-IML-1.3b. Prompts hold in-context exemplars of a task (repeat or capitalise) and end with an instruction (arithmetic or fill in the blank). The measure is whether the first word follows the in-context task. There are 4 task pairs of 100 samples each, so 400 per cell. Prompts appear in English, French, Spanish, Dutch, Hungarian, Leetspeak and Pig Latin, translated with Google Translate.
  - *Code fine-tuning*: LLaMa-2 against Code LLaMa on XNLI, with 2,490 validation prompts in each of English, French, Spanish and German.
  - *Safety tuning*: 100 AdvBench instructions to the models the paper calls GPT-3.5 and ChatGPT, in five languages. One author labelled each output by hand as ANSWER, REFUSE or AMBIGUOUS.
- **Further checks (Appendix C).** A model of 51.3 million parameters, pretraining for 2,000 and 10,000 steps, more values of α and γ, and fine-tuning on mixtures.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Synthetic, α = 0.2, 22.4M, loss on D_cont before and after fine-tuning (Table 4) | 0.625 and 0.867, a drop of 0.242 |
| The same, drop that remains with conjugate prompting at γ = 1.5 and 2.0 (Table 4) | 0.163 and 0.170 |
| Synthetic, α = 0.5, 51.3M, drop and the drops with γ = 1.5 and 2.0 (Table 4) | 0.228, then 0.108 and 0.094 |
| Synthetic, α = 0.8, 22.4M, drop and the drops with γ = 1.5 and 2.0 (Table 4) | 0.199, then 0.091 and 0.086 |
| Which prompts lose most after fine-tuning (Section 2.6, Figure 5) | D_cont prompts that are likeliest under D_disc. Loss improves again as the likelihood falls further |
| Synthetic, fine-tuning on a mixture with α = 0.99 or 0.75 (Appendix C.4) | Smaller loss rises on D_cont, with less speed-up on D_disc at α = 0.75 |
| ICL first-word accuracy, LLaMa then Alpaca, English (Table 1) | 92.00% then 35.25%, a drop of 56.75% |
| The same, French and Leetspeak (Table 1) | Drops of 29.00% and 1.50% |
| LLaMa then Vicuna, English and Spanish (Table 1) | Drops of 33.00% and 11.00% |
| OPT then OPT-IML, English and Hungarian (Table 1) | Drops of 21.00% and 4.25% |
| XNLI accuracy, LLaMa-2 then Code LLaMa (Table 2) | English 44.26% then 35.90%. French 33.53% then 34.98%. Spanish 38.11% then 38.88%. German 34.50% then 33.49% |
| ANSWER rate on 100 harmful instructions, GPT-3.5 then ChatGPT (Table 3) | English 92% then 3%. Malayalam 71% then 65%. Japanese 56% then 9% |

## Where the claims go beyond the evidence

**The split into inference and capability is an assumption, not a measure.** In language models g_θ is never read from the model. The authors say so in Section 2.7. A smaller drop under a transform fits their hypothesis, but it also fits a transform that reaches less-tuned weights or a different solution.

**"Suppressed, not forgotten" rests on partial recovery.** In Table 4, conjugate prompting leaves 0.163 of a 0.242 drop at α = 0.2. Most of the loss remains, so part of the capability may be lost, not hidden. Appendix C.5 itself says fine-tuning changes task inference "rather than only degrading" capabilities.

**The scaled prompt is a different task.** Scaling labels by γ gives a weight vector of a larger norm. The authors note that γ = 3.0 loses the gains, and they suspect the model has not seen tasks that far out. So the transform changes more than the task inference.

**Floor effects in the XNLI table.** XNLI has three labels, so chance is near 33.3%. LLaMa-2 scores 33.53% in French and 34.50% in German before code tuning. A drop of under 2% from chance is not evidence that the capability survived in those languages.

**Smaller drops start from other baselines.** In Table 1 the pretrained accuracy changes by language, for example 76.50% in Leetspeak against 92.00% in English. The drop is an absolute difference. Table 5 also shows cells where translation does not help, such as Alpaca on Repeat and Math in Spanish, at 0.00%.

**The model names in Section 4.3 disagree with Appendix F.1.** Section 4.3 names the model before fine-tuning gpt-3.5-turbo and the one after it text-davinci-003. Appendix F.1 and Table 3 imply the reverse order. The paper does not say what tuning text-davinci-003 had. If it had tuning to follow instructions, the "before" model is not a pretrained model.

**A second mismatch.** Section 2.5 says Figure 4 shows α = 0.5, and the caption of Figure 4 says α = 0.2.

**No seeds or intervals.** The regression results report no training seeds. The language-model tables give one run per cell, with no intervals and no test across languages. One author labelled the harmful outputs, with no second rater.

**The language results test fine-tuning that the authors did not run.** They compare public model pairs. They don't control the fine-tuning data, so its language mix is an inference ("we expect that fine-tuning is primarily in English").

**Credit.** The authors state that they can't test the decomposition mechanistically. Their Limitations paragraph names imperfect translation, low-resource languages, and the need to know the training data. Appendix C repeats the synthetic result at twice the parameters and at two pretraining lengths. They also release code.

## Use in this repo

**The repo does no fine-tuning.** EleutherAI/pythia-70m has no fine-tuning stage, and the repo has no fine-tuning code (`docs/literature/fixing-models.md`, *What this repo already has*). So the paper's mechanism, a fine-tune that moves task inference, can't be the cause of the #4781 " days" error. The paper bears on the repo only as a reading of that failure, and as evidence for any fix by training. The maintainer designs the fix campaign, so this section gives evidence and no design.

**The copy failure has the paper's shape at the answer position.** Read with the paper's split, the answer mixes a copy solution and a recall solution, with some weight g between them. `papers/copy-first-unit/README.md` finds the right unit at rank 2 to 4 after layer 4 on five of the six wrong prompts. The copied unit is rank 1 or 2 there. That is the paper's picture of a capability that is present but outweighed, and the copy-first-unit paper itself calls the right answer "present but weak".

**But the recall solution is weak on its own, unlike the paper's capabilities.** In the paper, a suppressed capability is strong when inference picks it. The copy-first-unit paper's clause-alone test works like a conjugate prompt: it removes the clause to copy and keeps the answer. The right unit is then top for only one prompt of six, at ranks 1 to 15 (its Table 2). No MLP at the second clause's subject ranks the right unit better than 731 of 50,304 tokens. So in the paper's terms, both the weight g and the recall capability are at fault. The paper's frame alone would overstate how much a change of inference could fix.

**The copy is not held in one place.** Switching off the three verified copy heads, and L4/H1 too, flips one prompt of six (copy-first-unit, Table 3). If those heads were the whole of g, the recall answer would win more often. The paper's Discussion asks whether task inference runs through sub-networks that could be changed directly. The repo's answer routes could test that question, but these numbers show no single sub-network for it here.

**Context changes the answer, as the frame expects.** After four unrelated facts, the right unit's rank on the clause alone falls from 1–15 to 3–254 (copy-first-unit, Table 2). That fits a context that moves what the model infers. The repo did not measure it as task inference, so it is a fit, not a test.

**For a fix by training, the paper predicts where the damage falls.** Its Figure 5 finds the largest losses on prompts outside but close to the fine-tuning distribution. For a fix fitted on the six wrong prompts, the closest prompts are the shortlist's right copies. Examples are " legs" (0.5494) and " wheels" (0.4149) in `docs/analyses/hallucination-shortlist.jsonl`. `fixing-models.md` already names these prompts as the hard negatives that a #4781 fix must pass. The paper adds a reason to expect them to break first. Appendix C.4 also reports that mixing pretraining-like data into the fine-tune reduces the losses.

**A fix may hide the copy without removing it.** The paper predicts that a fine-tune moves inference and leaves the old capability in place. A transformed prompt with the same answer can then bring the old behaviour back. That matches the *Suppression comes back* rows of `fixing-models.md`, from in-context attacks and relearning. Conjugate prompting is one more test of that type, and it needs no training.

**Stability of VERIFIED explanations.** If the paper is right, the units of a capability would survive a fine-tune while inference moves. The repo can't check this yet. `open-interpretability sisters` compares models of different sizes and families (`docs/analyses/sister-neurons.md`), not a model and its fine-tune. The nearest data are pretraining checkpoints. `papers/neuron-development/README.md` finds that Pythia-70m's VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. Adjacent checkpoints after that correlate at 0.78 to 0.82.

**An early solution can be replaced in training.** Appendix C.3 reports that a model trained on D_disc first behaves like ridge regression, near step 500. By step 5,000 it changes to discrete regression. Open issue #5061 tracks the " days" answer and the copy heads over Pythia-70m checkpoints. The same run could record whether " months" ever led at an earlier checkpoint. That would be a comment on #5061, not a new issue.

**What does not carry over.** The paper defines g exactly only in the synthetic task. It says nothing about single neurons, probes or explanations. Its language-model evidence is behavioural, on 1.3B to 7B models and an API model, all far larger than Pythia-70m.

**Question for the maintainer.** Should a report of a fix by training include a transformed-prompt test, to tell moved inference from a removed behaviour? This page records it as a question, not an issue.
