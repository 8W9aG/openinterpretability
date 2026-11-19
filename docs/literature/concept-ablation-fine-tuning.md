# Concept Ablation Fine-Tuning

**Citation.** Helena Casademunt (Harvard University, work done during MATS), Caden Juang (Northeastern University), Adam Karvonen (independent), Samuel Marks (Anthropic), Senthooran Rajamanoharan and Neel Nanda. "Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning". arXiv 2507.16795. Version 1 is from 22 July 2025. We read version 2, from 9 November 2025. [Paper](https://arxiv.org/abs/2507.16795), [code](https://github.com/cadentj/caft). The text that we read gave no affiliation for Rajamanoharan and Nanda. The abstract page lists no venue, so this is a preprint, no venue given. We found no funding statement in the part that we read. Fine-tuning used "a single H100/A100".

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, could use CAFT as one fix method. The repo has no fine-tuning code now, so nothing depends on the paper yet.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v2 HTML for us in four passes. The tool let it quote only short phrases, under 125 characters each. It returned short verbatim quotes from the abstract, the introduction, Section 3 (method), Section 4 (emergent misalignment) and the limitations. It also quoted Appendices B, C, G and H. Other parts came back as its summary. These include the author list, the baselines, the multiple-choice results and most table values. Two table values, the Qwen vulnerability rates, came back from a table and not from a sentence. We did not see the figures, the PDF, Appendices D to F in full, or the full abstract in one quote.

## Summary

**Claim.** Interpretability tools can find the directions in a model's activations that carry an unwanted concept. If you project those directions out during fine-tuning, the model learns the task without the unwanted generalisation. In the authors' words, "CAFT reduces misaligned responses by 10x without degrading performance on the training distribution."

**Method.**

- **Idea.** First, find "directions in the model's latent space corresponding to unintended concepts". Then, "at every forward pass during fine-tuning", project the activations onto the complement of those directions. The projection is in the forward and the backward pass (summary). "After fine-tuning, model inference is run as normal, with no ablations."
- **Finding directions with PCA.** The authors "collect the difference in residual stream activations between the base and fine-tuned models". A person reads the examples with the largest and smallest projection on each principal component, and picks the unwanted ones.
- **Finding directions with SAEs.** The authors "use sparse autoencoders (SAEs) to discover important concepts in model activations". They sort latents four ways and read the top ones. "Only the combination of all methods works for all cases." Appendix C repeats a part of the selection with an automatic interpreter.
- **No target data.** The concepts come from the training data and the model, and "we never use any data from" the out-of-distribution set.
- **Emergent misalignment.** Two models learn to write insecure code from 4,500 examples (summary). They are Qwen2.5-Coder-32B-Instruct and Mistral-Small-24B-Instruct-2501. The authors ablate 6 PCs for Qwen and 8 for Mistral, or 10 and 15 SAE latents. They use three layers per model: 12, 32 and 50 for Qwen, and 10, 20 and 30 for Mistral (summary).
- **Multiple-choice tasks.** Gemma-2-2B with Gemma Scope 16k SAEs (summary). In the gender-bias task, the training data makes "doctors are always male and nurses are always female", and the test data inverts this. In 16 double multiple-choice tasks, each question joins two sub-questions from different datasets, and the training answers are both right or both wrong (summary).
- **Baselines.** Random orthogonal vectors, random PCs or latents, and the top PCs or latents. Also ablation at test time only, and a skyline that adds benign training data (summary). For the top baseline, "we choose the top 5 PCs for each layer."
- **Statistics.** Emergent misalignment uses 5 seeds, and its error bars "show the full range". For the multiple-choice tasks, "we average all results over five seeds." Elsewhere the error bars are the standard error of the mean across seeds.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Emergent misalignment, CAFT with PCA | "It reduces misalignment rates from 7.0% to 0.51% for Qwen and from 6.6% to 1.2% for Mistral." |
| Emergent misalignment, Qwen, insecure-code rate (Table 1) | 84.8% vulnerable with CAFT and PCA, against 91.6% for the model fine-tuned without CAFT. This came from a table. The text calls it "a small drop in performance on the intended task" |
| Emergent misalignment, CAFT with SAEs | 2.4% misaligned for Qwen and 4.6% for Mistral (summary). The text says "SAEs struggle to find unintended concepts" here |
| Emergent misalignment, random PCs or latents | "The results have high variance and do not achieve consistent misalignment and vulnerability rates." For Mistral, random SAE latents did about as well as the chosen ones (summary) |
| Emergent misalignment, skyline | "Adding just 100 benign completions (2% of the training data) reduces misalignment to 0.5%" |
| Gender bias and 16 double multiple-choice tasks, CAFT with SAEs | "On gender bias and 10/16 double multiple choice tasks, CAFT with SAEs both improves generalization" over the baseline, with test accuracy above 50% |
| The same tasks, CAFT with PCA | Above 50% test accuracy on gender bias and 5 of 16 tasks (summary) |
| Ablation at test time only | It does "similarly but overall worse than CAFT" |
| Training accuracy, multiple-choice tasks | At least 97% (summary) |

## Where the claims go beyond the evidence

**The abstract says "without degrading", and the table shows a drop.** For Qwen, the insecure-code rate falls from 91.6% to 84.8%. The introduction is more careful: "with only a small loss of fine-tuning task performance".

**"10x" is an average of two different results.** The PCA numbers give about 14 times less misalignment for Qwen and about 5.5 times less for Mistral. The SAE results are much weaker, at 2.4% and 4.6%.

**A person chooses the directions after seeing them.** A person reads each PC or latent and decides that it carries an unwanted concept. The authors state this limit: "One limitation of our approach is the need for humans to interpret directions." The automatic interpreter in Appendix C is "a proof of concept", and it covers only a part of the multiple-choice tasks.

**The random baseline is close in one setting.** For Mistral, random SAE latents did about as well as the chosen ones. So in that setting, the choice of concept may not be the cause of the effect.

**A simple data fix does as well.** One hundred benign completions bring misalignment to 0.5%. That equals CAFT with PCA on Qwen. CAFT is for the case where you cannot write such data, and the paper does not test that case directly.

**Small samples and few tasks.** Each result has five seeds. There are two large models for one task, and one small model for the multiple-choice tasks. We found no correction for the 17 multiple-choice comparisons.

**Credit.** The authors report their failures. "While CAFT is successful at steering generalization in most tasks, it fails to do so in others." They name the entangled pronoun and verb tasks as a failure. They also say that "more powerful interpretability techniques might be necessary to fully isolate concepts". They give a random baseline, a top baseline and a test-time baseline, and they release code.

## Use in this repo

**The repo has the paper's test-time baseline, and not CAFT itself.** `ablate_pole` in `openinterpretability/harness/hooks.py` and `mean_ablation` in `openinterpretability/explain.py` remove a unit at inference. That is the paper's "test-time only" baseline, applied to native neuron poles and heads. The package has no fine-tuning loop. CAFT needs one, because it ablates during training and not at inference.

**Our units are a natural basis for CAFT.** CAFT projects out PCA or SAE directions. A VERIFIED pole is one neuron's positive or negative side, so ablating it is a projection onto one axis. A VERIFIED answer route also names heads. So the repo could choose the units to ablate from its own records, in place of a person reading PCs. That would replace the paper's main stated limit, the person who chooses, with a reviewed explanation.

**#4781 is not the paper's kind of failure.** CAFT prevents a model from learning a wrong generalisation during fine-tuning. The #4781 failure is in the base model before any fine-tuning. EleutherAI/pythia-70m answers " days" to "A week has seven days and a year has twelve". The VERIFIED route copies the first clause's unit through layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. Its necessity is 0.68 and its sufficiency 0.62 (`changelog.d/4781.added.md`). A CAFT-style fix would fine-tune on "a year has twelve months" data with the copy route ablated, so that the model learns another route. After training, the copy heads run again at inference. The paper does not test whether a route that comes back at inference can still win. This is our reading, not a result from the paper.

**The random null matters here as it does in the paper.** The paper's Mistral SAE result shows that random directions can do as well as chosen ones. The repo already compares each explanation with matched random units. On the #4781 prompt, coverage is 93.8% against 10.1% for matched random units. Any fix that uses our units must keep the same comparison: ablate matched random units during the same fine-tuning.

**What a fix must report, from the paper's metrics.** The paper reports two things together: the rate of the unwanted behaviour, and the rate of the task that the fine-tuning teaches. For #4781, the other five prompts in `docs/analyses/hallucination-shortlist.jsonl` copy the first clause's unit too. They could be the out-of-distribution set, if the fine-tuning data uses only one of them. The `neuron-steering-tests` skill already reads what the model writes with a pole ablated. How to combine these is a decision for the fix campaign, which the maintainer designs.

**Follow-up issues.** None. A fine-tuning loop and the choice of fix method are parts of the campaign design.

**Question for the maintainer.** Should the fix campaign add a fine-tuning method like CAFT, which ablates VERIFIED units during training, beside fixes at inference? It would need training code, which the package does not have. This page records the question and opens no issue.
