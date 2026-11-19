# Mechanistic Unlearning

**Citation.** Phillip Guo and Aaquib Syed (University of Maryland), Abhay Sheshadri (Georgia Institute of Technology), Aidan Ewart (University of Bristol) and Gintare Karolina Dziugaite (Google DeepMind). The exact title is `Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization`, arXiv 2410.12949. Version 1 is from 16 October 2024. We read version 2, from 5 December 2024. [Paper](https://arxiv.org/abs/2410.12949). The abstract page gives no code or data link, and we did not find one in the text that we read. The arXiv comment names no venue. Issue #5042 gives the venue as ICML 2025, PMLR 267:20964–20992. We did not check this on the proceedings page: the one PMLR page that we tried was a different paper. The acknowledgements say: "This project used compute provided by the Center for AI Safety". The authors did a part of the work in the ML Alignment & Theory Scholars Program. We saw no funding statement.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it. The maintainer designs that campaign. The paper tests the premise of [Does Localization Inform Editing?](does-localization-inform-editing.md) again, and finds that one kind of localization does help an edit.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v2 HTML for us, in two parts by offset, so it saw the full text. It returned short verbatim quotes and the full Table 1. It returned the rest as its summary: the abstract, the method, Table 3, the CounterFact results and the probe results. It quoted short parts of the soft-prompt appendix. We did not see the figures or the PDF. The model found no limitations or conclusion section. It found no seeds, confidence intervals or significance tests.

## Summary

**Claim.** An edit or unlearning step resists relearning better, and holds in more formats, when it changes only the components that look up a fact. It does less well when it changes the components that a method picks from their effect on the output. The abstract's words are "localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall".

**Method** (summary, except the quotes).

- **Models.** Gemma-7B, Gemma-2-9B and Llama-3-8b. The authors left out Pythia-2.8B and GPT-2, because they wanted larger models with measurable general capabilities.
- **Data.** Sports Facts, with 1,567 athletes, from Nanda et al. (2023). CounterFact, from Meng et al. The authors keep the facts to which the model gives a probability above 50%.
- **Tasks.** Editing the sport of 16 or 64 athletes, editing all sports, and unlearning all basketball athletes. Editing 16 or 64 CounterFact facts, and editing 64 CounterFact facts in four sets of 16.
- **Output-Tracing (OT) localizations.** The authors define them as "localizations which measure causal effects of components on only the output". They test Causal Tracing and attribution patching, and a form of each that keeps only MLPs.
- **Fact-lookup (FLU) localization.** For Sports Facts, "we use manually derived localizations for MLP layers". The FLU layers are "the layers at which the probe accuracies rapidly increase". In Gemma-7B, the accuracies "increase in accuracy significantly in layers 2 through 7". For CounterFact, "we first use path patching from Goldowsky-Dill et al. (2023)", then "We take the MLPs with the highest change".
- **Baselines.** No localization, random components, all MLPs and random MLPs. The authors match the parameter count across OT, FLU and random localizations.
- **Edit.** Fine-tuning of the localized weights only, with injection, retain and Pile SFT losses, for 50 iterations. The learning rate comes from a sweep on an objective that leaves out the robustness metrics.
- **Metrics.** Forget, edit and retain accuracy on the standard prompts. MMLU, as a measure of side effects. Multiple-choice (MCQ) accuracy, a second format of the same question. Paraphrase and neighborhood checks for CounterFact. Relearning: LoRA training on half of the forget set, then a test on the other half. Linear probes on each layer, for latent knowledge. Soft prompts, trained on half of the forget set.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Editing the sport of athletes, MCQ format | FLU edits generalize to MCQ, "exceeding other localization methods by more than 40% in MCQ Edit Accuracy" |
| Unlearning all basketball athletes (Table 1), forget / retain / MCQ / MMLU | FLU 0.002 / 0.995 / 0.110 / 0.613 |
| The same, Causal Tracing | 0.201 / 0.998 / 0.849 / 0.611 |
| The same, attribution patching | 0.000 / 1.000 / 0.767 / 0.602 |
| The same, all MLPs | 0.000 / 0.994 / 0.279 / 0.606 |
| The same, no localization | 0.000 / 0.985 / 0.196 / 0.595 |
| The same, random components | 0.952 / 0.980 / 0.822 / 0.612 |
| Relearning after athlete editing, OT localizations | Relearning "can recover a significant portion, as much as 63%, of the supposedly forgotten information" |
| The same, FLU | "with only about 20% of the forgotten information recovered" |
| Relearning after CounterFact editing | "none of the localizations relearn more than 7% accuracy". The authors say that this test is not strong enough for CounterFact |
| Soft prompts, Gemma-2-9B, athlete editing | Soft prompts "recover over 60% Forget Accuracy on OT localizations", "while FLU, Nonlocalized, and All MLPs remain under 40% Forget Accuracy" |
| Soft prompts, other tasks and models | "We find limited soft prompt success" |

In Table 1, a lower MCQ value is better: it is the share of MCQ questions on which the model still gives the forgotten sport. Causal Tracing and attribution patching remove the fact on the standard prompt. But in the MCQ format the model still gives it 85% and 77% of the time.

Two more results came back only as the model's summary, so we do not give their numbers as verified. The probes find the original answer near 100% in early layers after an OT edit, and least often after an FLU edit. A weight-mask analysis (Table 3) finds that attribution patching masks weights in the extraction components, and FLU masks none there.

The Discussion says that the work "suggests unlearning/editing as a potential testbed for different interpretability methods".

## Where the claims go beyond the evidence

**The FLU localization is chosen by hand.** For Sports Facts, a person reads the probe curves and picks the layers. For CounterFact, a person picks the MLPs with the highest path-patching change. The OT localizations come from a fixed rule with a threshold. So the comparison is between an expert reading and a rule, not between two rules. The authors state that the FLU localization was "manually derived".

**The weight-mask analysis can not test FLU.** FLU localizes the lookup MLPs by definition, so FLU can not put weights outside them. The model's summary says that the authors note this themselves.

**No seeds or intervals.** We found no seeds, confidence intervals or significance tests. Many results are averages over three models, or over the 16 and 64 fact settings. So we can not tell how much of the 40% MCQ gap is noise.

**The robustness tests are weak in some settings.** The authors say that relearning is not a valid test when the edit changes all sports, because the sports facts are not independent. On CounterFact, relearning brings back at most 7% for every localization, so that test does not separate them. Soft prompts give a clear gap in one model and one task only.

**The baselines are strong.** In Table 1, "all MLPs" and "no localization" give MCQ 0.279 and 0.196. FLU gives 0.110. So the large gap is between FLU and OT, not between FLU and editing every MLP. The authors report that MLP-only baselines are competitive on some relearning results.

**One kind of knowledge.** All facts are subject-attribute facts, as in the Hase et al. paper. The paper does not test a computation such as copying.

**Credit.** The paper measures robustness in several formats, which most editing papers do not. It matches the parameter count across localizations, and it chooses the learning rate on an objective without the robustness metrics. It names its own weak tests.

## Use in this repo

**The paper qualifies the Hase et al. result that the repo already records.** [Does Localization Inform Editing?](does-localization-inform-editing.md) found that Causal Tracing does not predict the best edit layer. This paper agrees that Causal Tracing is a poor guide. It finds that a localization of the mechanism, by probes or path patching, does give a better edit. So the question for the fix campaign is not "does localization help". The question is which localization the campaign uses.

**Our answer routes are closer to the paper's FLU than to OT.** An answer experiment (`openinterpretability/core/answer_records.py`) tests a route with necessity, sufficiency and path patching, against matched random sets. It also tests counterfactual prompts. Causal Tracing scores a component only by its effect on the output. The VERIFIED #4781 route has necessity 0.68, sufficiency 0.62 and patching 0.75, and random sets give −0.004 (`changelog.d/4781.added.md`). Our unit scores also use minimal pairs, not Gaussian noise (`activation-patching-practice` skill).

**The paper's split between lookup and extraction maps onto the #4781 route, with a difference.** In the #4781 route, the copy heads L3/H3, L4/H0 and L4/H7 carry " days" to the answer position. That is the extraction step in the paper's terms. The layer-0 count-word and time-unit neurons mark the tokens, which is closer to lookup. The paper predicts that an edit at the extraction step gives a fact that comes back in a second format. But #4781 is not a stored fact that is wrong. No verified unit supplies " months", so there is no lookup to edit. This is a reading of the paper, not a measurement. It bears on question 1 of #5015.

**The paper's checks give a fix experiment more to report.** A fix must work in a second format of the prompt, and it must resist a short relearning step. The other five prompts in `docs/analyses/hallucination-shortlist.jsonl` are a ready set of second prompts, and all of them copy the first clause's unit. The paper also shows that "all MLPs" and "no localization" are strong baselines. A fix guided by a VERIFIED route must beat them, not only beat random units. How to use these checks is for the maintainer, who designs the campaign.

**What the repo can not take from the paper.** The paper uses 7B to 9B models and fine-tuning of whole MLP layers. Our models are smaller, for example pythia-70m, and our units are single neuron poles and heads. The paper has no result on single neurons.

**Question for the maintainer.** Should a fix experiment report a second-format check and a relearning check, with "all MLPs" and "no localization" as baselines, as this paper does? It is a campaign design decision, so this page records it and opens no issue.
