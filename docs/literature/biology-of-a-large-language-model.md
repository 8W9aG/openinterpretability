# On the Biology of a Large Language Model

**Citation.** Jack Lindsey (lead contributor), Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus. Then Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones. Then Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah and Joshua Batson (correspondence). "On the Biology of a Large Language Model". Transformer Circuits Thread, published 27 March 2025. [Paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). The page names Anthropic as the only affiliation, and says that McDougall did the work while at Anthropic. Gurnee, Ameisen, Chen, Pearce, Turner, Citro and Batson are core contributors. The methods are in a companion paper, "Circuit Tracing: Revealing Computational Graphs in Language Models". The page links interactive attribution graphs, but no code and no transcoder weights. Venue: the Transformer Circuits Thread, an Anthropic web publication without peer review. The issue's date and first authors agree with the page.

**Status.** Planned. Issue #5015 asks why pythia-70m never reaches " months". The paper's known/unknown prompt pairs and its "why does the model not do X" limit bear directly on that question. `openinterpretability/explain.py` already cites the companion methods paper for its first-order edges.

**Read.** On 2026-11-18. We downloaded the page's HTML, turned it into text, and read these sections in the source's own text:

- the front matter, the introduction and "A note on our approach and its limitations".
- the method overview and the introductory example (multi-step reasoning, with its inhibition and swap experiments).
- all of "Entity Recognition and Hallucinations", with its four subsections.
- "Commonly Observed Circuit Components and Structure", all of "Limitations", and the Discussion up to "Going Forward".
- the citation information, the open questions and the special tokens.

A web-fetch model also read the first part of the page for the metadata, and it agreed with the text. We did not read these case studies: planning in poems, multilingual circuits, addition, medical diagnoses, refusals, the jailbreak, chain-of-thought faithfulness and the hidden-goal model. We also did not read the outlook, related work, the acknowledgments or the graph-pruning appendix. We searched the author contributions only for single terms. We did not see the figures or the interactive graphs, so no percentage from an intervention plot is here. We did not read the companion methods paper.

## Summary

**Claim.** Attribution graphs over a cross-layer transcoder show intermediate steps inside Claude 3.5 Haiku on single prompts. For hallucinations, the model has "can't answer" features that are active by default in Human/Assistant prompts. "Known answer" and "known entity" features inhibit them. At least some hallucinations are a "misfire" of this inhibition.

**Method.**

- **Model.** Claude 3.5 Haiku, Anthropic's production model of October 2024. The hallucination section also runs it as a base model, without the Human/Assistant format.
- **Units.** A cross-layer transcoder (CLT) with 30 million features over all layers replaces the MLP neurons. Error nodes hold what the CLT does not reconstruct. The original model's attention patterns stay fixed. The paper calls this the "local replacement model".
- **Graphs.** An attribution graph has features as nodes and their direct effects as edges, for one output token. The authors prune it to the important nodes. The authors label features from their top activations, and group them by hand into "supernodes".
- **Checks.** The authors intervene on supernodes in the original model with "constrained patching", at a chosen intervention layer. They inhibit a group by clamping it to "a negative multiple of their original value", or they turn features up. They say that they chose labels and supernodes before they measured the perturbations. They treat the effect on the model's output as "the most important source of validation".
- **Hallucination prompts.** "Which sport does Michael Batkin play?" (an unknown name) against the same question for Michael Jordan. "Name one paper written by" Josh Batson (a refusal) against Andrej Karpathy (a wrong paper).

**Findings, as the source reports them.** The hallucination section reports no numbers. Its results are directions of effect.

| Setting | Result |
| --- | --- |
| Batkin, Human/Assistant | "Can't answer" features outvote a "say a sport" path, and the model apologises. "Assistant" features activate them directly |
| Batkin, the name itself | The name's tokens and a generic "first names" feature activate "unknown name" features, which promote "can't answer" |
| Jordan | "Can't answer" features are "inactive or more weakly active". Jordan features and general "known answer/entity" features inhibit them |
| Batkin, known-answer features turned up | The model names "a seemingly random" sport, "including chess" |
| Jordan, known-answer features inhibited | The model "sometimes" answers with words of uncertainty, such as "Hypothetical" |
| Jordan, unknown-name features turned up | No refusal. The authors call this unexpected, and an instance of mechanistic unfaithfulness |
| Batson, known-answer features turned up | The model hallucinates. More activation gives a stronger effect |
| Karpathy (a wrong paper) | Some known-answer features are active, "albeit weakly" against Jordan. Inhibiting them "substantially" gives a refusal |
| All hallucination steering | The strengths needed are "quite high relative to the feature activations on the original prompts" |
| Dallas → Texas → Austin | Inhibiting "Dallas" lowers "Texas" but not "say a capital". Swapping in another state's features gives that state's capital |
| The method overall | "Satisfying insight for about a quarter of the prompts we've tried" |
| Prompt length | Not scaled past "about a hundred tokens" |

The limitations list six cases where the method fails. Two bear on this repo. One is "Why does the model not do X?", because the graphs do not show inactive features. The other is a completion that "is a copy of a word earlier in the sequence". There the graph shows only "an edge directly from (an input feature for) that word and the model output". The authors add that the graphs do not explain how the model computes its attention patterns.

## Where the claims go beyond the evidence

**Single prompts, with no counts.** The hallucination account rests on four prompt families. The text gives no number of prompts tried, no success rate and no per-prompt effect sizes. The authors say this themselves: the results are "existence proofs", and the cases are "a biased sample".

**The Karpathy case is a conjecture.** The text says "perhaps the model activates" known-answer features. The test inhibits them and produces a refusal. That shows that the features can block the hallucination. It does not show that their weak activation caused it. The Discussion describes this as "some evidence".

**Strong steering.** The doses that change the output are high against the features' own activations. Activating unknown-name features did not cause a refusal. The authors take both as signs that the circuit is "only a part of the story", and they deserve credit for reporting the failure.

**One closed model, with no released weights.** All results are on Claude 3.5 Haiku and an unreleased CLT. A reader cannot repeat the graphs or the interventions. The open questions ask how the mechanisms "manifest in different models".

**Error nodes and hand grouping.** The authors write that "even the graphs we have highlighted contain significant contributions from error nodes". Supernodes are hand-made, and the authors call that step "labor-intensive, subjective". Interventions act on features of the replacement model, not on the model's own units.

**"Meta-cognition" is a label.** The Discussion hedges it well. The circuit may be "simply making a plausible guess" from familiarity, not a model of its own knowledge.

## Use in this repo

**The repo already traces native units, not a replacement model.** `explain` attributes the answer's logit to every MLP neuron pole and every attention head, as (a − mean) · ∂logit/∂a (`docs/experiments.md`, *Attribute*). Its *Paths* step gives first-order edges "as in the attribution graphs of Ameisen et al. (2025)", but with no replacement model (`openinterpretability/explain.py`). So the repo has no error nodes and no CLT to trust. The paper's "mechanistic unfaithfulness" problem applies to its features, not to our units. Learned-basis work is in #3218, #3220 and `docs/literature/neurons-in-disguise.md`.

**Our baseline sees part of what is absent.** The paper cannot show inactive features. `explain` subtracts each unit's corpus mean, so a unit below its usual level has a non-zero attribution. That helps #5015, question 1: a " months" unit that is present but weak would show as a small or negative contribution. A unit that is at its mean, or one with no gradient to the answer, still has none. So the paper's limit holds for us in part.

**The copy limit is the #4781 case.** #4781's answer is a copy: " days" comes from the first clause. The paper says its graphs show only a direct edge from the copied word to the output there. Our verified route goes further, because it attributes heads. It names the copy heads L3/H3, L4/H0 and L4/H7, and layer-0 count-word neurons on " seven" and " twelve". The VERIFIED answer record (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) has sufficiency 0.62 and patching 0.75. Its 8 random-set nulls for patching lie between −0.017 and 0.011. Neither tool explains why the heads attend to " days". That is the paper's "missing attention circuits" limit, and `attention-head-audit` would need QK-side tests to close it.

**The known/unknown pair is a method #5015 can use.** The paper found its inhibitory circuit by "comparing two similar prompts". The route's 7 controls all passed, but each one tests the copy, and none is a prompt where pythia-70m answers " months". #5015 would need such a twin, for example a prompt without the first clause, if the model then gives " months". A contrast of `explain` between the twin and the failing prompt would show which units differ. That is evidence for #5015, question 1, and needs no new issue.

**Steering doses.** The paper needs high doses, and reads that as an incomplete circuit. `neuron-steering-tests` clamps a pole at 1, 2 and 4 × `max_activation`. An answer route that changes the answer only at a high multiple would point to the same gap. That belongs in #4778's answer experiments as a reported number. This review does not propose how to change the answer. The fix campaign is the maintainer's.

**The default-and-inhibit motif may not reach pythia-70m.** The paper's circuit is a refusal default that comes from finetuning the Assistant. Pythia-70m is a base model with no refusal, so the same circuit cannot be the cause there. The base-model example in the paper fits better: "Michael Batkin plays the sport of" → "Pickleball" is a plausible guess with no check. #4781 looks like that guess, with the copy path as the guess and no path that supplies " months".
