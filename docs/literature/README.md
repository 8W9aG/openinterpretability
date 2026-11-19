# Literature review

One page per paper or post. Each has its citation, a summary written from the source, and a separate discussion of how this repo can or can't use it. The issue *A literature review in docs/literature* sets the scope. Papers and posts waiting for a page are in [queue.md](queue.md). The `literature-review` skill (`.claude/skills/literature-review/SKILL.md`) gives the procedure and the form of a page.

Status marks how the work relates to this repo:

- **used**: code, a skill or the protocol depends on it.
- **planned**: an open issue would use it.
- **not applicable**: reviewed, and the repo has no use for it. The page says why.

## Agents and context management

| Page | Status | Why |
| --- | --- | --- |
| [Context Language Models](context-language-models.md) (Shao et al., 2026) | not applicable | Agent context management, with no look inside any model. Its skill-selection loop shows what the DSPy prompt-optimisation item must avoid: selection on one seed. |
| [Recursive Language Models](recursive-language-models.md) (Zhang, Kraska and Khattab, 2025) | not applicable | A scaffold for prompts longer than the context window, with no look inside any model. Our agents' prompts are short, and the proposer already reads its large evidence file from disk. |

## Explanation scoring

| Page | Status | Why |
| --- | --- | --- |
| [Scoring Explanations Without Reading Them](scoring-explanations-without-reading-them.md) (Chen, 2026) | used | Evaluation items can be scored by rules that never read the explanation. Our null-explanation control comes from it, and its check found that our fuzzing decoys leak contiguity. |
| [Beyond Top Activations](beyond-top-activations.md) (Oikarinen, Yan, Kulkarni and Weng, 2025) | planned | Evaluating on top activations alone measures only recall. Model-guided importance sampling finds false positives cheaply. Our detection already stratifies active windows, but draws inactive ones uniformly (#3595). |
| [Evaluating Neuron Explanations](evaluating-neuron-explanations.md) (Oikarinen, Yan and Weng, 2025) | used | Two sanity tests for explanation metrics, which `scripts/score_sanity.py` runs on our scores. Balanced accuracy fails the extra-labels test on rare concepts, so `open-interpretability score` may use it only because its sample is balanced. |
| [Faithful and Stable Neuron Explanations](faithful-stable-explanations.md) (Yan, Oikarinen and Weng, 2025) | not applicable | Sampling bounds and a bootstrap for vision-neuron explanations. Our P(useful) already carries the probe-count uncertainty: with 20 hard negatives, the paper's rate gives ±0.30. Replicate reviews test stability. |
| [Robustness of Model-Graded Evaluations](model-graded-robustness.md) (Lermen and Kvapil, 2023) | not applicable | Injected text sways LLM graders of quality. Our verdicts are measured, our scores are checked against measured activations, and prompts fence untrusted text. |
| [Causally Grounded Explanations](causally-grounded-explanations.md) (Mahale, 2026) | not applicable | An MSc thesis on six IOI heads in GPT-2 small, with no held-out tests. Its sufficiency is 100% by construction, which shows why our causal scores rank against matched neurons. |
| [Understanding AI from the Inside](understanding-ai-from-the-inside.md) (Barez, Wu et al., 2026) | planned | A 183-page survey with an evaluation protocol and guardrails for automated interpretability. The registry already has most of both. Its missing guardrail, a blind human audit of a random sample of verdicts, is #4034. |

## Model architecture

| Page | Status | Why |
| --- | --- | --- |
| [Conditional Memory via Scalable Lookup](conditional-memory-via-scalable-lookup.md) (Cheng et al., 2026) | not applicable | A new architecture (hashed N-gram memory beside MoE) with no released weights. Its premise, that early layers rebuild local patterns, is a question the registry's early-layer poles could answer. |

## Forecasting and evaluation

| Page | Status | Why |
| --- | --- | --- |
| [Training LLMs to Predict World Events](training-llms-to-predict-world-events.md) (Jeen, Aitchison and Mantic, 2026) | not applicable | No interpretability content. Its fine-tuned model is not one the registry can study, though two evaluation habits carry over. |

## Probing

| Page | Status | Why |
| --- | --- | --- |
| [Activation Probes Are Reliable With a Handful of Positive Examples](activation-probes-few-positives.md) (Tyagi and Heimersheim, 2026) | not applicable | Train on all the negatives, with a class-weighted loss, when positives are rare. `probe-compare` already does. Our thin probe rows lack held-out positives, which the paper does not address. |
| [Is a Linear Probe Evidence of a Linear Representation?](linear-probe-evidence.md) (Kutluozen, Internò and Klindt, 2026) | planned | A random feature map also decodes factors linearly, so probe accuracy needs a random-init control. Our verdicts have one (the untrained null). `probe-compare`'s dense probes don't yet (#3559). |

## Neurons, circuits and dictionaries

| Page | Status | Why |
| --- | --- | --- |
| [Language Model Circuits Are Sparse in the Neuron Basis](neuron-basis-circuits.md) (Arora, Wu, Steinhardt and Schwettmann, 2026) | planned | Task circuits of about 10² MLP neurons, the protocol's unit, control Llama 3.1 8B's behaviour. Its RelP attribution is a candidate for our wiring and attribution selection (#3582). |
| [Many Transcoder Features Are Neurons in Disguise](neurons-in-disguise.md) (Shabalin and Piotrowski, 2026) | used | On gated models, 14–59% of late-layer active transcoder features align with one neuron, and only those match the MLP across training runs. On GPT-2, 1 of our 40 audited latents is one (`docs/analyses/neuron-features.md`, #3608). |
| [What Survives Distillation?](what-survives-distillation.md) (Faumi and Sumanathilaka, 2026) | planned | DistilGPT-2 keeps GPT-2's early circuits and loses its late ones, by heads, CKA and SAE features. No neurons are studied. It makes DistilGPT-2 a cheap sibling for the sister search (#3853). |
| [Neurons Speak in Ranges](neurons-speak-in-ranges.md) (Haider, Rizwan, Sajjad, Ju and Siddique, 2026) | not applicable | Concepts occupy distinct activation ranges of residual-stream dimensions, and masking only the range spares other concepts. Not MLP neurons, but range-split modes are one form for #3215. |
| [Neurons as an Interpretable Basis](neurons-as-an-interpretable-basis.md) (Tigges, Kowal, McGrath, Lewis and Cammarata, 2026) | planned | The companion to the essay behind `native-neuron-audit`, on Qwen3.6-35B-A3B. 47% of poles have one mode. The double-negative SwiGLU regime is a third of firings but rare in top examples. Its mode split is a template for #3215. |
| [Calibrated Concept Profiles](calibrated-concept-profiles.md) (Wilam, 2026) | not applicable | It reads every neuron's response to a named Python construct in OpenCoder-1.5B, which is the opposite direction to ours. Its twin draws, nulls and untrained reference match controls we run. Its untrained probe result (0.80) supports #3559. |
| [Interpreting Any Residual Stream Direction by Scaling Activation Inversion](activation-inversion.md) (De Schamphelaere et al., 2026) | not applicable | A 27B generator, trained for one model, writes text that drives a chosen residual direction. No generator exists for our models. It reports EPO, our dreaming optimiser, far below it, so dreaming's miss rate is a floor (#3779). |

## Fixing models: editing and where it fails

Part of the review of how to fix a model's behaviour (#4784). These pages report evidence for the fix campaign, which the maintainer designs. They do not design it.

| Page | Status | Why |
| --- | --- | --- |
| [Does Localization Inform Editing?](does-localization-inform-editing.md) (Hase, Bansal, Kim and Ghandeharioun, 2023) | planned | Causal Tracing does not predict the best layer to edit a fact. The edit layer explains 94.7% of the variance in ROME's success, and tracing adds 0.1%. It tests the premise of a fix guided by an explanation (#4784, #5015). |
| [The Mirage of Model Editing](mirage-of-model-editing.md) (Yang et al., 2025) | not applicable | Teacher forcing and truncation inflate editing scores. Llama-2 single-edit success is 38.5% under free decoding, and sequential editing collapses by 1,000 edits. Our answer measures read one token, so a fix to #4781 also needs a free-decoded check. |
| [Editing Factual Knowledge in Language Models](editing-factual-knowledge.md) (De Cao, Aziz and Titov, 2021) | planned | A hyper-network edits one fact under a KL constraint on other outputs. Its success, retain, paraphrase and deterioration scores are a ready test for a fix of the #4781 prompt (#5015). |
| [Fast Model Editing at Scale](fast-model-editing.md) (Mitchell, Lin, Bosselut, Finn and Manning, 2022) | not applicable | MEND turns one example's gradient into a weight edit, and checks reliability, generality and locality. Its random locality inputs are too easy. The shortlist prompts that copying answers right are harder negatives for a #4781 fix. |
| [Memory-Based Model Editing at Scale](memory-based-model-editing.md) (Mitchell, Lin, Bosselut, Manning and Finn, 2022) | not applicable | A black-box editor that keeps edits in a memory and leaves every unit unchanged. Its hard out-of-scope test still bears on a fix: three shortlist prompts are right because the model copies the first unit. |
| [Aging with GRACE](aging-with-grace.md) (Hartvigsen et al., 2023) | not applicable | A codebook at one layer swaps that layer's output inside a key's radius, with no change to the weights. It uses no explanation, so it could be a control for the fix campaign. It reports generality in figures only. |
| [WISE: Knowledge Memory for Lifelong Model Editing](wise-lifelong-editing.md) (Wang, Li, Zhang et al., 2024) | not applicable | Sequential edits go in a gated side copy of one FFN matrix. It scores reliability, generalisation and locality together, and studies no units. Its untested near-edit locality case bears on a fix. |
| [Evaluating the Ripple Effects of Knowledge Editing](ripple-effects-of-knowledge-editing.md) (Cohen, Biran, Yoran, Globerson and Geva, 2024) | planned | An edit must also change the facts it implies. MEND, ROME and MEMIT fail most of these tests, and an in-context baseline does best. A #4781 fix needs such neighbour prompts (#5015). |
| [MQuAKE](mquake.md) (Zhong, Wu, Manning, Potts and Chen, 2023) | planned | Editors that recall over 88% of edited facts answer only 0.3–16.0% of the multi-hop questions that follow. So a test on the edited prompt alone overstates a fix. Its single-fact control bears on #5015. |
| [Editing Large Language Models: Problems, Methods, and Opportunities](editing-llms-survey.md) (Yao et al., 2023) | planned | A same-conditions comparison of about a dozen editing methods, on four properties. Single-neuron edits did poorly on GPT-J. Its Distract-Neighbour test is the copy habit behind #4781 (#5015). |

## Fixing models: why models give wrong answers

Part of the review of how to fix a model's behaviour (#4784). These pages read the #4781 failure, where pythia-70m copies " days" from the first clause, in each paper's terms. They report evidence for #5015 and do not design the fix campaign.

| Page | Status | Why |
| --- | --- | --- |
| [In-context Learning and Induction Heads](induction-heads-in-context-learning.md) (Olsson, Elhage, Nanda, Joseph et al., 2022) | used | Our head profiles measure its prefix-matching pattern on repeated random tokens, but not its copying score. Its fuzzy [A*][B*]…[A]→[B*] copy predicts the pythia-70m " days" answer of #4781, including the cost when the true continuation differs. |
| [Copy Suppression](copy-suppression-head.md) (McDougall, Conmy, Rushing, McGrath and Nanda, 2023) | planned | GPT-2 Small's L10H7 lowers in-context tokens that earlier layers predict, and CSPA explains 76.9% of its effect. In pythia-70m only L4/H1 has this OV signature, and it is too weak and too early to stop the #4781 copy (#5015). |
| [Characterizing Mechanisms for Factual Recall in Language Models](factual-recall-mechanisms.md) (Yu, Merullo and Pavlick, 2023) | planned | Attention heads in Pythia and GPT-2 promote the memorized or the in-context answer, and scaling one head's values switches between them. It is the closest known mechanism to the #4781 copy failure, and a value-scaling sweep on the copy heads would test #5015's first question. |
| [Dissecting Recall of Factual Associations](dissecting-factual-recall.md) (Geva, Bastings, Filippova and Globerson, 2023) | planned | Factual recall runs in three steps: early MLPs enrich the subject, attention moves the relation, and heads extract the attribute. It gives #5015 two places to look for the missing " months", and two tools the repo lacks: edge knockout and a head vocabulary lens. |
| [Transformer Feed-Forward Layers Are Key-Value Memories](ffn-key-value-memories.md) (Geva, Schuster, Berant and Levy, 2021) | used | The source of the key-value view in the `vocab-projection-lookahead` skill: keys detect prefix patterns, and values push next tokens. All of it is observational, from one 16-layer model with a uniform-random null. The page reads the #4781 " months" gap in its terms. |
| [Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations](non-factual-hallucination-mechanisms.md) (Yu, Cao, Cheung and Dong, 2024) | not applicable | Factual errors split into two types: lower MLPs fail to retrieve the answer, or upper attention fails to select it. Its errors come from recall, not in-context copying. Its logit-lens rank test could still sort the " days" failure (#5015). |
| [Do I Know This Entity?](knowledge-awareness-entities.md) (Ferrando, Obeso, Rajamanoharan and Nanda, 2025) | planned | SAE directions at an entity's last token signal whether Gemma 2 recognises it, and they gate how much attribute-extraction heads attend to it. That gives #5015 a test: do pythia-70m's copy heads attend to " days" because nothing at " year" recalls " months"? |
| [On the Biology of a Large Language Model](biology-of-a-large-language-model.md) (Lindsey, Gurnee, Ameisen et al., 2025) | planned | In Claude 3.5 Haiku, "can't answer" features are on by default and "known entity" features inhibit them, and a misfire gives some hallucinations. Its stated limits (copy completions, attention circuits, "why not X") match the #4781 copy failure. Its known/unknown contrast is a method for #5015. |
| [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](fine-tuning-new-knowledge.md) (Gekhman et al., 2024) | planned | Fine-tuning on facts a model does not know is learned slowly and then makes it wrong on other answers. Its few-shot knowledge test (SliCK) would show whether pythia-70m knows "twelve months" before any fix by training (#5015). |
| [Why Language Models Hallucinate](why-language-models-hallucinate.md) (Kalai, Nachum, Vempala and Zhang, 2025) | planned | A theory that wrong answers are forced classification errors, made in pretraining and kept by benchmarks that give no credit for "I don't know". It names no units. Its "poor model" factor frames the copy-the-first-unit failure and predicts a scale test (#5015). |

## Philosophy of mind

| Page | Status | Why |
| --- | --- | --- |
| [Levin's Ingressing Minds from Platonic Space](levins-ingressing-minds-from-platonic-space.md) (Closer To Truth, on Levin, 2025) | not applicable | A theory of consciousness with no method or result to apply to a model. Its "Platonic space" has nothing to do with the Platonic Representation Hypothesis. |
