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
| [Language Models Can Rewrite Their Own Weights](models-rewrite-own-weights.md) (Fan et al., 2026) | planned | A 27B model edits a direction or clips weight outliers, with no gradients. Its edit gains 6.6 points only after a framework check, and 1.1 without one. No random-direction control, so it bears on #5069 and #4782. |

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

## Fixing models: fixes guided by an explanation

Part of the review of how to fix a model's behaviour (#4784). These pages cover fixes that use an explanation, and the tests that show whether a fix removed a behaviour or only hid it. They report evidence and do not design the fix campaign.

| Page | Status | Why |
| --- | --- | --- |
| [Sparse Feature Circuits](sparse-feature-circuits.md) (Marks, Rager, Michaud, Belinkov, Bau and Mueller, 2025) | planned | SHIFT ablates SAE features that the authors judged irrelevant: Pythia worst-group accuracy goes from 24.4% to 76.0%. Its test design and matched random-ablation null fit the fix campaign (#4784). Its neuron comparison is not matched, so it is no fair test of native neurons. |
| [Mechanistic Unlearning](mechanistic-unlearning.md) (Guo, Syed, Sheshadri, Ewart and Dziugaite, 2024) | planned | It tests Hase et al.'s premise again. Edits at the fact-lookup MLPs hold in a second format and resist relearning better than edits placed by Causal Tracing or attribution patching. Editing all MLPs is still a strong baseline, and the paper reports no seeds (#4784). |
| [Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning](concept-ablation-fine-tuning.md) (Casademunt, Juang, Karvonen, Marks, Rajamanoharan and Nanda, 2025) | planned | CAFT removes interpretable directions during fine-tuning, so the model learns the task without an unwanted generalisation. VERIFIED units could supply those directions, but the repo has only test-time ablation and no training loop (#4784). |
| [Gradient Routing](gradient-routing.md) (Cloud, Goldman-Wetzler, Wybitul, Miller and Turner, 2024) | planned | Masking gradients by data point during training puts a capability in a chosen region, which can then be removed. It cannot apply to the already-trained pythia-70m of #4781. Its retrain attack and its check without the routed tokens are tests that a fix report could use (#4784). |
| [Persona Vectors](persona-vectors.md) (Chen, Arditi, Sleight, Evans and Lindsey, 2025) | planned | Difference-of-means directions track traits. Steering at inference costs MMLU at large coefficients, and steering during fine-tuning keeps it. Its rule, to report the target, coherence and a capability score for each fix, bears on #4784. |
| [Improving Alignment and Robustness with Circuit Breakers](circuit-breakers.md) (Zou, Phan, Wang, Duenas et al., 2024) | planned | Representation Rerouting trains adapters so that harmful representations turn orthogonal, with a retain loss to keep capability. It fixes by stopping a behaviour, not by giving the right answer, and it picks layers by hand, not from an explanation (#4784, #4781). |
| [The WMDP Benchmark and RMU Unlearning](wmdp-rmu-unlearning.md) (Li et al., 2024) | planned | RMU pushes one layer's activations on forget data toward a random vector and keeps others close to the frozen model. It checks the fix with probes and an attack, and its near-topic test shows a loss that overall MMLU hides. That is a baseline and a set of checks for a #4781 fix (#4784). |
| [Eight Methods to Evaluate Unlearning in LLMs](robust-unlearning-evaluation.md) (Lynch, Guo, Ewart, Casper and Hadfield-Menell, 2024) | planned | An unlearned model loses 77% on its target test. Yet in-context examples, a small relearn and linear probes bring the knowledge back, and related domains suffer. A fix for #4781 needs the same checks beyond its own target (#4784). |
| [Distillation Robustifies Unlearning](distillation-robustifies-unlearning.md) (Lee, Foote, Infanger, Shor et al., 2025) | planned | Unlearning by behaviour training only suppresses a capability, and 500 finetuning steps bring it back. Distilling into a fresh or noised model (UNDO) matches a model that never saw the data. It gives a test of whether a fix removed a capability or only hid it (#4784, #5015). |

## Fixing models: activation steering

Part of the review of how to fix a model's behaviour (#4784). These pages cover fixes at inference time by steering, and how to tell a steering effect from what any push of that size does. They report evidence and do not design the fix campaign. The synthesis of all four areas is [Fixing models](fixing-models.md).

| Page | Status | Why |
| --- | --- | --- |
| [Steering Language Models With Activation Engineering](activation-addition.md) (Turner et al., 2023) | used | The steering samples and the `neuron-steering-tests` skill cite it, but they clamp one unit instead of adding a contrast-pair vector. Its random-vector control of the same norm points to a null that our steering samples lack. |
| [Steering Llama 2 via Contrastive Activation Addition](contrastive-activation-addition.md) (Rimsky/Panickssery et al., 2024) | not applicable | Mean-difference residual vectors from contrast pairs shift Llama 2 Chat behaviours, but the paper has no random-vector null and only 50 test items per behaviour. Our steering clamps single units, so the page records evidence and questions for the fix campaign. |
| [Inference-Time Intervention](inference-time-intervention.md) (Li, Patel, Viégas, Pfister and Wattenberg, 2023) | planned | ITI adds a class-mean shift to the top probe-ranked heads (LLaMA-7B TruthfulQA True*Info 30.5 to 43.5). It shows how to steer heads, which our steering samples skip, but it has no random-head null (#4784). |
| [Representation Engineering](representation-engineering.md) (Zou, Phan, Chen et al., 2023) | planned | It reads and steers population-level directions in 7B–70B chat models. Its test found the best-correlating direction changed behaviour least, which the correlation-only `probe-compare` cannot see. Its random-vector control is missing from our steering samples (#3209). |
| [Extracting Latent Steering Vectors from Pretrained Language Models](latent-steering-vectors.md) (Subramani, Suresh and Peters, 2022) | planned | A free vector fitted to one sentence can make GPT-2 117M write it at almost any middle layer. So a steering success neither locates a cause nor fixes more than one prompt. Its mean-difference offset and its gibberish null are the parts a steering test in #4784 could reuse. |
| [Refusal in Language Models Is Mediated by a Single Direction](refusal-single-direction.md) (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda, 2024) | planned | One difference-in-means direction carries refusal in 13 chat models. Its ablation is checked for side effects by KL, loss and benchmarks, but with no random-direction null. Our steering samples have the reverse gap: our units have random nulls, but the steering has no side-effect number (#4784, #3209). |
| [Analyzing the Generalization and Reliability of Steering Vectors](steering-vector-reliability.md) (Tan, Chanin, Lynch et al., 2024) | not applicable | Mean-difference steering vectors move almost half the inputs the wrong way on many datasets, and fail under some prompt changes. It is evidence for testing a fix on all six kept shortlist prompts, and for a per-prompt sign in steering samples (#3209). |
| [Representation Surgery: Theory and Practice of Affine Steering](affine-steering.md) (Singh, Ravfogel, Herzig, Aharoni, Cotterell and Kumaraguru, 2024) | planned | Under a linear-guardedness constraint, the least-squares steering map is a translation by the difference of class means. It gives the theory behind difference-of-means steering (#4784). Its ungated runs double perplexity, which supports reading our 4x clamps as off-distribution. |
| [AxBench](axbench.md) (Wu et al., 2025) | planned | On 500 concepts in Gemma-2, prompting steers best (0.894) and SAE steering scores 0.165. Any steering fix needs a prompting baseline and a fluency score, and our steering samples have neither (#4784, #3209). |
| [Steering Safely or Off a Cliff?](steering-specificity.md) (Goyal and Daumé, 2026) | planned | It splits a steering fix's side effects into general, control and shift checks: steering keeps refusal on harmful queries but loses 8–55% of jailbreak safety. For #4781 the control set already exists: same-frame prompts where copying is right (" legs", " wheels"). |
| [Does Steering Break Your Model?](does-steering-break-your-model.md) (Yang et al., 2026) | planned | On two Gemma 2 models, none of 23 activation-steering methods beats prompt steering on effect against side effects such as fluency. That is a reason to score the side effects of our steering samples, and to compare any steering fix with a prompt (#4784, #3209). |

## Catastrophic interference and forgetting

Part of the review of interference, weight-space programming and low-rank fine-tuning (#4782). These pages ask what a change to a network's weights does to what it already knows.

| Page | Status | Why |
| --- | --- | --- |
| [Why There Are Complementary Learning Systems in the Hippocampus and Neocortex](complementary-learning-systems.md) (McClelland, McNaughton and O'Reilly, 1995) | planned | Slow, interleaved learning finds shared structure, and focused learning of one new fact damages related knowledge where the answers differ. That is the risk for a fix of the #4781 error, and it fits the late changes in the neuron-development paper. |
| [Catastrophic Forgetting, Rehearsal and Pseudorehearsal](rehearsal-and-pseudorehearsal.md) (Robins, 1995) | planned | Rehearsing a few old items with each new one, drawn fresh every epoch, almost removes forgetting in a 32-16-32 network. Rehearsing on random inputs labelled with the network's own outputs keeps much of that benefit. The paper checks only the outputs, so it is no evidence that a VERIFIED pole's explanation survives a change (#4782). |
| [Catastrophic Forgetting in Connectionist Networks](catastrophic-forgetting-connectionist.md) (French, 1999) | planned | The standard 1990s review: shared, overlapping weights cause sudden forgetting, and rehearsal, sparser codes or dual networks reduce it. It frames #4782. Its exact-recognition and savings measures bear on how we count a pole's explanation as lost. |
| [An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks](catastrophic-forgetting-empirical.md) (Goodfellow, Mirza, Xiao, Courville and Bengio, 2013) | planned | Across three task pairs, dropout forgets least, and the best activation function depends on the pair. The results are frontier figures from one run per setting, chosen on test error. Pythia has no dropout, and our base-instruct analysis already measures unit stability under a fine-tune (81–88% of hypotheses keep). |
| [Overcoming Catastrophic Forgetting in Neural Networks](elastic-weight-consolidation.md) (Kirkpatrick et al., 2017) | planned | EWC slows learning on each weight in proportion to its diagonal Fisher, which protects old tasks on permuted MNIST and Atari. The authors find that the diagonal Fisher misjudges which weights don't matter. Our base-instruct runs already show that most VERIFIED poles keep without any penalty (88% and 81%). |
| [Continual Learning Through Synaptic Intelligence](synaptic-intelligence.md) (Zenke, Poole and Ganguli, 2017) | planned | Each weight sums gradient × update over training to rate how important it is. A penalty on moving important weights cuts forgetting on split and permuted MNIST and split CIFAR. Evidence for #4782. The page notes that Pythia-70m's poles keep changing late, so a low importance need not keep an explanation true. |

## Philosophy of mind

| Page | Status | Why |
| --- | --- | --- |
| [Levin's Ingressing Minds from Platonic Space](levins-ingressing-minds-from-platonic-space.md) (Closer To Truth, on Levin, 2025) | not applicable | A theory of consciousness with no method or result to apply to a model. Its "Platonic space" has nothing to do with the Platonic Representation Hypothesis. |
