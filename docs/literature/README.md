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
| [Evaluating Neuron Explanations](evaluating-neuron-explanations.md) (Oikarinen, Yan and Weng, 2025) | used | Two sanity tests for explanation metrics, which `scripts/score_sanity.py` runs on our scores. Balanced accuracy fails the extra-labels test on rare concepts, so `openinterp score` may use it only because its sample is balanced. |
| [Faithful and Stable Neuron Explanations](faithful-stable-explanations.md) (Yan, Oikarinen and Weng, 2025) | not applicable | Sampling bounds and a bootstrap for vision-neuron explanations. Our P(useful) already carries the probe-count uncertainty: with 20 hard negatives, the paper's rate gives ±0.30. Replicate reviews test stability. |
| [Causally Grounded Explanations](causally-grounded-explanations.md) (Mahale, 2026) | not applicable | An MSc thesis on six IOI heads in GPT-2 small, with no held-out tests. Its sufficiency is 100% by construction, which shows why our causal scores rank against matched neurons. |

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
| [Many Transcoder Features Are Neurons in Disguise](neurons-in-disguise.md) (Shabalin and Piotrowski, 2026) | planned | On gated models, 14–59% of late-layer active transcoder features align with one neuron, and only those match the MLP across training runs. Our GPT-2 transcoder baseline may partly compare neurons with neurons (#3608). |
| [Neurons Speak in Ranges](neurons-speak-in-ranges.md) (Haider, Rizwan, Sajjad, Ju and Siddique, 2026) | not applicable | Concepts occupy distinct activation ranges of residual-stream dimensions, and masking only the range spares other concepts. Not MLP neurons, but range-split modes are one form for #3215. |
| [Neurons as an Interpretable Basis](neurons-as-an-interpretable-basis.md) (Tigges, Kowal, McGrath, Lewis and Cammarata, 2026) | planned | The companion to the essay behind `native-neuron-audit`, on Qwen3.6-35B-A3B. 47% of poles have one mode. The double-negative SwiGLU regime is a third of firings but rare in top examples. Its mode split is a template for #3215. |

## Philosophy of mind

| Page | Status | Why |
| --- | --- | --- |
| [Levin's Ingressing Minds from Platonic Space](levins-ingressing-minds-from-platonic-space.md) (Closer To Truth, on Levin, 2025) | not applicable | A theory of consciousness with no method or result to apply to a model. Its "Platonic space" has nothing to do with the Platonic Representation Hypothesis. |
