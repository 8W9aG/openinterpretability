# Literature review

One page per paper or post. Each has its citation, a summary written from the source, and a separate discussion of how this repo can or can't use it. The issue *A literature review in docs/literature* sets the scope. Papers and posts waiting for a page are in [queue.md](queue.md).

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
| [Calibrated Concept Profiles](calibrated-concept-profiles.md) (Wilam, 2026) | not applicable | It reads every neuron's response to a named Python construct in OpenCoder-1.5B, which is the opposite direction to ours. Its twin draws, nulls and untrained reference match controls we run. Its untrained probe result (0.80) supports #3559. |

## Philosophy of mind

| Page | Status | Why |
| --- | --- | --- |
| [Levin's Ingressing Minds from Platonic Space](levins-ingressing-minds-from-platonic-space.md) (Closer To Truth, on Levin, 2025) | not applicable | A theory of consciousness with no method or result to apply to a model. Its "Platonic space" has nothing to do with the Platonic Representation Hypothesis. |
