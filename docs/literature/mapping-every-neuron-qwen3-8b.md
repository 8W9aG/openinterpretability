# Mapping Every Neuron in Qwen3-8B

**Citation.** Fermion Research. "Mapping every neuron in Qwen3-8B". Company blog post, 18 November 2026. [Post](https://fermionresearch.com/research/mapping-every-neuron-in-qwen3-8b/). No author is named, and the post says "We don't write research papers." It links four JSON data files (`atlas-meta.json`, `atlas-switches.json`, `atlas-stars.json`, `playground.json`). The framework is not released: the post says it will be open-sourced "in the near future". We saw no funding, compute or tool-use statement. The research digest that listed it (#6012) gave 16 November, and the post's own citation line gives 18 November.

**Status.** Planned. #6032 would use its push-and-judge method on a registry model, to compare what a neuron does when pushed with its VERIFIED reading. No code, skill or protocol here depends on it now.

**Read.** On 2026-11-22 a `web-fetch` subagent read the post in two parts, all of its text with the figure captions. It read `atlas-meta.json` in full and the first 40,000 characters of each of the other three files. A model read the source for us. We didn't see the figures, the interactive atlas or the playground, the rest of the three large JSON files, or the linked speech-model post. Every number below comes from the text or the files that we read.

## Summary

**Claim.** Pushing each of Qwen3-8B's 442,368 MLP neurons one at a time, in both directions, finds a few thousand "switches". Each switch reliably turns one behaviour on, such as a language, a persona or a topic. Random neurons of its layer, pushed as hard, do far less.

**Method.**

- **Model.** "Qwen3-8B", with no variant named. The replies read as chat-tuned, and a separate test uses "the pretrained Qwen3-8B", so the main model is probably the chat model.
- **Unit.** "the activated gate projection multiplied by the up projection, which feeds the down projection".
- **Push.** "A push adds a constant to it at every token under greedy decoding." `playground.json` holds 12 push sizes from −48 to +48 in steps of 8, plus no push, with replies of 80 new tokens. The post doesn't say whether the prompt's tokens are pushed too.
- **Screen.** Every neuron was pushed in both directions, and the post measured "how much each push changed what the model was about to say". `atlas-meta.json` holds a `kl_max` field, which suggests a KL divergence, but the post doesn't name the measure.
- **Judge.** "Qwen3-32B judges each reply for the behaviour and for fluency." Three judges, not named, vote in a re-test.
- **Prompts.** 30 questions for the headline cards, 15 in `playground.json`, screens of 32 and 93 prompts, and 150 new questions for the re-test.
- **Controls.** Each reported neuron "had to beat 30 random neurons from the same layer, pushed exactly as hard". It also had to "keep working on questions it wasn't picked on". Random neurons move a behaviour "by 1.4 points out of 100 on average", and a switch must move it "by at least 40". Switches are tested again on the held-out half of the questions.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Atlas | 132 verified switches, 20 persona finds, 11,079 more neurons with a described effect |
| Kinds of the 132 switches | word obsession 62, language 21, topic 16, style and register 11, persona 8, format 7, mood and tone 7 |
| German, L5/N5224, 30 questions | German in 7% of replies, then 100% in one notch |
| Pirate, L9/N7601 | 0% in character with no push, 100% at best, matched random neurons at most 0% |
| Re-test, 150 new questions, three judges | 93 kept a real effect, and a random neuron passes about once in 2,200 tries |
| Described neurons that "just break it" | 3,905 of 11,079 |
| Census, 1,000 random neurons in each of 14 models | about 5,000 real switches in Qwen3-8B (95% CI about 2,300 to 8,200) |
| Same, with 93 more questions | about 6,600 (95% CI about 3,900 to 10,900) |
| Switches tested on the pretrained Qwen3-8B | 67 of 70 still beat random neurons |
| Lying, flattery, manipulation, sycophancy: best candidate against best random | +3/+0, +3/+3, +3/+3, +27/+13 |
| Twelve Chinese-language neurons pushed together | no better than the best single one |
| About 140 reasoning neurons switched off together | problem-solving "within a few points" of baseline |

## Where the claims go beyond the evidence

- **The judge defines the behaviour.** A single judge model, Qwen3-32B, decides whether each reply shows the behaviour. The post reports no agreement between judges or with people, except in the three-judge re-test.
- **Inconsistent counts.** The headline control uses 30 random neurons, but a figure caption uses 24, and `atlas-meta.json` holds 24 and 48. The post says 22 new switches took the total from 110 to 132, and the file says 25. The layer table sums to 28 in layers 5–11 and 51 in layers 25–29, against the post's 25 and 50. The census interval in the post (2,300–8,200) matches neither interval in the file (2,646–8,533 and 2,117–10,565).
- **No seeds and no correction.** Pushing 442,368 neurons in two directions is close to a million tests. The held-out half and the re-test guard against selection, but the post states no correction for multiple comparisons. Only the census gives intervals.
- **Unreleased code, and some neurons withheld.** The framework is not released. The post says some of the strongest neurons are "withheld under a rule". `atlas-meta.json` records 34 dropped for harm and 0 withheld, which doesn't settle what was withheld.
- **Credit.** Every reported switch had to beat matched random neurons pushed as hard. The authors tested again on new questions, with new judges, and on the pretrained model. They report their negative results too: no honesty neuron, no gain from pushing a group, and mostly unclean middle layers.

## Use in this repo

- **The same unit, read from the other side.** The post pushes the input to the down projection, which is the unit that the registry hooks for Qwen models (`harness/hooks.py`, `mlp.down_proj`). The registry reads a pole from its top activations, then tests a stated rule with probes. The post starts from what a push makes the model write. These are two readings of one unit, and the repo has no study of how often they agree.
- **The controls match the registry's.** A switch must beat 30 random neurons of its own layer, pushed as hard, on held-out questions. The registry asks the same of an explanation. Its prompt coverage is against random units of the same kinds (`explain_coverage.py`). Its explanation scores are against explanation-free rules. The post's ratio is large: 1.4 points for random neurons against at least 40 for a switch.
- **Steering is a skill here, not a gate.** `neuron-steering-tests` clamps a pole and reads what the model writes, but no verdict depends on it. The post's thresholds and random-neuron control are a worked design for such a gate. Making steering a gate would change the decision rule, so it needs a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). That is a question for the maintainer, not an issue.
- **Ranking units for the planner.** The planner ranks poles by their effect on the loss (`pole-effects.md`, attribution patching). A push census ranks by effect on generation. The two measures could disagree for units whose effect shows only in long text, such as a persona. A full census of Qwen2.5-0.5B's 116,736 neurons, with 12 pushes and 30 prompts each, would be about 42 million replies. At 80 new tokens each, that is about 3.4 billion generated tokens. That is too much for this machine. A pilot on the VERIFIED units would cost far less, and #6032 holds it.
- **Middle layers.** The post finds the middle of the model mostly unclean: "almost none of those effects are about one thing". That is a claim about generation under a push, not about activations. So it doesn't contradict a VERIFIED reading of a middle-layer pole, which states when the pole fires, not what pushing it does.
