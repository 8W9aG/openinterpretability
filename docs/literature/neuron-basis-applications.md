# Returning to the Neuron Basis for Interpretability, Part 2: Using Neurons For Safety Applications

**Citation.** Jasmine Shone and Lisa Dunlap (Anthropic). Shone's work was part of the Anthropic Fellows Program. "Returning to the Neuron Basis for Interpretability, Part 2: Using Neurons For Safety Applications". Alignment Science Blog, 23 November 2026. [Post](https://alignment.anthropic.com/2026/neuron-basis-applications/). The post gives no code, no data and no licence. Part 1 is a companion post ([Part 1](neuron-basis-objections.md)).

**Status.** Planned, in #7151. The fix campaign (#5220) must show that a fix removed a failure and did not only hide it. This post's unlearning audit gives two cheap tests of that: cloze scoring, and removing one mean-shift direction.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the post's HTML and turned it into text. It read the whole post verbatim, with the Discussion, the Limitations, Appendices A to D and Tables A1, C1, C2 and D1. It saw the eleven figures only as captions and alt text. So per-set and per-behaviour values that only the figures show are not on this page. It did not read Part 1 or the cited works. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** Neurons chosen from their descriptions and output readouts, with no labels, work as safety tools. They detect harmful text, steer behaviour and expose what a finetune changed, about as well as methods that need training or labels. They also show that most WMDP-cyber unlearning methods "appear to hide hazardous knowledge rather than remove it".

**Method.**

- **Units:** each sign of a SwiGLU neuron, read by its Transluce activation description and its J-lens readout, as in Part 1.
- **Selection:** by embedding similarity to a concept description, or by a Claude agent that reads candidate neurons. The agent sees no labels and no evaluation outputs.
- **Detection:** AxBench Concept500 on Llama-3.1-8B-Instruct. Then 16 safety sets and retrieval over 416,381 Pile documents, on six models. A neuron's safety score is the cosine between the residual stream and its output vector.
- **Steering:** 16 WeirdChat behaviours on Llama-3.1-8B-Instruct. A neuron is steered by adding `s·c·‖rL‖·p·w/‖w‖` to the residual stream at every position, with the dose `c` at 0.1, 0.2 or 0.4. This is the same as adding a constant to the neuron's activation.
- **Finetunes:** seven model organisms, a blind auditing game over 68 LoRA adapters, and "neuron voting" for backdoor triggers in 74 models, from the weights alone.
- **Unlearning:** 23 WMDP-cyber checkpoints from 11 methods. The tests are cloze scoring, removing one direction, resetting the 30 most-changed neurons, and copying their changes into the original model.
- **Baselines:** text search, SAE latents chosen the same way, trained probes and steering vectors, random neuron sets and random directions.
- **Judges:** Claude models judge the steered outputs, the audits and the documents it finds.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| AUROC 0.903, against 0.875 for embedding search and 0.888 for the best SAE baseline. A trained probe and asking the model both reach 0.97 | AxBench Concept500, Llama-3.1-8B-Instruct, 20 neurons per concept |
| Beats text search on 11 of 16 sets, and matches or beats probes trained on 40 labels on 7 | Safety sets, averaged over six models. The baselines are chosen on the test set |
| AUROC 0.49, against 0.95 for a probe trained on 40 labels | WMDP-cyber against security news, the authors' own limitation |
| More harmful documents in the top 50 on 11 topics, and tied on 6 | Retrieval, against the best of four text searches |
| The behaviour appears in 20.8% of responses, against 21.4% with trained vectors and 1.8% unsteered | 15 induced WeirdChat behaviours, held-out prompts. The neuron set beats random sets on 9 of 16 behaviours |
| MMLU 0.69 unsteered, 0.67 with the neuron set and 0.67 with the trained vector | 200 questions |
| A one-line prompt gives 15.7%. The best Claude-written prompt gives 84.5%, against 23.2% for neurons | 7 behaviours |
| Right in 40%, 61% and 75% of audits | 174 audits each, for an auditor with outputs only, with steering and with neuron readouts |
| 54% of trigger words in the top 50, against 42% for summed change and 3% for embedding shift | 74 backdoored models. All methods do poorly on the 24 Qwen2.5 models |
| 14 of 23 checkpoints still rank first most answers that the original model knew | WMDP-cyber cloze scoring, even with accuracy near chance |
| Removing one direction raises the share from 0.32–0.47 to 0.72–0.84 | Three K-FADE checkpoints and CUT, 200 held-out questions |
| Resetting 30 of about 460,000 neurons recovers 0.43 to 0.71 of the lost answers, against at most 0.04 for random neurons | Zephyr RMU, RMU-b, RMU-LAT and SIFT, and Llama-3 circuit breakers (Table D1) |
| The most-changed neurons fire 20 to 100 times more strongly on cyber text after unlearning | RMU, RMU-LAT, ELM, circuit breakers and K-FADE. The retain text stays low |

## Where the claims go beyond the evidence

- **The judges and the agents are Claude.** Claude picks the neurons, judges the steered outputs, labels the documents it finds, runs the audits and grades them. No human rates anything.
- **Baselines chosen on the test set.** The best text search and the better probe are chosen on test results. The authors say that "favors the baselines", which is fair. But the neuron sets and steering doses are also chosen on development outputs that a Claude judge scored.
- **Many results are only in figures.** Per-set AUROCs, per-topic retrieval and per-behaviour steering rates are figure-only. The page can't check them.
- **One model for steering and AxBench.** Both use only Llama-3.1-8B-Instruct. The backdoor vote fails on all 24 Qwen2.5 models, and the authors say so.
- **"Hide rather than remove" rests on one domain.** All 23 checkpoints are WMDP-cyber. Cloze scoring shows that the ranking survives, not that the model can use the fact. Gradient ascent, NPO, SimNPO, LAT and TAR damage all text. The authors say so, and that their low cloze shares "may not reflect removed facts".
- **A contamination incident.** Some auditors read each other's files. The authors reran those audits and report it, to their credit.
- **No code or data.** The 3 K-FADE checkpoints, the 5 adapters the authors trained and the neuron sets are not released.

## Use in this repo

- **Two cheap removal tests for the fix campaign.** #5220's robustness section asks whether the failure comes back after a relearning attack or an in-context attack. The post adds two tests that need no training. **Cloze scoring** compares the probability of each answer's text, with no letters. **Direction removal** projects out the mean change that the fix makes to the residual stream, after the last changed layer. If either test brings the old answer back, the fix hid it. Both fit the campaign's templated failure family (#5232), whose answers are single tokens. The maintainer chose on 2026-11-27 to add both, and #7151 adds them to the fitting harness.
- **A fix that works like a backdoor.** The post finds that selective unlearning trains the equivalent of a probe into the weights, and adds noise when it fires. #5220 already reads each fix through the vocabulary, to tell "supply" from "suppression". The post's mechanism is a third reading: a gate on the input text that disrupts the computation. The campaign's gated arms are built that way on purpose. So the post's finding is a caution: a gated fix can pass held-out tests and still keep the wrong route intact behind the gate.
- **Steering adds the output vector, and the repo clamps the activation.** The `neuron-steering-tests` skill clamps a pole to 1, 2 and 4 × `max_activation`. The post adds the unit output vector, scaled by the layer's mean residual norm. Both set the neuron's write at every position, so the two are the same kind of test. The post's scale compares across layers. The repo's scale compares with the pole's own observed range. The post's MMLU check, 0.69 to 0.67, and its prompt baseline are the controls that [AxBench](axbench.md) asks for too.
- **Label-free selection differs from the repo's sparse probing.** The `sparse-probing-search` skill finds neurons for a concept from labelled data. The post picks them from descriptions and readouts with no labels. In the registry, the explanation of each VERIFIED pole could play the description's part. That would be a search over registry explanations, not a change to the protocol. It isn't opened here, because the post's own limit applies: neurons picked from top activations failed on security news, at AUROC 0.49.
- **Neuron voting and dormant-neuron backdoors.** The vote ranks tokens by how much a finetune changed each neuron's gate and up weights on their embeddings. #3471 reviews backdoors planted in dormant neurons. The vote is a check that the review can name.
- **Part 1's units carry over.** Part 2 uses the same signed units and the same J-lens readout as Part 1. So the gate rule in #7127 bears on these results too.
