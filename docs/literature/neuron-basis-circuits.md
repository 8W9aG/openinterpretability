# Language Model Circuits Are Sparse in the Neuron Basis

**Citation.** Aryaman Arora (Stanford University and Transluce), Zhengxuan Wu (Stanford University and Transluce), Jacob Steinhardt (Transluce) and Sarah Schwettmann (Transluce). "Language Model Circuits Are Sparse in the Neuron Basis". arXiv 2601.22594, v1 30 January 2026, v2 11 June 2026. [Paper](https://arxiv.org/abs/2601.22594). [Code](https://github.com/TransluceAI/circuits). ICML 2026, spotlight. v2 is the camera-ready version. We found no funding or tool disclosure.

**Status.** Planned. Issue #3582 tries the paper's RelP attribution for our activation wiring and for attribution selection, on Qwen's gated MLPs. The paper also supports the registry's choice of unit.

**Read.** The abstract page and the v2 HTML, on 2026-11-13. A model read the text and reported it to us, with short quotes. It read Sections 1–8 and Appendices A to the start of H.2 as the paper's own text. The page cut off the rest of H.2 and all of H.3. We did not see the figures, so the numbers on the figures' curves are not here. We did not read v1.

## Summary

**Claim.** MLP neurons are "as sparse a feature basis as SAEs" for circuits. A circuit of about 10² neurons controls a model's behaviour on a task, and no dictionary has to be trained to find it.

**Method.**

- **Units.** A neuron is one coordinate of the MLP's hidden activation, the input to the down projection. The paper compares that basis with four others. They are the MLP output, the attention output, the residual stream and SAE latents (Llama Scope, Gemma Scope).
- **Models.** Llama 3.1 8B (base for the main results, Instruct for the case studies), and Gemma-2-2B and 9B in an appendix.
- **Circuits.** A circuit is the units with the largest attribution, taken from training pairs. Two scores, measured on held-out pairs by mean ablation, judge it. *Faithfulness* is the share of the task metric that the circuit keeps when everything else is ablated. *Completeness* is the share left when the circuit itself is ablated. A perfect circuit has faithfulness 1 and completeness 0.
- **Attribution.** Integrated gradients (10 steps), and **RelP**, which takes one backward pass. RelP replaces each nonlinearity by a linear stand-in in the backward pass only: it freezes the RMSNorm denominator, the SiLU factor and the attention weights. It splits the gated MLP's product by the "half rule" of layer-wise relevance propagation.
- **Tasks.** Subject–verb agreement has four templates, with 300 training and 40 held-out pairs each. The MIB circuit benchmark adds five tasks. 50 two-step questions ask, for example, for "the capital of the state that contains Dallas". The appendices add addition (10,000 sums) and antonyms in 9 languages.

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| Subject–verb agreement, Llama 3.1 8B | Circuits on MLP neurons are about 100 times smaller than circuits on MLP outputs, for the same faithfulness |
| The same, with RelP | Faithfulness and completeness are near perfect at about 200 neurons |
| The same, edges | RelP keeps over 80% faithfulness with about 10⁵ edges, 10% of the candidates |
| MIB benchmark, mean CMD over 5 tasks (lower is better, 3 seeds) | RelP 0.08, EAP with counterfactuals 0.08, EAP-IG on inputs 0.10, random 0.74 |
| Gemma-2 | About 100 times smaller than MLP outputs again |
| The Dallas → Texas → Austin question | 257 neurons found automatically. The authors chose 23 by hand and put them in six groups. Steering each group except "state" changes the top prediction |
| One neuron, L23/N8079 | Steering it changes the answer from the capital to the state "in a majority of examples" |
| Addition | For each last digit of the sum, some neurons separate it at AUROC ≥ 0.9 |

The authors also took 12 always-active neurons out of the circuits by hand.

## Where the claims go beyond the evidence

- **"As sparse as SAEs" rests mostly on one task.** The size comparison with SAE bases uses subject–verb agreement, with four templates and 40 held-out pairs each. MIB tests the attribution method, not the basis. The case studies are qualitative.
- **No random basis.** No experiment compares the neuron basis with a random rotation of the same layer. Without one, a reader can't tell how much of the sparsity comes from the neurons and how much from the MLP's width.
- **Few statistics.** Most results come from one run. There are no intervals, no significance tests and no per-task variance in the text. MIB is the exception, with 3 seeds.
- **The neurons' meanings are hand-picked.** The authors read the 23 neurons of the Texas circuit themselves, out of 257. They take descriptions from Transluce's automatic neuron descriptions, which they say are "often not specific", and which miss the digit role of the addition neurons.
- **The text and Table 3 disagree.** The text says RelP is best on three MIB tasks and second on two. In the table it is third on ARC (easy), behind two baselines.
- **Small errors.** Table 9's caption names Claude Sonnet 3.5 for a study on Llama 3.1 8B Instruct. The abstract's "about 10²" neurons matches no single number in the body. The nearest is about 200.

The authors name two limits themselves: a full circuit still has too many neurons for a person to read, and their implementation is slow.

## Use in this repo

**It uses the protocol's unit.** The paper's neuron is the protocol's neuron (`docs/PROTOCOL.md` §1): one coordinate of the input to the down projection, `act(gate(x)) * up(x)` for gated MLPs. Qwen2.5 has the same gated MLP as Llama 3.1. So the paper's main result, that behaviour runs through few of these units, supports the registry's choice to study them directly. The `native-neuron-audit` skill takes the same view from Goodfire's work.

**It measures something different from the transcoder baseline.** Our learned-basis baseline (`docs/analyses/transcoder-baseline.md`) found that 92% of random transcoder latents verify, against 50% of random positive neuron poles, on GPT-2 small. That measures how often a single unit has an explanation that survives adversarial probes. The paper measures how many units a task's circuit needs. A basis can win on one and lose on the other, so the two results don't conflict. The paper's own caveat applies to both: it says that "the amount of uplift from SAEs is unknown".

**Its descriptions show why we probe.** The paper names its neurons with Transluce's descriptions, chosen by a simulator's score, and finds them often not specific. Issue #3481 reviews those descriptions. The registry tests each explanation with hard negatives and held-out positives, which catch a description that is too broad.

**Its always-active neurons match our default-on stop.** It removes 12 such neurons from circuits by hand. The protocol stops on a unit active on half its scan's tokens or more (`docs/PROTOCOL.md` §1, *Default-on*). Our reason is that probes can't test it. The paper's reason is that the unit gets in the way of circuits.

**RelP could improve our attribution.** Activation wiring (`VerificationVerdict.wiring`, `openinterp/harness/wiring_activations.py`) and attribution selection (`attribution_patching`, `activation-patching-practice`) use plain gradient × activation. That is the method that RelP improves on, and its half rule is for the gated product in Qwen's MLPs. Issue #3582 compares the two against exact patching.

**Circuits as a later stage.** The registry records direct wiring and finds circuits from it (`openinterp/builder/circuits.py`), but it does not trace task circuits. The paper's subject–verb agreement task is a ready test for a neuron circuit on Qwen, once the attribution work is done.
