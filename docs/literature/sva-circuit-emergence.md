# The Emergence and Sparsification of a Syntactic Circuit

**Citation.** Qing Yao, Sasha Boguraev, Tiago Pimentel and Kyle Mahowald. "The Emergence and Sparsification of a Syntactic Circuit". [arXiv:2610.33299](https://arxiv.org/abs/2610.33299), v1 of 23 October 2026. The arXiv comment and the code's README say EMNLP 2026. The PDF has no proceedings header, and this page found no ACL Anthology entry. The paper is under CC BY 4.0. The [code](https://github.com/dounick/sparsification-of-sva-circuits) has no licence file. The issue's URL, `dounick/sva-circuits`, redirects there.

**Status.** Planned. #7186 runs the repo's agreement circuit across Pythia-70m's checkpoints, with the circuit gates' random-set null, for `papers/neuron-development/`. That paper should cite this one.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the v1 PDF and turned it into text. It read all 26 pages verbatim: Sections 1 to 6, the limitations, and Appendices A to F with Tables 1 to 19. It also read the code's README and its checkpoint script. It saw the figures only as captions and axis text, and it did not read the code or the data files. The reader of this page was a model (Claude Opus 5.5), through that subagent's report. The issue's summary of Sections 5 and 6 was checked against this text, and the corrections are below.

## Summary

**Claim.** Pythia learns subject–verb agreement early, but the circuit that carries it keeps changing after that. It first spreads over hundreds of neurons, then shrinks to a few dozen, and its main attention head moves to another layer.

**Method.**

- **Models.** Pythia-1b, one seed, across its training checkpoints. Ten seeds of pythia-410m: the code names seed 0 as `EleutherAI/pythia-410m` and seeds 1 to 9 as PolyPythias.
- **Data.** Synthetic minimal pairs with a prepositional phrase between subject and verb: 400 to train and 192 to evaluate. Relative-clause, no-distractor and perturbed templates, and 80 WikiText pairs, test generalisation.
- **Metric.** The agreement gap G, a mean log-probability ratio of the right and wrong verb forms. For single tokens it equals the logit difference.
- **Necessity.** Nec = (G − G with the circuit mean-ablated) / G. The means come from the training set.
- **Sufficiency.** Suff = (G − G with the circuit's activations swapped from the other number) / (G − G with everything swapped).
- **k50.** The smallest number of subject-position neurons, ranked by RelP attribution, with both measures at 50% or more. It is interpolated between powers of two.
- **Heads and routing.** Heads are ranked by min(Nec, Suff). Routing from neurons to heads is a path-patching-style drop in sufficiency when a head is blocked.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Agreement is learned "within 2% of training", by step 2,000 | Pythia-1b. Step 2,000 is 1.4% of 143,000 |
| k50 is 555 at step 2k, 83 at 5k, 42 at 8k, 21 at 18k, 41 at 50k and 31 at 143k | Pythia-1b, subject-position neurons only (Table 7) |
| The main head is L15 H2 at 1k, L15 H7 from 2k to 8k, and L13 H7 from 18k on | Pythia-1b |
| The final circuit, 18 neurons and L13 H7, has 78% necessity and 83% sufficiency, against 83% and 88% for the whole scope | Pythia-1b at step 143,000. 10 of the neurons sit at the subject and 8 at the prediction |
| Sufficiency 80–89% on relative clauses, and 82% on WikiText | Generalisation of the final circuit |
| An adverb after the object drops sufficiency to 42% | A perturbed template |
| The final k50 ranges from 6 to 255. Seeds 3 and 4 grow again at loss spikes | Ten pythia-410m seeds |
| Within a seed, a median 85% of the smallest k50 set is shared across checkpoints after step 8,000 | Pythia-410m |
| The leading head changes in 6 of 10 seeds, with "no predictable pattern" | Pythia-410m (Table 9) |
| The 10 upstream neurons are plural detectors. L14 n1160 promotes plural verbs and L14 n4146 singular ones | Pythia-1b (Table 14) |

## Where the claims go beyond the evidence

- **No random null.** The paper tests no random set of neurons or heads of the same size, and runs no statistical test. Its controls are the whole scope as a ceiling, swaps from donors off the circuit, a distributed-alignment-search complement and a head-only baseline. So "18 neurons carry 78%" has no reference for what 18 random neurons carry.
- **One run for the main result.** The trajectory of pythia-1b is one seed. The ten pythia-410m seeds show that the final circuit's size varies 40-fold. So the single pythia-1b path is one draw of a wide spread.
- **k50 is coarse.** It counts subject-position neurons only, and it interpolates between powers of two. The "18 in the final k50" of Section 4.2 is a different set from the 18-neuron final circuit.
- **No account of why.** The authors say that they don't explain the sparsification or the rerouting. Their "scaffold" reading is a hypothesis, with a parallel to grokking.
- **Neuron identities across seeds aren't compared.** The seed results compare sizes and heads, not which neurons.
- **One phenomenon in English, found by hand.** The authors state this.

## Use in this repo

- **It agrees with the neuron-development paper, at the level of a circuit.** `papers/neuron-development/` finds that Pythia-70m's VERIFIED neurons correlate with their final form at a median of only 0.62 at step 16,000. That is when the loss is within 1% of its final value. This paper finds that a circuit keeps changing after its behaviour is learned. The neuron-development paper doesn't cite it. #7186 adds the citation, and runs the circuit across Pythia-70m's checkpoints.
- **The repo's agreement circuit is the same task on another model.** `docs/analyses/sva-circuit.md` (#3225) finds that about 100 neurons carry agreement in Qwen2.5-0.5B. With all others mean-ablated, they keep a faithfulness of 1.01, and ablating them leaves 14% of the metric. Random sets of the same size keep none of it. That random-set check is what this paper lacks. #7186 carries it to the checkpoints.
- **Its thresholds differ from the circuit gates.** This paper asks for 50% necessity and 50% sufficiency. The circuit gates of `docs/PROTOCOL.md` §4 ask for three things. Necessity must be at least 0.02, and the loss must beat 95% of 64 random sets of the same layers and kinds. The circuit must also be specific to its task. The gates are looser on size and stricter on chance. A set that passes this paper's 50% bar may still fail `beats_null` if many random sets do as well. The repo's Qwen result suggests that they don't.
- **Its neuron readings fit the registry's.** The paper reads its 10 upstream neurons as plural detectors. In the registry, targeted audits of the top 20 agreement neurons in Qwen2.5-0.5B verified 13. The explanations of 17 of the studied neurons mention number or agreement (`docs/analyses/sva-circuit.md`). So in two model families, the agreement circuit is made of neurons with clear single readings.
- **RelP attribution chooses its neurons.** It is a gradient attribution, as in `attribution-patching.md`. The repo's circuit method ranks by an integrated-gradients attribution. Neither is the scan that the registry's random draws use, so circuit units come from targeted draws, as the SVA analysis says.
