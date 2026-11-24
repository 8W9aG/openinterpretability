# How to Turn an Attention Neural Network into an Interpretable Hard-Logic Predicate Circuit

**Citation.** Yuchen Li, Zongqi Fan, Xintong Li, Nguyen H. Tran and Ken-Tye Yong (The University of Sydney). "How to Turn an Attention Neural Network into an Interpretable Hard-Logic Predicate Circuit: Truth-Table Predicate Extraction and State-Machine Merging under Ship-of-Theseus Replacement". arXiv:2611.24381v1, submitted 18 November 2026, cs.LG with cs.AR. [Paper](https://arxiv.org/abs/2611.24381). Preprint, no venue given. No code is linked. The data and checkpoints are public: TweetEval, DistilGPT2 and BERT-base. We saw no funding statement. The paper states: "An AI assistant (Claude, Anthropic) assisted with engineering code, data organisation and the writing of the paper."

**Status.** Not applicable. The paper turns a fine-tuned text classifier into a logic circuit for an FPGA, and it doesn't test what any predicate or state means. No code, skill or protocol here depends on it. One idea carries over to the patch format's `compiled` operation (*Use in this repo*).

**Read.** On 2026-11-24 a `web-fetch` subagent read the abstract page and the HTML full text. A model read the source for us. We have the body's own text up to the references. That is Sections 1 to 7, Equations 1 to 3, Algorithm 1 and Tables 1 to 3. It also covers the statements on software, impact and AI use. Appendices A to F came back only as a summary, so this page uses no number from them. We didn't see the figures, the PDF or the TeX source.

## Summary

**Claim.** A trained attention network can be replaced, block by block, with a hard-logic circuit of predicates, small integer tables and rules. The circuit keeps most of the network's accuracy on a 4-class tweet task, and runs in microseconds on an FPGA.

**Method.**

- **Teachers.** DistilGPT2 (6 blocks) and BERT-base-uncased (12 blocks), each fine-tuned on TweetEval-emotion and quantised to FP8 E4M3. The task has 4 classes, with 3,257 training, 374 validation and 1,421 test tweets.
- **Fitting set.** Only the training tweets that the teacher classifies correctly: 3,032 for DistilGPT2 and 2,695 for BERT.
- **Replacement.** A "reverse Ship-of-Theseus" procedure (Algorithm 1) replaces the blocks from the output down. Each block becomes code tables, a 0-or-1 attention selection and rules. After each replacement, the blocks below that are still soft adapt to the frozen circuit.
- **Predicates and readout.** A predicate is a 0-or-1 function of a token's discrete state and its lexical code. Attention scores become sums of small integer tables, and softmax becomes a 0-or-1 selection, by order (Equation 1) or by a learned rule (Equation 2). The class is the argmax of a bias plus small integer weights on the predicates that hold (Equation 3).
- **Hardware.** Logic synthesis to 6-input LUTs, with clock rates bracketed and no place-and-route, so the circuit's timing and power are estimates.
- **Baselines.** A bag of word pieces, a logistic regression on the teacher's word-piece counts fitted to the teacher's labels. Also the teacher, and the teacher on a GPU for the hardware comparison.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| DistilGPT2 test accuracy, teacher | 0.747 |
| DistilGPT2 circuit, order selection, 3 seeds | 0.665 ± 0.013 (agreement with the teacher 0.695 ± 0.012) |
| DistilGPT2 circuit, rule selection, 3 seeds | 0.574 ± 0.025 |
| BERT test accuracy, teacher | 0.778 |
| BERT circuit, order selection, 3 seeds | 0.700 ± 0.005 (agreement 0.711 ± 0.007) |
| Bag of word pieces | 0.640 for each model |
| Size, order selection, seed 0 | DistilGPT2 47.87 Mbit against 655.3, BERT 29.62 against 875.9 |
| One to three circuit blocks, the rest soft, seed 0 | 0.752–0.757 for DistilGPT2, 0.779–0.786 for BERT |
| Latency per tweet, DistilGPT2 circuit on an Alveo U55C (estimated) | 1.06–1.91 µs, against 2.21 ms for the teacher on an RTX 5090 |
| Bit-exactness | The integer evaluator matches on all 1,421 test tweets |

## Where the claims go beyond the evidence

- **"Interpretable" is not tested.** The paper calls the decisions "readable predicates and rules". Its only reading is one worked tweet, with three predicates and one rule traced, and one state described by example word pieces. There is no systematic test of what predicates or states mean, no human study, and no count of rules a reader would face.
- **No null for readability.** The only interpretable comparator is the bag of word pieces. Nothing checks whether the predicates are more meaningful than those of a circuit fitted to a random or untrained network.
- **The circuit loses about eight points.** With order selection, it falls 8.2 points below the teacher on DistilGPT2, and 7.8 on BERT. The hybrids that beat the teacher by about half a point are single-seed, on 1,421 tweets.
- **The hardware side is an estimate.** The authors say so: no board measurement and no place-and-route. The microsecond figures leave out about 1–2 ms of tokenisation on the board's ARM cores, which the abstract doesn't mention.
- **Small inconsistencies.** The abstract says the models are fitted on the training set alone, but Section 4.1 makes conversion choices on the validation split. Section 4.3 says copying the teacher's errors "does not help", yet the rule-selection accuracy rises from 0.552 to 0.575 when fitting on all tweets.
- **Credit.** The authors report 3 seeds for the main results and an ablation of the fitting rule. They also check the circuit bit for bit against the integer evaluator. They name state-machine merging, generative decoders and larger encoders as work not yet done.

## Use in this repo

- **A total explanation in form, not in meaning.** A circuit of predicates is a complete description of what the model computes. The registry asks a different question: whether a stated reading of a unit holds against adversarial probes. Here the blocks above a predicate define it, by what they must distinguish. Nobody tested a meaning for it. So the circuit is closer to a compiled program than to an explanation the protocol would score.
- **The units are not the model's.** The circuit replaces the network's neurons and heads with new discrete states and tables. Nothing here reads a native MLP neuron, which is the registry's unit. The paper is a cousin of [Learning Transformer Programs](learning-transformer-programs.md) and [Tracr](tracr.md), going from weights to a program rather than from a program to weights.
- **One idea for the `compiled` operation.** The patch format registers `compiled`, `read_adapter` and `write_adapter` but doesn't build them (`patches/schema.py`, #5298, phase 5 of #5220). The paper's replacement order is one worked design for putting a compiled module into a live model. It replaces from the output down, then lets the soft blocks adapt to the frozen part. It is evidence, not a design for the campaign, which the maintainer designs.
- **No follow-up issue.** The paper suggests no measurement this repo should run.
