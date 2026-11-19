# Learning to Control Fast-Weight Memories

**Citation.** Jürgen Schmidhuber, Institut für Informatik, Technische Universität München. A footnote gives his current address as the Department of Computer Science, University of Colorado, Boulder. "Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks". *Neural Computation* 4(1):131–139, 1992. [DOI 10.1162/neco.1992.4.1.131](https://doi.org/10.1162/neco.1992.4.1.131). The journal version is behind a paywall. We read the author's copy, a 9-page PDF at `https://sferics.idsia.ch/pub/juergen/fastweights.pdf`, which gives the journal reference on its first page. An earlier version is Technical Report FKI-147-91, TU München, 26 March 1991, at `https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf`. The paper links no code or data, and it gives no funding statement. The acknowledgements thank Klaus Bergner for the experiments, and Mike Mozer, Bernhard Schätz and Jost Bernasch for comments on a draft.

**Status.** Planned. #4782 would use it in its strand on weight-space programming and fast weights, as the first trained network that writes another network's weights. No code, skill or protocol depends on it, and this page proposes no follow-up issue.

**Read.** We read it ourselves on 2026-11-19, with no subagent. The text layer of the author's copy is broken, so we turned its 9 pages into images and read each page. That covers the abstract, Sections 1 to 5, Equations 1 to 7 and the references. We also read the 1991 report as OCR text, in full. We did not see the journal's typeset pages. So the author's copy may differ from the printed text in small ways. The paper has no figures and no tables.

## Summary

**Claim.** A network can keep a short-term memory in weights, not in activations. A slow feed-forward network S learns, by gradient descent, to write weight changes into a second feed-forward network F at each time step. F then maps the same input in different ways at different times. The system solves two small tasks with unknown time delays, one of them a temporary binding of a variable to a value.

**Method.**

- **Task.** A sequence of input and target pairs, with squared error summed over time steps. In the on-line version there are no episode boundaries.
- **Architecture.** S reads the input x(t). Its outputs set a change □w_ab(t) for each weight w_ab of F. Equation 1 sets the new weight to σ(w_ab(t−1), □w_ab(t)), where σ is a differentiable function, for example sum-and-squash. F then reads the same input with its new weights. The author says that Equation 1 is "essentially identical" to Equation 1 of Möller and Thrun (1990).
- **Two interfaces.**
  - Equation 4 gives S one output unit for each weight of F, so □w_ab(t) = s_ab(t).
  - Equation 5 is the FROM/TO interface. S has one output unit for each unit of F that a fast weight leaves. It has one more for each unit that a fast weight enters. The change is the product □w_ab(t) = s_a(t)·s_b(t). The paper calls this "Hebb-like". It reads the FROM pattern as a *key* and the TO pattern as its *entry*.
- **Learning rule.** The paper derives the exact gradient for S's weights. A variable p_ij^ab, updated at each step as in real-time recurrent learning, carries ∂w_ab/∂w_ij forward in time. The rule is local in time, at O(|W_F||W_S|) per step, but not local in space. An unfolding-in-time variant is also given.
- **Experiments.** Both nets are feed-forward, with no hidden units and linear output units. Equation 7 sets the fast weight to a logistic function of the old weight plus the change, minus 0.5, with steepness T. That bounds each fast weight between 0 and 1. S's weights start uniform in [−0.1, 0.1]. A task counts as solved when F's error stays at or below 0.05 for 100 steps in a row.
  - *Flip-flop* (from Williams and Zipser, 1989). Inputs are A, B or C, one at a time, with random gaps. F's single output must switch on the first time a B follows an A, and stay off at all other times. F has 3 fast weights.
  - *Variable binding.* F has one input unit, the variable WHERE-IS-MY-CAR?, and three output units for parking slots P1, P2 and P3. S has six inputs: three slot detectors and three random binary distractors. A simulated driver stops driving with probability 0.25 at each step and finishes business with probability 0.25.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Flip-flop, one output unit of S for each fast weight (Equation 4), T = 10, η = 1.0 | Solved within 300 time steps |
| Flip-flop, FROM/TO interface (Equation 5), T = 10, η = 0.5 | Solved within 800 time steps |
| Variable binding, Equation 4 with Equation 7, T = 10, η = 0.02 | Solved within 6,000 time steps |
| Flip-flop, the "typical solution" | When A occurs, S makes a large weight on F's B input line. When B occurs, S makes that weight small again, which "resets" F |
| Variable binding, the weights after training | A large fast weight leads from WHERE-IS-MY-CAR? to the slot where the car is. The other fast weights stay suppressed |

## Where the claims go beyond the evidence

- **"An alternative to dynamic recurrent networks" has no comparison.** No recurrent network runs on the same two tasks. The flip-flop comes from Williams and Zipser (1989), but the paper gives no number from that work next to its own.
- **The storage efficiency is a potential, not a measurement.** The abstract says that one weight "may be sufficient" in place of a full unit. No experiment counts weights or units against the capacity that a task needs.
- **Each result looks like one run.** The paper gives one number of time steps for each setting, with no seeds, no repeats and no spread. "The typical solution" implies several runs, but the paper does not say how many there were or how many found it.
- **The tasks are very small.** F has 3 fast weights in each task, and neither net has a hidden unit. The variable-binding task uses only the one-output-per-weight interface, so the FROM/TO interface is not tested on binding.
- **The weight readings are not tested.** The two "typical" solutions come from looking at the weights after training. No test removes or clamps a fast weight to show that the behaviour depends on it.

The author states some limits himself. Section 2.3 says that two feed-forward nets can only respond to changes in the input, and can't make an oscillation while the input stays fixed. Section 2.1 says that the on-line version needs a learning rate small enough to avoid instability. Section 2 says that the rule is not local in space. Section 3.2 credits the local output code for how easy the binding is to read.

## Use in this repo

- **No unit explanations, no probes.** The paper reads three weights by eye, in networks with no hidden units. It has no neuron explanations, no adversarial probes and no null. So nothing in it bears on a pole's verdict.
- **Its weights are easy to read only because its codes are local.** One input unit stands for the variable and one output unit for each slot. The author says that this local representation is why "the binding patterns were easy to understand". A transformer's MLP reads and writes the residual stream, which has no such code. So the repo reads a neuron's weights through the vocabulary: `vocabulary_lens` in `openinterpretability/harness/jlens.py`, from the key-value view of [`ffn-key-value-memories.md`](ffn-key-value-memories.md). That view is the same split into key and entry that the FROM/TO interface makes.
- **In a fast-weight system, the weights that hold the meaning don't exist until a forward pass.** S writes F's weights at time t from the input so far. A reading of S's slow weights alone doesn't say what F holds at any step. For a transformer's MLP neurons, the weights stay fixed at inference, so a weight-only reading of them is possible, as ROTATE (#3446) proposes. Whatever a transformer computes from the context, such as an attention pattern, is the part that changes with the input. A weight-only reading can't see that part. #4782 lists Schlag, Irie and Schmidhuber (2021), *Linear transformers are secretly fast weight programmers*. That paper is the place to test whether attention is a fast weight in this sense. We have not read it.
- **The repo's evidence says that a weight reading doesn't check an explanation by itself.** Two analyses agree:
  - [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md) tests ROTATE's weight-only kurtosis on 518 test poles. It doesn't predict VERIFIED: AUC 0.51 for the output side and 0.54 for the input side, and each interval includes 0.5.
  - [`docs/analyses/copy-suppression-l4h1.md`](../analyses/copy-suppression-l4h1.md) finds pythia-70m's L4/H1 with a copy-suppression signature in its weights (42.0% on the OV test). On ordinary text, that signature explains 1.8% of the head's effect.

  This paper's own readings are of the same kind: a pattern seen in weights, with no test of the behaviour. So for the question in #3446, the evidence here and in the repo points the same way. A weight reading can suggest an explanation or rank candidates. The verdict still needs forward passes.
- **What it adds to weight-space programming (#3381, #4471).** Gradient descent trains S to write F's weights, and the written weights can be read afterwards. #4471 has an LLM agent write a model's weights as Python code, with no gradient. Both depend on a small, local code for the written weights to be readable. #4471 proposes hand-programmed ReLU blocks as known ground truth for the native-neuron audit. This paper's nets have no hidden units, so they offer no such case.
- **For the interference question of #4782.** A fast weight holds information only until S overwrites it. In the flip-flop, S clears the B weight on purpose. The paper says that, with no episode boundaries, activations from past episodes "could have a harmful effect" on later ones. It doesn't measure that effect. So it gives no evidence on how much one write damages another. The repo's slow weights also keep changing. In [`papers/neuron-development/README.md`](../../papers/neuron-development/README.md), Pythia-70m's VERIFIED poles have a median correlation of 0.62 with their final form at step 16,000. By then the loss is within 1% of its final value.
