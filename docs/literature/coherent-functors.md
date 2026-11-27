# Presenting Neural Networks via Coherent Functors

**Citation.** Matthew Pugh (University of Cambridge), Jo Grundy, Corina Cirstea and Nick Harris (University of Southampton). "Presenting Neural Networks via Coherent Functors". arXiv:2604.15100, v1, 16 April 2026, math.CT, CC BY 4.0. [Paper](https://arxiv.org/abs/2604.15100). Preprint, no venue given. The paper gives no code, data, funding or tool disclosure. It builds on the same authors' "Learning Is a Kan Extension" (arXiv:2502.13810, 2025).

**Status.** Not applicable. The paper is theory, with no experiments, and it covers only dense feed-forward networks over a finite set of floats. Its view of a network as a model of a formal theory doesn't give a test of what one unit computes. Its weight fixing is a constraint on an architecture, not an edit of a trained model. So no code, skill or protocol here would use it.

**Read.** On 2026-11-26 a subagent (Claude) downloaded the abstract page and the full HTML of v1 with curl, and turned them into plain text. It read every section verbatim: Sections 1 to 5, all definitions, theorems, proofs and footnotes, and the references. It did not see Figure 1, an image of an example database. The commutative diagrams came through only as node and arrow labels, so their arrows were read from the prose. The paper has no appendix. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A dense feed-forward network architecture over floats can be presented as a coherent category G. G's models in sets are exactly the networks of that architecture, and inference is precomposition with a coherent functor into G. The authors read this as follows. An architecture is "a hypothesis about the structure of a dataset". Training lifts a dataset into a more constrained theory.

**Method.** Category theory, with no experiments.

- **Models of theories.** A κ-coherent theory is a set of sequents ("whenever Φ holds, Ψ holds"). A model of a theory in sets is the same as a coherent functor from its syntactic category to Set. These are Theorems 2.27 and 2.30, from Makkai and Reyes. A functorial database schema is such a theory (Theorem 3.1).
- **Hard coding (Theorem 3.7).** Take a functor F into a small well-pointed coherent category. Then there is a coherent category δ, and every coherent functor out of δ reproduces F up to isomorphism. So chosen data is fixed in every model.
- **Floats.** Hard coding needs a finite set, so the paper uses a fixed finite set R of floating-point numbers. Addition, multiplication and an activation σ are fixed as tables on R (Lemma 4.1).
- **A layer** (Lemma 4.11) adds to the float category one free weight matrix and one free bias vector. A model in sets then chooses their values, and the layer computes σ(wx + β). Layers compose by pullback of spans (Lemma 4.12).
- **Weight fixing and tying (Theorem 4.14).** A 2-coequaliser forces two global elements to be equal in every model: two weights, two biases, or a weight and a fixed constant. Sparse layers fix weights at zero. Convolutional layers tie weights to a kernel.

**Findings, as the source reports them.** The results are theorems. The paper reports no measurement.

| Result | Setting |
| --- | --- |
| Theorem 4.13: the models in sets of G are the networks of the architecture, and precomposition along ι is the inference function | Any dense feed-forward architecture of finite depth, with any activation per layer and biases, over a finite float set. The claim holds "on objects" |
| Theorem 4.14: the same holds for the networks that satisfy a finite list of fixing and tying constraints | As above, with sparse and convolutional layers as special cases |
| Theorem 3.7 (hard coding): fixed data is in every model | A small well-pointed κ-coherent category, with κ-small sets of points |

"Inference" here is the network's whole input-to-output graph over the finite float domain (Definition 4.5), not its outputs on a sample.

## Where the claims go beyond the evidence

- **"Training as lifting" is a reading, not a result.** The authors say so. The introduction says that the natural isomorphisms of their 2-category "are not immediately suited to realising the Kan extension as a loss-minimising model". They leave this to future work. In the conclusion: "It is likely that such a naive assignment of 2-morphisms will not produce the desired Kan-extensions". So nothing in the paper says that training finds the models of the lifted theory.
- **"All feed-forward neural networks" is wider than the theorems.** The introduction says the paper represents "all feed-forward neural networks". Theorem 4.13 covers dense layers of the form σ(wx + β), composed in sequence. The paper never mentions residual connections, attention, normalisation, softmax or products of activations. These are not ruled out, but nothing shows that they fit. A footnote names kernel SVMs as covered, and the paper gives no proof for them.
- **Inference is shown on objects only.** Theorems 4.13 and 4.14 say what the functor does to each network. They say nothing of what it does to the maps between networks.
- **Finite floats only.** The construction needs a finite set of values, so it doesn't reach real-number networks. The authors state the reason, an open 2-completeness proof, and they expect that proof to follow Kanalas (2021).
- **No worked network.** No float format, activation or concrete network is worked through. The only example is a shop database.

## Use in this repo

- **It gives no test of what a unit computes.** The registry's claim is about one neuron pole: a natural-language hypothesis that probes, counterfactuals and causal readouts test (`docs/PROTOCOL.md` §4). The paper's object is a whole network as a model of a theory, and its only "constraints on behaviour" are the architecture's equalities between weights. A sequent of that theory states which weights are equal or fixed, not what a neuron responds to. So the paper can't say what a VERIFIED pole guarantees, and it doesn't try. It never mentions neurons, features, concepts or probes.
- **Its reach stops before the registry's units.** One GPT-2 MLP block on its own is two dense layers, the first with GELU and the second with the identity. The theorems cover that block as a function of its input. But in the model, the block reads a residual stream that attention and layer norm write. Its output is added back, which is a residual connection. Qwen2.5's MLP is gated, `act(gate(x)) * up(x)` (`docs/PROTOCOL.md` §1), so it takes a product of two activations, which the paper's layer form lacks. Pythia's blocks also run attention and the MLP side by side. None of the configured models is covered as a whole.
- **Hard coding is a constraint on an architecture, not a fix.** Theorem 4.14 defines the set of networks with chosen weights pinned to chosen values. That is close to a `replace` op of a stored patch (`openinterpretability/patches/schema.py`, `docs/patches.md`), which sets weights in a base model. A patch here is judged by its evidence: the fix's checks and the interference report (#5220 §9). The paper's construction only says which networks satisfy the equalities. It says nothing about what pinning the weights does to the network's other behaviour. So it adds nothing to the fix campaign's question of interference. The weight-space programming pages, such as [Sparks of Weight-Space Programming](rikiyat-weight-space-programming.md), measure that question, and this one doesn't.
- **One idea carries over, as framing only.** The paper frames an architecture as a hypothesis about data, and its fit as a test of that hypothesis. That matches the registry's habit of treating each explanation as a falsifiable hypothesis with its own test. It changes nothing here, and no follow-up issue is needed.
