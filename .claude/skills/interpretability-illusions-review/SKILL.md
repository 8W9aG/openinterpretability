---
name: interpretability-illusions-review
description: Reviewer's checklist of documented ways neuron explanations fool their authors, used to write adversarial probes and judge submissions. Use whenever acting as the reviewer agent, writing hard negatives or positive holdouts, or sanity-checking a VERIFIED result that seems too clean.
---

# Interpretability illusions: reviewer checklist

## Papers

- Bolukbasi et al., 2021, *An Interpretability Illusion for BERT*, arXiv:2104.07143. The same neuron gets a different "meaning" depending on the dataset used to find its top activations.
- Huang, Geiger, D'Oosterlinck, Wu & Potts, 2023, *Rigorously Assessing Natural Language Explanations of Neurons*, arXiv:2309.10312. Confident explanations had precision of about 0.64 and recall of about 0.5, and showed no causal mediation.
- Makelov, Lange & Nanda, 2023, *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*, arXiv:2311.17030. Patching can activate a dormant pathway and produce an effect the component doesn't have in normal use.
- Sun, Chen, Kolter & Liu, 2024, *Massive Activations in Large Language Models*, arXiv:2402.17762. A few activations are enormous at fixed positions (first token, delimiters) and act as biases.
- Goodfire, *Neural Networks Are Already Clean* (2026). Misreadings come from short context, narrow data, backward-only tools, unknown motifs and miscalibrated explainers.

## Checklist when writing probes

1. **Same token, other senses.** Use the claimed trigger token in unrelated senses (for "including": *"the price including tax"*, *"Including him was a mistake"*).
2. **Same topic, no feature.** Keep the domain and vocabulary but remove the claimed feature.
3. **Feature in an unusual position.** Sentence-initial, after a newline, at the end of a long document, inside code comments.
4. **Paraphrase holdouts.** Express the claimed concept with words absent from the top examples ("for example", "among them", "like"). This is where `L11/N4255` failed.
5. **Cross-dataset holdouts.** Draw at least one holdout from a different genre than the scan corpus (Bolukbasi).
6. **Position and massive-activation controls.** If peaks cluster on the first token or on delimiters, add probes with the same delimiter and none of the claimed content (Sun et al.).
7. **Long-context probe.** Put the feature deep inside a ≥ 512-token passage. Short probes can miss long-context neurons (Goodfire).
8. **Label by the hypothesis.** Every probe's `expected_active` is what the hypothesis text implies, even if you suspect the neuron disagrees. That is the point of the test.

## Checklist when judging

- **Selection.** Was the neuron chosen at random, or picked because it looked clean? (`provenance.selection`)
- **Scan breadth.** Corpus diversity and `snippet_tokens`. Under 256 tokens is a reason to ask for a rescan.
- **Counterfactuals.** The proposer measured these before submitting, so they are a reproduction check, not a prediction (`docs/PROTOCOL.md` §4).
- **Patching effects.** If only patching, and not ablation, shows an effect, suspect a dormant pathway (Makelov et al.).
- **Independence.** The reviewer model must differ from the proposer's. The `independent_review` gate enforces this; don't work around it.
