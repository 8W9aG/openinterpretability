# Calibrated Concept Profiles

**Citation.** Piotr Wilam (University College London). "Calibrated concept profiles: making the reading of a model's neurons a measurement". arXiv 2611.09196v1, 6 November 2026. [Paper](https://arxiv.org/abs/2611.09196). [Code](https://github.com/piotrwilam/AtlasCalibratedConceptProfiles). [Dataset](https://huggingface.co/datasets/piotrwilam/AtlasOpenCoder) (tag v1, doi:10.57967/hf/11472). Preprint, one author, no venue given.

**Status.** Not applicable. The paper reads every neuron's response to a *named* concept in a model and a language that the registry doesn't study. The registry reads one unit and tests a free-text explanation of it. Its controls match ones we already run, and one of its results supports issue #3559.

**Read.** The abstract page and the v1 HTML, on 2026-11-13. A model read the text in three passes and reported it to us, with short quotes. Every pass came back as the paper's own text, including the appendices. We did not see the figures.

## Summary

**Claim.** Reading a model's neurons directly, without a second trained model such as a probe or a sparse autoencoder, can be made a measurement. To do that, declare the inputs, measure the noise, and state each number in units of that noise.

**Method.** The model is OpenCoder-1.5B (24 layers, SwiGLU). The units are its 147,456 MLP neurons, each read as the gate branch times the up branch. The concepts are 206 Python constructs: 63 grammar entries and 143 builtin names. 169 of them pass every build check. For each construct:

- **Three sets of generated prompts.** Set A uses the construct in its role, set B uses its token out of that role, and set C is ordinary code. The generator draws each nearby factor at its rate in 253,398 public Python files, and it never runs the model.
- **The profile.** For each neuron, the mean activation on A minus the mean on B, at the construct's first token. Each activation is first standardised by the neuron's mean and spread on set C, in the same *cell*. A cell is a token class (keyword, identifier and so on) and a fifth of the prompt's length.
- **The noise unit.** The battery is drawn twice ("twins"), on the same 8 cores with different surroundings. One twin unit is `sqrt(2(1 − cos))`, from the cosine between the two profiles. Bootstrap intervals resample the cores.
- **The licence.** Nine checks run before any reading. Two of them gate the reading: a scrambled-label null and a split-half null, which the real profile must beat at the 95th percentile. The other seven are reported.
- **Readings over training.** 61 references: 35 pretraining checkpoints from 4.19 B to 2,034 B tokens, 24 annealing checkpoints and 2 untrained models. The readings are the profile's norm, its sparsity, twin agreement and the *carrier set*, the top 5% of neurons.

**Findings, as the text reports them.**

| Reading | Result |
| --- | --- |
| Twin agreement at the final checkpoint | Median cosine 0.986 over 188 constructs |
| Nulls (31 constructs) | The real profile is a median 10.7× the scrambled null's 95th percentile (minimum 7.2×), and 12.8× the split-half null's (minimum 6.0×). None of 31 falls at or below either |
| Resolution | The median step between neighbouring checkpoints is 8.56 twin units early in training and 2.91 in annealing |
| Ablating the carrier set (16 constructs, against 200 random sets of equal size) | The median loss rise above random is 0.06 nats at 4.19 B tokens, 0.25 at 83.9 B and 0.40 at 2,034 B. It is above random for 12, 14 and 14 of 16 constructs (sign test p = 0.038, 0.002, 0.002) |
| One token, two roles | `for` as a loop and `for` in a comprehension share 0.25–0.27 of their carriers. Chance is 0.05, and the twin ceiling is 0.87 |
| Linear probes, A against B, 164 constructs | Untrained: median accuracy 0.653 on the residual stream and 0.802 on the hidden units. Trained: median 0.998 at the first checkpoint and 0.999 later |
| A scale artefact | From the first checkpoint to the last, the standardised norm rises for 144 of 169 constructs while the raw norm falls for 148 |

## Where the claims go beyond the evidence

- **One model, one language.** OpenCoder-1.5B was trained on 607 programming languages, but the battery is Python only, drawn from public files and not from the training data. The author names a second model as outstanding.
- **Named concepts only.** The method can't find a concept that the battery doesn't name. The author says this.
- **The nearby factors are not matched.** The generator aims for a gap of at most 0.15 between A's and B's companion shares. The gaps are 0.19–0.68. So part of an A − B difference can be the surroundings, not the role.
- **The prompts are generated.** No comparison with real code is reported, and the prompts are short (median about 55 tokens).
- **Two twins on shared cores.** The noise unit rests on two draws that share their 8 cores. So it does not measure how much the reading depends on the cores themselves. The author says that it is too small for intervals.
- **Ablation on 16 constructs.** The function result uses only constructs with a fixed next token. After a Holm correction, the share of constructs that are significant alone is 2, 8 and 8 of 16.
- **The untrained reference can't be rebuilt.** Its random seed was not stored. The full record (about 2 TB) is not released.
- **The licence changed after a re-run.** A fault fix on 19 October changed the length check and the nulls. The paper reports both versions, which is good practice. But the licence numbers in the table come from the second run.

## Use in this repo

**It reads the other direction.** The paper starts from a construct and asks which neurons carry it. The registry starts from a neuron pole and asks what it carries, in a free-text explanation that probes must not break. The paper does not score explanations of single neurons, and it says "The carrier set is a measurement, not a claim that neurons are concepts." The nearest repo tool is `sparse-probing-search`, which also goes from a labelled concept to neurons.

**Its controls match ours.**

| The paper | This repo |
| --- | --- |
| Twins: two draws of the battery give a noise unit | Verdict stability (`docs/analyses/verdict-stability.md`): three reviews of one submission agree on the status in 67% of pairs. Same-reviewer noise is 21% of the variance in balanced accuracy |
| Scrambled-label and split-half nulls gate the reading | Null calibrations (`docs/analyses/null-calibration.md`): shuffled probes, swapped hypotheses and a rotated basis |
| Untrained references | The untrained-model null (`docs/analyses/untrained-null.md`) |

**Its probe result supports #3559.** On the untrained model, a linear probe on the hidden units already separates A from B at median accuracy 0.802. The trained model reaches 0.998. The probe also "names no neurons", in the author's words. So a dense probe near 1.0 on a wide layer is a weak reading without a random-init floor. That is the case that #3559 makes for `probe-compare`.

**A question it raises, not an issue.** The paper states each neuron's response in units of its own spread on ordinary input, in cells by token class and position. The protocol's line is a fraction of the pole's maximum on the scan (`activation_threshold × scale`). The paper's scale artefact shows that the two scales can move in opposite directions over training. A spread-based line would change the decision rule and need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). That choice is the maintainer's, so the page records it as a question.
