# Understanding AI from the Inside

**Citation.** "Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety". arXiv 2611.08438, v1 5 November 2026, v2 9 November 2026. [Paper](https://arxiv.org/abs/2611.08438). Preprint, no venue given.

- **Core contributors:** Fazl Barez and Tung-Yu Wu.
- **Other authors, A to F:** Hunar Batra, Gabriele Dominici, Florent Draye, Yanai Elazar and Damiano Fornasiere.
- **Other authors, G to L:** Antía García, Abir Harrasse, Riley Harris, Sara Hooker, Michael Lan and Belinda Z. Li.
- **Other authors, M to S:** Aaron Maiwald, Narmeen Fatimah Oozeer, Naomi Saphra, Or David Shafran and Adi Simhi.
- **Other authors, T to Z:** Robert Trager, Shriyash Kaustubh Upadhyay, Ziyu Yao and Yu Zhao.
- **Senior advisors:** Shay B. Cohen, Mor Geva, Yoshua Bengio, Stella Biderman and Hassan Sajjad.
- **Universities, as the first page prints them:** Oxford, Università della Svizzera italiana, Bar-Ilan, MIT, Boston and Tel Aviv.
- **More universities:** the Technion, George Mason, Edinburgh, and NYU Abu Dhabi with Dalhousie.
- **Other affiliations:** Martian, the Max Planck Institute for Intelligent Systems, LawZero, Adaption Labs, Anthropic, Mila and EleutherAI.
- **Code:** the paper says it releases a repository of notebooks. The PDF's text links only "Notebooks" and "Website", with no address we could read.
- **Funding and tools:** the acknowledgements thank seven colleagues. They name no funding and no use of AI tools.

**Status.** Planned. The survey is a 183-page reading list with no experiments of its own. But its evaluation protocol (Section 7.5.3) and its guardrails for automated interpretability (Section 12.6.2) give a checklist against the registry. One guardrail the registry lacks, a blind human check of a random sample of verdicts, is a new issue (#4034).

**Read.** On 2026-11-15. A model read the abstract page and the v2 HTML through a web-fetch subagent. The HTML came back as the paper's own text up to Section 7.5.2, where the page was cut off. We then extracted the v2 PDF's text locally (183 pages). From it we read Sections 7.5.3, 7.5.4, 12.6, 14 and 15 in the source's own words, with the first page and the acknowledgements. Sections 8 to 11, 13 and the rest of 12 we saw only as headings and in passing. We did not see the figures, the appendices or the notebooks.

## Summary

**Claim.** The survey asks whether understanding how models work makes them safer. Its answer is that the assumption is "largely untested", and that mechanistic interpretability is "not a panacea". It names "the evaluation methodology and safety benchmarks" as "the field's most pressing open problems" (Sections 1.1 and 15).

**Method.** A survey of mechanistic interpretability for transformers, up to mid-2026. A method counts as mechanistic if it engages with model internals and makes causal claims through interventions such as ablation, patching or steering (Section 1.2). It covers feature extraction (sparse autoencoders, transcoders and their rivals), probing, and circuit discovery with its causal checks. It also covers natural-language reading of activations, learning dynamics, data attribution, six research orientations and safety uses. It proposes two things of its own:

- **An evaluation framework** (Section 7.5) with five dimensions: attribution faithfulness, completeness, stability, causal validity and human interpretability. Its protocol has five steps:
  1. Define the scope.
  2. Set baselines: "random directions in activation space, PCA components, individual neurons, and supervised probes".
  3. Apply the method over seeds and settings.
  4. Score all five dimensions.
  5. Report all five, failures included.
- **Guardrails for automated interpretability** (Section 12.6.2), in five layers:
  1. A blind human review of a random sample of explanations, for example 5%.
  2. Consistency and causal checks.
  3. Adversarial tests with features built to fool the pipeline.
  4. Calibrated confidence, and tests on features whose explanation is known.
  5. Rules that keep "automated hypotheses (weak evidence that requires validation)" apart from findings that passed human or causal checks.

**Findings, as the source reports them.** The survey reports numbers from the work it cites, not from its own experiments:

| Claim | Where | Source it cites |
| --- | --- | --- |
| A circuit "typically explains 60–80% of behavioural variance", and is rarely sparse, necessary and sufficient at once | 7.3 | none given |
| Across SAE seeds, "often, fewer than 50% of features remain the same" | 6.5.4 | none given |
| A poorly tuned SAE can have dead features in "up to 10–50% of the dictionary" | 6.5.1 | Bricken et al. (2023) |
| Probing "has outperformed SAEs in concept detection" | 6.9.4 | Wu et al. (2025b) |
| With a transcoder replacing every layer, top-1 predictions match the model's on "only about 50% of prompts" from the Pile | 6.6 | Ameisen et al. (2025) |

On neurons, it says polysemanticity "complicates neuron-level explanations" (Section 4.2). It says a privileged basis lets "a single neuron … serve as a scalar feature" in an MLP hidden layer (Section 5.1). And it says polysemantic neurons do "not, by itself, imply" superposition (Section 5.2). Among the gaps in current evaluation, it lists "missing baselines": gains not compared with "behavioural methods, random activations, or neuron-level analysis" (Section 7.5.1).

## Where the claims go beyond the evidence

**Numbers without a source.** Two numbers bear most on practice: 60–80% of variance for a circuit, and under 50% of SAE features kept across seeds. Neither has a citation in the text we read. A reader can't check their setting.

**A framework that no one has run.** The five dimensions and the guardrails are proposals. The survey doesn't apply them to any method. So it doesn't show that they rank methods differently from current practice. The authors say so themselves. The metrics "are not uniquely correct", and thresholds "may lack theoretical grounding". The framework is "a starting point for standardisation, not a final answer" (Section 7.5.4).

**Explanation scoring has one paragraph.** The automated pipeline it describes verifies an explanation with a simulator's correlation (Bills et al., 2023) and treats a high score as proof. It doesn't discuss detection or fuzzing scores, or that a judge can score items without reading the explanation. It also doesn't discuss whether a score can prove a hypothesis at all. That is a weaker claim than the causal checks it asks for elsewhere.

**Neurons mostly as a baseline.** Neuron-level analysis appears as a baseline that other methods should beat, and polysemanticity as the reason to move past it. The survey cites no study that tests neuron explanations at scale against probes chosen to break them.

## Use in this repo

**Most of the protocol's steps are already here.** Against Section 7.5.3, with numbers from the analyses:

- *Baselines.* The registry already compares neurons with each of the suggested baselines. Random directions are the rotated-basis null (`docs/analyses/null-calibration.md`). Learned dictionaries are compared in `papers/native-vs-learned/`. Probes are compared in `docs/analyses/probe-comparison.md`. An untrained model is a further null: 9 of 20 trained GPT-2 poles verified against 2 of 20 random-init ones (`docs/analyses/untrained-null.md`).
- *Stability.* Three reviews by one reviewer agree on VERIFIED or not for 82% of pairs (κ 0.61, `docs/analyses/verdict-stability.md`). The reviewer crossover separates reviewer from proposer (`docs/analyses/reviewer-crossover.md`).
- *Causal validity.* Every record carries sufficiency and necessity against matched neurons, and causal checks stay part of the review.
- *Report everything.* NEEDS_REVISION and REJECTED records are published like VERIFIED ones, and the home page gives an estimated false-discovery count.

**Most of the guardrails are here as well.**

- *Layer 2 (consistency):* replicate reviews and the crossover above.
- *Layer 3 (adversarial):* the reviewer writes probes meant to break each hypothesis. An explanation that gives the judge orders scored no higher (`docs/analyses/judge-injection.md`: no attack beat chance, and none of 60 judges followed an injected instruction).
- *Layer 4 (known ground truth):* planted neurons, where every explanation named its plant's concept (`docs/analyses/ground-truth-calibration.md`).
- *Layer 5 (hypothesis versus finding):* the status line, signed and reproduced records, and a protocol version on every verdict.

**One guardrail is missing: blind human review of a random sample.** Layer 1 asks experts to review a random share of explanations without knowing which were sampled, flagged or already checked. No human reads a random sample of verdicts here. The judge-injection and ground-truth studies test the pipeline against known answers, but not against a person's reading of real units. The new issue #4034 proposes a small blind audit. A person judges a random sample of VERIFIED and REJECTED records, shown without their status.

**Adversarial plants are a natural next step for the ground-truth study.** Layer 3 builds features meant to fool the pipeline. The planted neurons of #3201 were honest: each fires on its rule. A plant could be built to mislead: it fires on its rule, and also on a rare, unrelated token that its top examples hide. It would test whether the reviewer's probes find what the scan hides. This belongs on the ground-truth study's follow-ups rather than a new issue, and the comment is on #3201.

**The registry has run the neuron baseline that the survey asks for.** The survey treats neurons as a baseline for learned features. The registry's comparison at equal effort (`papers/native-vs-learned/`) found both sides of that. On GPT-2 small, a live transcoder latent verified 37 times in 40, against 20 of 40 positive neuron poles. Counting the latents that never fire, the shares are 61% against 50%, and the gap is not significant. On Qwen2.5-0.5B, neurons in their native basis verified 18 times in 30, against once in a random rotation of the same layers. So the neurons are a privileged basis, as the survey's Section 5.1 allows. Per unit, a dictionary still verifies more often.
