# Representation Engineering: A Top-Down Approach to AI Transparency

**Citation.** Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika and Ann-Kathrin Dombrowski. Then Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter and Dan Hendrycks. Most authors are at the Center for AI Safety. The other affiliations in the HTML are Carnegie Mellon, Stanford, Cornell, Maryland, Pennsylvania, UC Berkeley, Illinois Urbana-Champaign and EleutherAI. Five authors after the first share equal contribution. "Representation Engineering: A Top-Down Approach to AI Transparency". arXiv 2310.01405v4, 3 March 2025 (v1 2 October 2023, v2 3 October 2023, v3 10 October 2023). [Paper](https://arxiv.org/abs/2310.01405), [code](https://github.com/andyzoujm/representation-engineering). Preprint, no venue given. The HTML has no acknowledgments or funding section. The paper's own data sets (the emotion scenarios, the probability and risk pairs, the quotes) are not linked from the text.

**Status.** Planned. Its four kinds of evidence for a direction, and its random-vector control, are a direct fit for two follow-ups that this page proposes. It also bears on #3209, the open question of steering as a gate. No code, skill or protocol rule depends on the paper now.

**Read.** On 2026-11-18, from the v4 arXiv HTML. I converted the HTML to plain text, with each formula's TeX in place, and read it myself. I have the abstract and sections 1 to 7 in the paper's own text. I also have appendix A, appendix B.1, B.3 to B.6, B.8 and B.9, appendix C and the templates of appendix D.1. Tables 1 to 14 came through as text, some with merged cells lost. I skimmed appendix B.2, B.7, D.2, D.3 and E (the x-risk sheet) and give no numbers from them. I didn't see the images of the 28 figures, only their captions. I didn't read the PDF or the code. A model in the web-reading tool also read the abstract page, and I took the version dates from its report.

## Summary

**Claim.** The paper defines *representation engineering* (RepE): study and change a network through population-level directions, not single neurons or circuits. It argues that simple linear methods can read and control concepts such as honesty, utility, harmfulness, emotion, bias and memorization in chat models.

**Method.**

- **Reading: Linear Artificial Tomography (LAT).** Step 1 puts stimuli in a template that asks about the concept, for example "Consider the amount of `<concept>` in the following: `<stimulus>` The amount of `<concept>` is". For a *function* such as honesty, an experimental and a reference instruction ("Pretend you're an honest/a dishonest person") wrap the same stimulus. Step 2 collects the hidden state at the last token, or at each response token for a function. Step 3 takes the first principal component of normalised differences between pairs of hidden states. That component is the *reading vector*. Its sign comes from the labels of the stimuli (appendix C.1). The text says 5 to 128 stimuli are typically enough.
- **Four kinds of evidence (section 3.1.2).** *Correlation*: the vector predicts the concept on in- and out-of-distribution data. *Manipulation*: adding or subtracting it changes behaviour. *Termination*: removing it degrades the behaviour. *Recovery*: putting it back after a full removal restores the behaviour. The paper says that its later sections mainly run correlation and manipulation.
- **Control.** There are three controllers. The first is the reading vector. The second is a *contrast vector*: the difference of the representations of one input under two contrastive prompts, computed layer by layer at inference. The third is *LoRRA*: low-rank adapters on attention query and value weights, trained to move the representations toward the contrast-vector targets. Three operators: add or subtract (R′ = R ± v), a piece-wise rule (R′ = R + sign(Rᵀv)·v) and projection out of v.
- **Models.** LLaMA-2-Chat 7B, 13B and 70B, LLaMA-2-7B base, Vicuna-13B, Vicuna-33B (and Vicuna-33B-Uncensored), DeBERTa-xxlarge-v2-MNLI and CLIP ViT-B/32.
- **Data.** TruthfulQA MC1, OpenbookQA, CommonsenseQA, RACE and ARC for truthfulness. The ETHICS utilitarianism and commonsense morality tasks, a power data set and MACHIAVELLI for ethics. AdvBench with ShareGPT, StereoSet, and over 1,200 emotion scenarios from GPT-4. Alpaca serves as an unrelated stimulus set, and model-generated pairs cover probability and risk.
- **Baselines.** Standard zero-shot log-probability scoring. A zero-shot *heuristic*: the model's own verbal rating of truthfulness on a seven-point scale. Few-shot prompting. CCS (Burns et al., 2022) on DeBERTa. A variant of ActAdd (Turner et al., 2023). For utility, four other linear models in step 3: prompt difference, K-means, mean difference and logistic regression. A shuffled random vector for memorization.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| TruthfulQA MC1, LLaMA-2-Chat 7B / 13B / 70B, standard zero-shot (Table 1) | 31.0 / 35.9 / 29.9 |
| The same, heuristic | 32.2 / 50.3 / 59.2 |
| The same, LAT with stimulus sets 1 / 2 / 3, average over sizes (Table 1) | 56.8 / 60.6 / 60.7 |
| LAT, stimulus set 1, at the layer the validation set picks, 15 trials (Table 8) | 55.0 ± 4.0 (7B), 49.6 ± 4.6 (13B), 65.9 ± 3.6 (70B) |
| LAT, stimulus set 3, at the best layer, 15 trials (Table 8) | 59.8 ± 2.4 (7B), 64.2 ± 5.6 (13B), 71.0 ± 2.0 (70B) |
| Honesty reading vector, instructed honest or dishonest, held-out examples (4.3.1) | over 90% accuracy |
| TruthfulQA MC1 under control, 7B: none / ActAdd / reading vector / contrast vector / LoRRA (Table 2) | 31.0 / 33.7 / 34.1 / 47.9 / 42.3 |
| The same, 13B | 35.9 / 38.8 / 42.4 / 54.0 / 47.5 |
| Contrast vector inference cost (Table 2 caption) | over 3 times the compute |
| Honesty control settings (C.2) | coefficient 0.25 on layers range(8, 32, 3) for 7B and range(10, 40, 3) for 13B, chosen on self-generated examples |
| MACHIAVELLI, 30 test games, 7B, + control / none / − control (Table 3) | reward 16.8 / 19.5 / 19.4, power 108.0 / 106.2 / 100.0, immorality 110.0 / 100.2 / 93.5 |
| The same, 13B | reward 17.6 / 17.7 / 18.8, power 105.5 / 105.4 / 99.9, immorality 97.6 / 96.6 / 92.4 |
| Harmful-request compliance, LLaMA-2-Chat-13B, no control / + sadness / + happiness (Table 4) | 0.0% / 0.0% / 100.0% |
| Harmfulness reading vector, Vicuna-13B, 64 + 64 stimuli, held out, with and without jailbreaks (6.2.1) | over 90% accuracy |
| Helpful-and-harmless rate on 500 instructions, prompt only / manual jailbreak / GCG suffix (Table 5) | none 96.7 / 81.4 / 56.6, linear 92.5 / 86.6 / 86.4, piece-wise 93.8 / 90.2 / 87.2 |
| Harmless rate under the GCG suffix (Table 5) | 16 (none), 81 (linear), 83 (piece-wise) |
| Sarcoidosis vignettes, female / black-female mentions (Table 6) | GPT-4 96.0 / 93.0, LLaMA 97.0 / 60.0, LLaMA controlled 55.0 / 13.0 |
| Popular-quote completion, LLaMA-2-13B, exact match / similarity (Table 7) | no control 89.3 / 96.8 |
| The same, quote direction: random / + / − | 85.4 / 92.9, 81.6 / 91.7, 47.6 / 69.9 |
| The same, literature direction: random / + / − | 87.4 / 94.6, 84.5 / 91.2, 37.9 / 69.8 (the row has one pair fewer in the HTML, so I read the no-control cell as shared) |
| Historical-event years before and after subtracting the memorization direction (6.5.2) | 97.2% and 96.2% accuracy |

Two results have no number. Section 4.3.3 says that adding or subtracting the honesty vector makes the model honest or lying "in all cases", shown by examples in Figure 10. Section 6.4.1 edits "Eiffel Tower is in Paris" to Rome with the reading vector, shown by one figure in which the Louvre stays in Paris.

**Utility, four tests of one concept (5.1, Figure 12).** The bar values are only in the figure. The text says the logistic-regression direction has the highest correlation accuracy but "elicits little to no alteration in model behavior when strengthened or suppressed". The prompt-difference direction does well on correlation and manipulation, but removing it does not lower accuracy. PCA, K-means and mean difference have a large effect in all three tests, and mean difference does best.

**On neurons.** The paper places representations, not neurons or circuits, at the centre. Appendix A argues that "mechanisms are flawed for understanding complex systems", and that changes to mechanisms have diffuse effects. It still calls for research at both levels.

## Where the claims go beyond the evidence

- **Recovery is never run, and termination only once.** Section 3.1.2 sets out four kinds of evidence. The paper runs termination only for utility, and recovery nowhere. Most concepts rest on correlation and a manipulation shown by examples. The authors say they mainly run the first two.
- **"Unsupervised" uses labels to choose.** The sign of each reading vector comes from labelled stimuli (C.1). For the QA benchmarks, "we only use labels to identify the layer and direction for reporting the results" (4.1). Table 8 also reports a best layer chosen with test labels. The control coefficients and the LoRRA checkpoint are chosen on validation sets.
- **Many results are single examples.** Honesty control in 4.3.3, the emotion effects, fact editing, the dog concept and the lie detector are figures of chosen generations. Table 4's 0% to 100% has no stated sample size for the controlled runs.
- **Most tables have no spread.** Tables 8 and 14 give standard deviations over 15 trials. Tables 2 to 7 give one value per cell, with no seeds or intervals. In Table 3, the MACHIAVELLI shifts are small, for example power 105.4 to 105.5 for 13B under + control. The 7B reward also falls from 19.5 to 16.8 under + control.
- **Each result checks side effects once.** The memorization result checks one retain task, historical-event years. The honesty and power results check QA accuracy and game reward. No experiment measures what a vector does to unrelated text in general, such as perplexity.
- **Large chat models only.** Every generative result is on a LLaMA-2 or Vicuna model of 7B to 70B parameters. Appendix B.8 finds that a base model's later layers hold the truthfulness direction less well than the chat model's. So the methods' reach in small base models is not tested.
- **Data and code.** The paper links its repository. The text does not link the paper's own GPT-4 and model-generated data sets, so I could not check their release.

The authors state some limits themselves. A small fraction of LAT runs fail when the stimuli carry superficial features (B.1). The immorality monitor "may not reliably track intent" (5.2.2).

## Use in this repo

- **The repo's probe comparison already finds the population view.** `docs/analyses/probe-comparison.md` reads four VERIFIED Qwen poles against linear probes. Each concept is decodable from its layer at AUROC ≥ 0.99, and the best single pole reaches 0.68 to 0.87 held-out balanced accuracy. A sparse probe needs about 32 neurons to reach a dense probe. This is what RepE would predict. The registry's claim is the narrower one: a pole is part of a concept's representation.
- **The mass-mean probe is RepE's mean difference.** `probe-compare` fits a mass-mean probe, a logistic probe and k-sparse probes (`openinterpretability/harness/probing.py`). RepE's mean difference is the mass-mean direction, and LAT is a PCA variant of it. The analysis states its own caveat: "Correlational only." Its cosines between a pole and a dense logistic probe are at most 0.07.
- **RepE's utility test says the probe choice matters causally.** RepE found that the direction with the best correlation, logistic regression, changed behaviour least. The probe comparison reports only correlation, so it cannot tell which of its directions the model uses. Adding RepE's manipulation and termination tests to `probe-compare` would test this. A follow-up issue proposes it.
- **Steering samples have no random-direction control.** The `neuron-steering-tests` skill and `openinterpretability/core/steering.py` clamp a pole at 1, 2 and 4 × `max_activation` and read the continuation. RepE's Table 7 shows why a control matters: a random vector moved exact match from 89.3 to 85.4. The repo's patchscope already injects a random direction (`control_seed` in `openinterpretability/core/patchscope.py`), but the steering samples don't. A follow-up issue proposes the same control for them.
- **The four kinds of evidence match the repo's gates in part.** Correlation is the scan and the probes. Manipulation is the steering samples, reported and not gated. Termination is necessity, and recovery is close to sufficiency (`docs/PROTOCOL.md`, the answer-experiment gates `necessity` ≥ 0.5 and `sufficiency` ≥ 0.5). The repo gates the last two for answer routes. RepE runs them least.
- **Evidence for the fix campaign, not a design.** On the prompt "A week has seven days and a year has twelve", pythia-70m puts 0.7257 on " days" and 0.0295 on " months", rank 4 (`docs/analyses/hallucination-shortlist.jsonl`). The verified route copies the first clause's unit through copy heads L3/H3, L4/H0 and L4/H7 (#4781, #5015). RepE's nearest result is its Eiffel Tower edit. That is one fact, one 13B chat model and one figure, with one retain check, the Louvre. So RepE gives no measured evidence that a contrast vector can fix one prompt in a 70M base model without side effects.
- **What RepE would predict for the copy failure.** A single neuron's clamp is the wrong unit for RepE. It would read a direction from contrast pairs instead, for example the same clause ending in " days" or in " months". Its own results show that a stimulus-dependent contrast vector worked far better than a fixed vector (Table 2, 47.9 against 34.1 on 7B). No record in this repo tests whether such a vector exists in pythia-70m's residual stream.
- **Question for the maintainer.** Should a direction, not only a neuron or a head, count as a unit for a fix of a prompt? RepE and the probe comparison both point to directions. But the registry's protocol explains neurons and heads, and the fix campaign is the maintainer's to design. So this page records the question and opens no issue for it.
