# Sparse Feature Circuits

**Citation.** Samuel Marks (Northeastern University), Can Rager (Independent), Eric J. Michaud (MIT), Yonatan Belinkov (Technion – IIT), David Bau (Northeastern University) and Aaron Mueller (Northeastern University). "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models". arXiv 2403.19647. Version 1 is from 28 March 2024, version 2 from 31 March 2024 and version 3 from 27 March 2025. We read version 3. [Paper](https://arxiv.org/abs/2403.19647), [code and data](https://github.com/saprmarks/feature-circuits), [demonstration](https://feature-circuits.xyz). The arXiv journal reference gives "International Conference on Learning Representations, 2025". We did not check the proceedings page. We saw no funding statement in the text that we read.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use its evaluation design. SHIFT is the closest published case of "explain the units, then ablate the ones that cause the wrong behaviour". No code in the repo depends on it.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v3 HTML for us, in two parts. The tool limits each quote to 125 characters. So we have short verbatim quotes from the abstract, Sections 1 to 4, Section 8 and Appendices E.1, F and H. The rest came back as the model's summary. That includes the method details, the definitions of the baselines and Table 2, which the model read as a table. The node counts in Section 3 came back garbled, and the model rebuilt them. So this page leaves them out. We did not see the figures or the PDF. The model found no seeds, variances or confidence intervals in any part that it read.

## Summary

**Claim.** Sparse autoencoder (SAE) features give circuits of units that people can read, unlike circuits of "polysemantic and difficult-to-interpret units" such as neurons. A person can read such a circuit and ablate the features that carry an unwanted signal. The authors call this SHIFT. It makes a classifier stop using that signal, "even without data that isolate these signals".

**Method.**

- **Models.** Pythia-70M, with SAEs that the authors trained (64 times the model width, ReLU). Gemma-2-2B, with the public Gemma Scope SAEs (8 times the width, JumpReLU).
- **Units.** SAE features on the residual stream and on the MLP and attention outputs. Each SAE also has an error term, the part of the activation that it does not reconstruct. The circuits keep the error terms as nodes.
- **Circuit discovery.** The indirect effect (IE) of each node on a metric comes from attribution patching, a first-order Taylor expansion. A more accurate form uses integrated gradients with 10 steps. Nodes and edges above an IE threshold make the circuit.
- **Circuit metrics.** Faithfulness is the share of the model's metric that the circuit keeps when the nodes outside it are mean-ablated. Completeness is "the faithfulness of the circuit's complement".
- **Tasks.** Subject–verb agreement in four structures: simple, within a relative clause, across a relative clause and across a prepositional phrase. The pairs differ only in the number of the subject (from Finlayson et al., 2021).
- **SHIFT setup.** Bias in Bios: classify a biography's profession. The training set is "ambiguous". It has only male professors and female nurses, so "the unintended signal is perfectly predictive of training labels". The "balanced" test set has all four groups.
- **The classifier.** A linear head, trained by logistic regression on residual activations mean-pooled over tokens. The layer is 4 for Pythia. For Gemma it is 22, the layer "for which the baseline probe generalizes worst".
- **SHIFT steps.** Find the circuit for the classifier's output. A person reads each feature "using the Neuronpedia interface", which shows top activating examples and the feature's direct effect on the logits. The person marks the features that are not about the task. SHIFT zero-ablates them. *SHIFT + retrain* then trains the linear head again on the ambiguous set, with the features still ablated.
- **Baselines.** Concept Bottleneck Probing (CBP, from Yan et al., 2023). Random ablation of the same number of SAE features. A feature skyline that zero-ablates the features "most causally implicated in spurious feature accuracy on the balanced set". A neuron skyline that does the same, but mean-ablates as many neurons. An oracle head trained on the balanced set.
- **Interpretability ratings.** Crowdworkers from the ARENA Slack channel read the tokens on which a unit fired most. They rated neurons and features from 0 to 100.

**Findings, as the source reports them.**

The SHIFT table (Table 2) came from the model's reading of the table. The text reports the counts of judged features verbatim.

| Setting | Result |
| --- | --- |
| Circuits for the classifier | 67 features for Pythia and 46 for Gemma (summary). The authors judged "55 of the Pythia features and 43 of the Gemma features to be task-irrelevant" |
| Pythia, original head | Profession 61.9%, gender 87.4%, worst group 24.4% |
| Pythia, SHIFT | Profession 88.5%, gender 54.0%, worst group 76.0% |
| Pythia, SHIFT + retrain | Profession 93.1%, gender 52.0%, worst group 89.0% |
| Pythia, random ablation, feature skyline, neuron skyline | Worst group 24.4%, 62.9% and 41.5% |
| Pythia, CBP and oracle | Worst group 67.7% and 91.9% |
| Gemma, original head | Profession 67.7%, gender 81.9%, worst group 18.2% |
| Gemma, SHIFT | Profession 76.0%, gender 51.5%, worst group 50.0% |
| Gemma, SHIFT + retrain | Profession 95.0%, gender 52.4%, worst group 92.9% |
| Gemma, random ablation, feature skyline, neuron skyline | Worst group 18.0%, 56.7% and 5.6% |
| Gemma, CBP and oracle | Worst group 86.7% and 93.1% |
| Gemma, SHIFT without retraining | "the feature ablations damage model performance". Retraining the head brings it back |
| Neurons as SHIFT's units | "essentially none of the neurons are interpretable", so the authors did not run SHIFT on neurons |
| Interpretability ratings, 0 to 100 (Table 7, summary) | Neurons 32.6, 30.2 and 36.0. Features 52.8, 62.3 and 81.5. The settings are random units, agreement circuits and Bias in Bios |
| Error terms | "removing residual stream SAE error nodes severely disrupts the model and curtails its maximum performance" |
| Attribution patching against the true IE (Appendix H) | "generally quite good, but sometimes underestimates the true IEs". The layer-0 MLP and the early residual stream are exceptions (summary). The figure uses 30 inputs |

The summary says that the neuron skyline fails because the neurons that carry gender also carry the profession.

The authors state these limits in Section 8. "The success of our technique relies on access to SAEs for a given model." "Model components not captured by the SAEs will remain uninterpretable after applying our method." "Much of our evaluation is qualitative." "Feature labeling is also a qualitative process".

## Where the claims go beyond the evidence

**One task for SHIFT, with a signal that a person can name.** The unwanted signal in Bias in Bios is gender, a concept that a reader can see in top examples. The paper does not test a signal that has no name, or one that also helps the task. The neuron skyline's failure shows that such shared units exist.

**The authors judged the features themselves.** The text says "we manually interpret each feature". We found no count of judges, no agreement between judges and no blind condition. The judges knew that gender was the target. So SHIFT's result also measures how well its authors read features about a signal that they already knew.

**The neuron comparison is not matched.** The neuron skyline mean-ablates neurons, and the feature methods zero-ablate features. It ablates as many neurons as the feature skyline has features. An SAE feature is sparse, so one feature touches far fewer inputs than one neuron. The paper also dropped SHIFT on neurons on the strength of a judgement that neurons were not interpretable. That judgement came from top activating tokens alone.

**Without retraining, SHIFT on Gemma is below the feature skyline.** Worst-group accuracy is 50.0% against 56.7%, and profession accuracy is 76.0% against 80.8%. Most of the gain on Gemma comes after the authors train the head again. Retraining a linear head on top of ablated features is a second fix. The paper reports it in its own row.

**No seeds or intervals.** The model found no seeds, variances or intervals. The text that we read does not give the sizes of the Bias in Bios sets. The random-ablation row is one draw, as far as the text shows.

**Credit.** The training set leaves no clean group, so no data can isolate the signal. That is a hard test, and the authors chose it on purpose. The random ablation of the same number of features is the right null. It shows that the change does not come from ablating many features. The authors also keep the SAE error terms in the circuit and state how much they matter.

## Use in this repo

**The repo has the first two steps of SHIFT for #4781.** SHIFT finds a circuit and a person reads each node. The #4781 answer experiment gives a VERIFIED route with an explanation for each unit. Its necessity is 0.68, sufficiency 0.62 and patching 0.75, against −0.004 for random sets (`changelog.d/4781.added.md`). Prompt coverage is 93.8%, against 10.1% for matched random units. The route's necessity uses mean ablation (`openinterpretability/answer_review.py`, `docs/PROTOCOL.md` §4). Our reviewer's probes and measures replace SHIFT's unchecked human label.

**But the #4781 failure is not a spurious signal.** SHIFT ablates units that carry a signal the task does not need. The #4781 route copies " days" through the layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. These units do the copying that the model uses on many prompts. No verified unit supplies " months". An ablation can take a signal away, but it cannot add one. So SHIFT predicts that ablating the copy heads cuts " days". It does not predict that " months" then wins. This is a prediction from the paper's method, not a measurement. #5015 asks why nothing supplies " months".

**The paper's test design fits a fix experiment.** SHIFT trains on a set with no clean group and tests on all groups. It reports the worst group, and it compares against random ablations of the same size. For #4781 the six prompts in `docs/analyses/hallucination-shortlist.jsonl` all copy the first clause's unit. They could serve as groups. The answer experiment's matched random sets already give the null. How to use these is for the fix campaign, which the maintainer designs.

**The neuron result does not transfer as stated.** The paper's crowdworkers read top activating tokens, and they rated neurons 30 to 36 out of 100. The registry reads a neuron pole with adversarial probes, counterfactuals and causal scores. `native-neuron-audit` and [neuron-basis-circuits.md](neuron-basis-circuits.md) give the evidence that native neurons can be read. The neuron skyline is also not matched to the feature methods. So the paper gives no fair test of a fix on native neurons.

**No Pythia-70M dictionary in the repo.** `openinterpretability/harness/dictionaries.py` has presets for GPT-2, Gemma-2-2B, Llama, Qwen3 and Qwen2.5, but none for Pythia-70M. The authors released Pythia-70M SAEs. A preset for them would give #4781's model a learned-basis baseline. Then a fix on SAE features and a fix on native units could run on the same prompt. We did not check which Pythia-70M checkpoint their SAEs fit. The #4781 model is `EleutherAI/pythia-70m`.

**Ablation type.** `activation-patching-practice` says that zero is off-distribution for a unit with a non-zero baseline. SHIFT zero-ablates SAE features, whose baseline is near zero, and mean-ablates neurons. Our route measures use mean ablation. A fix experiment that compares units of the two kinds must give each kind its own baseline, as the paper does.

**Question for the maintainer.** Should a fix experiment report a "+ retrain" row apart from the fix itself, as SHIFT does? On Gemma, most of SHIFT's gain is in that row. It is a campaign design decision, so this page records it and opens no issue.
