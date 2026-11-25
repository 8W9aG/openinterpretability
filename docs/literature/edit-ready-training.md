# Training Vision Models for LLM-Driven Weight-Space Programming

**Citation.** Timo Ropinski and Sebastian Kranz, both of Ulm University. "Training Vision Models for LLM-Driven Weight-Space Programming". arXiv:2611.26408v1, 19 November 2026, cs.CV. Preprint, no venue given. [Paper](https://arxiv.org/abs/2611.26408), [project page](https://viscom.uni-ulm.de/wsp). The authors promise the tasks, agent transcripts, weight changes, scores and code, but link no code or data yet. Support: the state of Baden-Württemberg through bwHPC. **Tool disclosure:** Claude Code (Opus and Fable) wrote the method, the tool interface and the analysis, ran the jobs and made the result tables. The authors say they checked the code and the numbers. One of the two main editors under study, Claude Opus 5.5, is from the same model family as that tool.

**Status.** Planned. #4782 would use it in its strand on weight-space programming, as evidence on how a training method changes the edits that an LLM makes. No code, skill or protocol in this repo depends on it. It edits vision classifiers only, and it never measures whether its units are easier to read.

**Read.** On 2026-11-25 a `web-fetch` subagent (Claude) read the arXiv HTML of v1 as text, with formulas as LaTeX. It had the source's own text for the abstract, Sections 1–6, the LLM usage statement, the references and Appendices A–H, with all tables as text. It did not see the figures as images, the bold marks in Tables 1 and 9, the PDF or the project page. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** LLM agents can fix image classifiers by editing their weights directly, with no gradients. A network trained with rehearsed edits, Edit-Ready Training (ERT), is easier for such agents to edit than one trained with plain cross-entropy (ERM).

**Method.**

- **Edited models.** ImageNet-initialised ResNet-18 and ViT-B/16 on Waterbirds, CIFAR-100 (20 superclasses) and PACS, and ResNet-50 on Entity-30. That makes nine settings, with one trained network for each training method and setting.
- **Editors.** GPT-6 Astra and Claude Opus 5.5 are the main editors. GLM-5.3-Flash, DeepSeek-V4.1-Flash and Gemma-4-26B are open-weight editors.
- **Tasks.** There are three kinds. Relabel a concept, such as a bird species or a CIFAR-100 fine class. Fix a confusion between two classes. Fix a subgroup, such as PACS sketches of one class. The agent gets 5 or 8 support images and 64 reference images, but never sees the images.
- **Tools.** The agent reads weights, probes predictions, ranks channel activations by their difference between image groups, and reads each feature unit's contribution to a class margin. It can set, add to or scale entries, rows, channels or tensors anywhere in the network. The budget per edit is 20 probes, 30 weight reads and 8 edit calls.
- **Edit-Ready Training.** A pre-study found that Opus mostly made rank-one edits to the classification head, on the few features that differ most between support and reference images. ERT rehearses that edit during training. For each episode it moves weight from the true class's row to the target class's row along a direction d. The direction is either the top 8 contrast units or the normalised contrast vector. The loss adds the edited head's cross-entropy on query images and a KL term that keeps reference predictions. Gradients pass through the edit into the backbone, and the edit is then thrown away.
- **Baselines.** ERM, label smoothing (LS) and ENN (Sinitsin et al., 2020), which rehearses gradient-step edits. A screen of ten methods with scripted editors chose them. Hyperparameters were tuned with scripted editors only.
- **Metric.** Net score: the success rate on held-out images minus the side-effect rate on other images.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Mean net score, 9 settings, Astra | ERM 0.63 [0.59, 0.66], LS 0.63, ENN 0.71, ERT 0.74 [0.70, 0.78] |
| Mean net score, 9 settings, Opus | ERM 0.60 [0.56, 0.64], LS 0.65, ENN 0.68, ERT 0.72 [0.69, 0.76] |
| ERT over ERM, both editors | +0.12, p < 0.001 over paired edit processes. With the 9 settings as units, p = 0.047 for Opus |
| ERT over ENN | +0.03 (Astra) and +0.04 (Opus), not significant (p = 0.10 and 0.07) |
| Largest gains of ERT over ERM | Waterbirds ViT, Opus +0.44. Entity-30, Astra +0.33. Waterbirds ResNet-18, Astra +0.23 |
| CIFAR-100 relabel and PACS, Opus | Gain of −0.02 to +0.03 |
| Gain against the ERM score, 9 settings | Rank correlation −0.83 (Astra) and −0.82 (Opus) for ERT |
| Open-weight editors, ERT against ERM, 8 settings | GLM −0.01 (p = 0.90), Gemma +0.06 (p = 0.30), DeepSeek +0.03 (p = 0.44) |
| Test accuracy after the final edit, all methods | 3 to 4 points below the unedited network |
| Where edits go | Head weights in 291 of 292 Astra processes and 296 of 308 Opus processes |
| Shape of Opus edits | Rank one (first singular value above 99% of the energy), on 2–8% of the features |
| Top-8 contrast features in an edit | 3.2 to 4.5, against 0.1 to 1.4 by chance |
| Seed ablation | SD between networks 0.00 (upper bound 0.03), against 0.17 between edit processes on one network |

The statistics are percentile bootstrap intervals over edit processes and exact paired sign-flip tests. Most cells have 8 or 9 paired processes.

## Where the claims go beyond the evidence

- **One trained network per cell.** The intervals cover variation across edit processes, given fixed networks. The authors state this themselves. A seed ablation on one setting finds little spread between networks, but that is one setting.
- **ERT is not shown to beat ENN.** The gain over ENN is 0.03 to 0.04 and not significant. So the result supports rehearsing edits in general more than this edit rule in particular.
- **The gain comes from the hard settings.** It shrinks as the ERM score rises, and it is near zero for Opus on CIFAR-100 relabel and PACS.
- **The open-weight editors show no gain.** None of the three differs from ERM significantly. The authors say a prepared network must be evaluated with the editor that will work on it.
- **No random-edit null.** The reference is the unedited network. A random rank-one edit of the same size would show how much of the net score the agent's choice of units explains.
- **No correction for multiple comparisons.** The paper reports many per-cell tests with no correction.
- **Tuning on test settings.** Hyperparameters were tuned on ResNet-18 Waterbirds and CIFAR-100, which are also evaluated settings, though with other seeds and scripted editors.
- **Arms not interleaved.** The ERM and LS arms ran one to three days after the ENN and ERT arms.
- **Code and data are promised, not released.**

The authors state the main limits themselves. They test three backbones and four datasets, and their small cells detect only large differences. Scripted editors chose the settings, and every method loses accuracy.

## Use in this repo

- **It does not say whether units become easier to read.** ERT aims to keep "each concept in a few distinctive units or a clean direction". The paper measures only edit success. It gives no unit labels, no explanations and no dissection. So it answers the issue's first question with "not measured".
- **It does not carry over to the registry's language models.** The edited models are image classifiers, and the edits are rank-one changes to a classification head. The registry audits MLP neurons inside pretrained open language models, and it does not train them. An ERT for language models would need pretraining with rehearsed edits, which this repo can't run on its models.
- **The editing agent works much like the repo's proposer.** Its tools rank units by the activation difference between two image groups, then test a change with forward passes. The proposer reads top contexts and the reviewer writes contrast probes (`docs/PROTOCOL.md`). Both use activations, not weights alone. The paper also finds that the features of an edit "barely predict its score". So a good contrast unit is not the same as a causal one.
- **Weights alone are not the reading here.** The agents probe activations before they edit. That fits `papers/token-set-ceiling/` (#5144), which finds that only 147 of 826 VERIFIED poles (17.8%) are token-set rules that a weight-only check could reach.
- **A random-edit null is what the repo would ask for.** The protocol's `beats_null` gates rank a group or route against random sets of the same size (`docs/PROTOCOL.md`). A random rank-one head edit of the same size would test the paper's main claim the same way. That is the authors' work, not this repo's.
- **The edits fit the patch format.** A rank-one change to a head is a declarative weight alteration of the kind #5229 records. The fix campaign (#5220) is the maintainer's, so this page only records it.
- **Neighbours.** [`rikiyat-weight-space-programming.md`](rikiyat-weight-space-programming.md) (Takehi et al., 2026), which this paper cites, edits Qwen3.5 with a frontier LLM. [`fixing-models.md`](fixing-models.md) covers fixing models by editing. [`network-dissection.md`](network-dissection.md) is the unit-labelling method that the paper cites but does not apply.
- **No new issue.** The paper changes nothing the repo does, and its evidence goes to #4782 only.
