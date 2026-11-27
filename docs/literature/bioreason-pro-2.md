# Building stronger biological reasoning models

**Citation.** Jatin Nainani, Michael Pearce, Eric Bigelow, Archa Jain, Nicholas Wang, Adibvafa Fallahpour, Savitha Srinivasan, Benedict Choi, Arman Seyed-Ahmadi, Parsa Idehpour, Jared Wilber, Peter St. John, John St. John, Bo Wang, Timur Rvachov and Hani Goodarzi. Goodfire Research, with NVIDIA, the Arc Institute and Xaira Therapeutics. "Building stronger biological reasoning models". Blog post, 24 November 2026. [Post](https://www.goodfire.com/research/building-stronger-biological-reasoning-models). There is no paper yet: the post says a preprint is coming, and the model card says "in preparation". The post states no licence, funding or conflicts of interest. The links are live:

- [code](https://github.com/NVIDIA-BioNeMo/BioReason-Pro-2), under MIT.
- the models `nvidia/BioReason-Pro-2-9B-SFT` and `-RL`, under OpenMDW-1.1.
- the [Hypothesis Atlas](https://huggingface.co/datasets/Goodfire/BioReason-Pro-2-Hypothesis-Atlas), 33,533,750 generations over 300,345 proteins, under CC BY 4.0.

**Status.** Not applicable. The registry explains text models, and this work reads a protein–language model through SAE features. It is a worked case of using interpretability to find a fault and fix it through training. So *Use in this repo* compares it with the fix campaign (#5220). It opens no issue.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the post's HTML and turned it into text. It read the whole post verbatim, with the collapsed SMIM32 case study, which the HTML holds in full. It read the labels of all 13 figures from their SVG files, and it looked at the two wet-lab images. It also read the code's README, the model and dataset cards, and their licences. It did not see the 3D structure viewers or the hypotheses explorer, which scripts load. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** The original BioReason-Pro, a language model that reasons about proteins from ESM3 embeddings and text annotations, relied mostly on the text. Interpretability showed why, and three changes to training made BioReason-Pro 2 use the protein embedding and ground its claims in sequence positions.

**Method.**

- **Model.** ESM3 embeddings (layer 37, frozen) go through a learned adapter into a language model, beside GO-GPT and InterPro text. Every stage here is a retrain on Qwen3.5-9B, not the released 4B model.
- **Diagnosis.** The authors replayed one reference trace with ESM3 or GO-GPT blocked, and read the per-token KL. They also removed ESM3 at generation and measured the drop in feature recall. A freezing intervention showed that the updates at protein positions barely mattered, which pointed to a scale mismatch.
- **Fixes.**
  - Drop InterPro from the training inputs.
  - Add about 230,000 detailed reasoning traces. The repo gives 272,688, written by Kimi K3.
  - Put a block RMS norm on the ESM3 embeddings.
- **SAE features.** At layer 23, on the residual stream and the attention input. The width, the training corpus and the sparsity are not given.
- **Judges.** Nemotron 3.5 Super VL scores the evaluation on a fixed 1,000-protein sample. The RL reward is a different judge, Kimi K3, which also wrote the SFT traces. The post names only the first.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Feature-presence recall 0.44 → 0.83 → 0.84 | Without InterPro training, then the data change, then the norm change (Figure 2) |
| Cited-residue precision 0.11 → 0.12 → 0.95 | The same three stages. The norm change makes the difference |
| Feature location within five residues 0.51 → 0.57 → 0.85 | The same three stages |
| Mean judge score 5.62 → 6.44 → 6.83 → 7.24 | All four stages, from the retrain with InterPro training, judged by Nemotron 3.5 Super VL |
| GO recall 0.62 → 0.71 | All four stages |
| Removing ESM3 cuts feature recall by 72%, against 21% in BioReason-Pro | At generation |
| ESM3's share of the per-token KL 0.54 → 0.81 → 0.94 | Across the three training stages, on one replayed trace |
| An RL reward raises one generation's judge score from 6.8 to 7.6 | RL on the judge |
| All 20 LRR proteins score above all 27 controls on one LRR feature | An SAE feature with a fixed scoring rule on a held-out panel |

**Wet-lab check.** The model proposed S-acylation at SMIM32 Cys75. An acyl-PEG-exchange blot shows a hydroxylamine-dependent shift in the wild type and none in C75A. The post gives no replicate count or quantification. The lab used C75A, and the model experiment used C75S, which the authors note. They also say that the model's explanation had an implausible signal-peptide assignment and a wrong protein family. They show that the proposal survives all three signal-peptide readings. The RCDG1 co-IP is from the earlier BioReason-Pro preprint. Only its SAE analysis is new.

## Where the claims go beyond the evidence

- **Association, not mechanism.** The authors say so themselves, for the SAE features and for the input-influence analysis.
- **One run per stage.** No stage is repeated over seeds, and the post gives no intervals.
- **A judge chain the post doesn't show.** Kimi K3 wrote the SFT traces and is the RL reward. Nemotron judges the evaluation, which keeps evaluation apart from the reward. But the post names only Nemotron and never says which model the reward uses.
- **Numbers that differ between sources.** The post says "over 230 thousand" traces, and the repo and model card say 272,688. The headline "44% to 84%" spans the data change and the norm change. The data change alone gives 83%.
- **One trace for the KL shares.** The per-token KL shares come from replaying one reference trace.
- **A wet-lab result with no counts.** The SMIM32 blot has no stated n, replicates or quantification.

## Use in this repo

- **A diagnosis that led to a fix of the training, not of the weights.** The fix campaign (#5220) edits weights at the units that an explanation names. This work found the fault with interventions, a replay with one input blocked and a freezing test. It then fixed the data and one normalisation, and retrained. That is the "fix the input" side, which #5220's prompt arm (E) stands for at test time. It supports reading a fault's cause before choosing where to fix it. It gives no evidence on fixes at single units.
- **Its interventions are not the review's tests.** The replay and blocking measure how much one input matters to the output, as a share of KL. The registry's necessity and sufficiency tests ablate units against a random set of the same size (`docs/PROTOCOL.md` §4). The post has no random baseline for its blocking or freezing.
- **SAE features, with no word on neurons.** The post's analysis uses SAE features. It never mentions neurons, though the same site promotes Goodfire's [Neurons as an interpretable basis](neurons-as-an-interpretable-basis.md). It gives no reason for the choice and no neuron baseline. Its LRR feature test is close to a single-unit audit: a fixed feature, a fixed rule and a held-out panel with controls. The registry doesn't hold this model, so whether native neurons of its Qwen3.5-9B backbone show the same protein–language link is untested here.
- **Judges.** Its evaluation judge differs from its reward judge, which avoids scoring a model on the judge it was trained against. That is the separation that [LLM judges are not fair](llm-judges-not-fair.md) and the [MT-Bench judge](llm-as-a-judge-mt-bench.md) pages ask for. But its trace writer is its reward judge, and the post doesn't say so.
