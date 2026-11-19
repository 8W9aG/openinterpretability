# Gradient Routing

**Citation.** Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller and Alexander Matt Turner. "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks". arXiv 2410.04332. Version 1 is from 6 October 2024. We read version 2, from 29 November 2024. [Paper](https://arxiv.org/abs/2410.04332), [code](https://github.com/kxcloud/gradient-routing). The page marks the first three authors as equal contributors, and Joseph Miller as a core contributor. The only affiliation on the page is "ML Alignment and Theory Scholars (MATS)". Preprint, no venue given: the arXiv page has no journal reference and no comment. The text says that the work "was financially supported by the MATS program, Manifund, and the Long Term Future Fund (via Effective Ventures)". Neel Nanda gave funding through Manifund for the larger-model runs.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it as a method to compare. The paper puts a capability in a chosen region during training, then removes the region. That is the reverse of our order, where we explain a trained unit first.

**Read.** On 2026-11-18. A model in WebFetch read the abstract page and the first half of the HTML, and gave us a summary. We used that summary only to find the version dates. We downloaded the v2 HTML with curl and read it as plain text, in the source's own words. We read Sections 1 to 6, the acknowledgements and the reproducibility statement. We also read Appendices A, C (text and Table 3), D (text and token list), E, G, H, I and K. We read Appendices B, F, J and L only in part. The text conversion dropped the inline mathematics. So some values are missing, for example the run count in the Figure 5 caption and the mask weight in Appendix E. We did not see the figures, the PDF or the code. Every quote and number on this page is from the downloaded text.

## Summary

**Claim.** Gradient routing masks gradients by data point during backpropagation, so that chosen data trains only a chosen part of the network. The authors say it "isolates capabilities to specific subregions of a neural network". After training, that part can be read, steered or removed.

**Method.**

- **The rule.** The user gives a weight for each edge of the computation graph and each data point. The backward pass multiplies the gradient by that weight. The forward pass does not change. Weights of 1 everywhere give normal backpropagation. In practice, the masks are on the activations of a few layers, and most are binary.
- **Code.** Figure 2 gives the PyTorch form: `x = mask * act + (1 - mask) * act.detach()`.
- **MNIST.** An MLP autoencoder with a 32-dimensional encoding. Digits 0–4 train the top half, and digits 5–9 the bottom half. An L1 penalty on the encoding is part of the loss. "Certificate" decoders, trained on one half each, test where the information is.
- **Steering scalar.** A 20-layer, 303M-parameter Transformer, trained on FineWeb-Edu. On positions that predict " California", gradients go only to residual dimension 0 in layers 6–18. Then they rank tokens by the cosine of their unembedding with that dimension.
- **Expand, Route, Ablate (ERA).** Add new MLP neurons to target layers. Train from scratch, and on forget tokens reduce the learning rate of the original dimensions, possibly below 0. Then remove the new neurons, and fine-tune for a few steps on retain data.
- **ERA on TinyStories.** TinyStories-28M (8 layers), trained on 400,000 stories. The forget set is any story with "tree", "forest" or "woodland", which is 20% of the data. Baselines are data filtering, RMU and DEMix plus ablation. A control model has the expand, ablate and fine-tune steps but no routing.
- **ERA at 0.7B.** A 20-layer model trained on about 13B tokens of FineWeb-Edu, with about half of the WMDP-bio forget set added. 20 virology tokens go to MLP dimensions 0–79 in layers 0–7.
- **Metrics for unlearning.** Unlearning is the rise in forget loss that the method causes. A second metric is the rise that stays after retraining on forget data (64 stories for TinyStories). This page calls it *unlearning after retraining*. The retrain result is the lowest forget loss over the retrain steps.
- **Reinforcement learning.** A 5×5 gridworld with two terminal squares, Diamond and Ghost. REINFORCE trains a mixture-of-experts policy. When an episode is under oversight, its update goes to the expert for the square that the agent reached. The policy can see which squares are under oversight. Baselines are data filtering and naive training.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| MNIST, average of 20 runs (Figure 3) | The Decoder reconstructs all digits. The bottom-half certificate reconstructs only 5–9, and its outputs for 0–4 are "approximately constant". "Run-to-run variation is negligible" |
| Steering scalar, cosine with dimension 0 | " California" ranks first, then "California", " Californ", " Oregon", " Colorado", " Texas", " Florida", " Arizona", " Sacramento" and " Los". The authors routed only " California" |
| TinyStories, labels below 100% (Figure 5, 95% intervals) | "ERA dominates, outperforming even the gold-standard data filtering baseline", on unlearning and on unlearning after retraining. The retain loss gets worse in proportion to the routed data. The text gives no numbers |
| TinyStories, 100% labels | RMU has the largest rise before retraining, and data filtering loses the least after retraining. ERA plus RMU does better after retraining "than either method alone" |
| TinyStories, control model (Figure 12) | "Ablation has a negligible effect on the control model", so the effect of ERA comes from the routing |
| 0.7B, WMDP-bio forget set (Table 1) | Loss 2.596. Ablated 4.283 (+1.687). Retrained 2.778 (+0.182) |
| 0.7B, WMDP-bio without the 20 routed tokens | Loss 2.567. Ablated 4.205 (+1.638). Retrained 2.738 (+0.171) |
| 0.7B, FineWeb-Edu | Loss 2.925. Ablated 4.864 (+1.939). Retrained 2.957 (+0.032) |
| Gridworld, 512 evaluation environments (Figure 6, 95% intervals) | The Diamond expert gets "nontrivial performance at oversight as low as 1%". At 5% oversight it equals the other methods "trained with six times greater oversight" |
| Gridworld, naive training | It learns "to avoid Ghost with oversight and seek nearby Ghost without oversight" |

The authors name the general effect *absorption*. Routing a narrow part of the data makes features that serve a wider task. Those features then lower the loss on related data that was not routed, so the features do not form elsewhere. The effect did not appear for DEMix.

## Where the claims go beyond the evidence

**Absorption is an explanation that the authors propose, not a measured mechanism.** The text says "we posit absorption". The evidence is the outcomes: the California neighbours, and the forget loss without the routed tokens. No experiment measures step (iii), that the features "are not learned elsewhere".

**The 0.7B result has no baseline and no spread.** Table 1 compares the routed model with itself before and after ablation. It has no data-filtering, RMU or unrouted control at this scale. It gives no seeds or intervals. The "Loss" column is the routed model, so the cost of routing against a normal model is not shown.

**The steering scalar has no null.** The paper ranks tokens by cosine with dimension 0. It does not show the same ranking for dimension 0 of a model trained without routing. Steering evidence is five sampled texts for each condition, with no measure.

**The retraining attack is small.** The authors test unlearning after retraining with 64 stories, or two WMDP-bio examples for 20 steps. The paper does not test a larger attack. The authors use the lowest validation loss over the retrain steps, which favours the attack, so that choice is fair.

**The hyperparameters decide the result, and the authors say so.** "Gradient routing's performance is sensitive to its hyperparameters". Appendix A says the clean MNIST split needs "heavy L1 regularization" and the chosen setup. With other settings, "the qualitative results are poorer".

**The oversight result is one toy setting.** The authors state that it "pertain[s] to a simplistic, narrow setting". They also state that the method needs the ablated policy to stay coherent, which "does not hold in general".

**Credit.** The TinyStories study has a no-routing control, many baselines and 95% intervals. The forget loss without routed tokens tests the obvious objection directly. Appendix G separates capabilities from dispositions and names the problem of each.

## Use in this repo

**The paper works in the reverse order to the registry.** The registry explains units of trained models that it did not train. Gradient routing chooses the unit before training, so it needs no explanation. Nobody routed gradients when Pythia-70m trained, so ERA cannot apply to #4781. The paper's related work also warns about the post-hoc order. It says that models pruned to remove a concept "can very quickly relearn the concept", citing Lo et al. (2024).

**The repo can already train small models.** `scripts/train_resting.py` trains small GPT-2s from scratch for `papers/why-neurons-rest/`. Each condition there changes one thing about training. A routing condition is a few lines on the MLP activations, as in the paper's Figure 2. So the paper's guess in Appendix A can be tested here. It guesses that routing many tokens to many neurons could make neurons that "corresponded to cleanly interpretable features". Our audit protocol could then give such a neuron a verdict, with matched random units as the null.

**Our protocol would ask for more than the steering-scalar evidence.** The paper names dimension 0 from the unembedding cosine alone. That is the output view of the `vocab-projection-lookahead` skill. Our verdicts also need minimal-pair causal scores against matched same-layer units (`docs/PROTOCOL.md` §4). A routed neuron would be a positive control for the protocol, because we would know its meaning before the audit.

**What the paper's metrics would add to a fix report.** Two of its tests carry over to any fix of #4781, whatever method makes it. The first is the retrain attack: does the fixed answer survive a few fine-tune steps on the old behaviour? The second is the "sans routed toks" check: does the fix change more than the logit of one token? A fix that only raises " months" would fail the second test. How to use these is for the fix campaign, which the maintainer designs.

**The copy route is a shared capability, so removal is costly.** Appendix G says that a capability is hard to localize when other capabilities share its computation. The verified #4781 route copies " days" through the copy heads L3/H3, L4/H0 and L4/H7. All six prompts in `docs/analyses/hallucination-shortlist.jsonl` copy the first clause's unit. Copy heads are likely to serve many other prompts. So ablating them would be the costly kind of fix that the paper warns about. This is our reading of the paper, not a measurement.

**Question for the maintainer.** Should a fix experiment restrict its fine-tune gradients to the units of a VERIFIED route, as a post-hoc form of gradient routing? The paper does not test routing during a fine-tune of a pretrained model. It is a campaign design decision, so this page records it and opens no issue.
