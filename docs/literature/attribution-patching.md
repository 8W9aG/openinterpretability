# Attribution Patching Outperforms Automated Circuit Discovery

**Citation.** Aaquib Syed (University of Maryland, College Park), Can Rager (independent) and Arthur Conmy (independent). "Attribution Patching Outperforms Automated Circuit Discovery". [arXiv:2310.10348](https://arxiv.org/abs/2310.10348), version 1 of 16 October 2023, revised as version 2 on 20 November 2023. The arXiv comment says "NeurIPS 2023 ATTRIB Workshop". Code is at [github.com/Aaquib111/acdcpp](https://github.com/Aaquib111/acdcpp). The text gives no funding statement.

**Mismatch.** Version 1 refers to Equation 3 in two places where version 2 refers to Equation 2. Both mean the attribution score.

**Status.** Used. `openinterpretability/pole_effects.py` estimates each pole's effect on the loss by attribution patching, and `open-interpretability next` draws some units by that estimate. `openinterpretability/core/wiring.py` uses it for a unit's upstream links. The `activation-patching-practice` skill cites it.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of both versions for this page's author, a model (Claude Opus 5.5). It had Sections 1 to 6 in the source's own words. The HTML dropped the figures, the appendices and the reference list. So the AUC values, which are only in the figures, weren't seen.

## Summary

**Claim.** Attribution patching, a linear estimate of activation patching, finds known circuits better than ACDC, and much faster. The method is called edge attribution patching (EAP).

**Method.**

- **The estimate.** The change in a metric when an edge takes its corrupted value is estimated as (e_corr − e_clean) times the metric's gradient at e_clean. The paper calls this "a linear approximation to activation patching". It needs "just two forward passes and a backward pass".
- **The circuit.** Rank every edge by the absolute score, averaged over clean and corrupted pairs, and keep the top k.
- **Units.** Edges between attention heads and MLPs. A footnote says the method "also applies to nodes", but "we only use edges in this work".
- **Tests.** Indirect object identification and Greater-Than in GPT-2 small, and a docstring task in a smaller model. The ground truth is each task's published circuit, scored by ROC AUC.
- **Origin.** The paper credits attribution patching to an earlier source, whose entry the HTML didn't show.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| All three tasks (abstract) | "averaged over all tasks our method has greater AUC from circuit recovery than other methods" |
| Indirect object identification and Greater-Than (Figure 2) | EAP "has the maximal AUC" |
| Docstring (Figure 2) | "ACDC used with the KL Divergence metric outperforms EAP" |
| Docstring, attribution against activation patching (Section 5.1) | R² = 0.27. The fitted slope of 0.531 means attribution patching rates edges "twice as important as it really is" |

## Where the claims go beyond the evidence

- **A weak fit where it was checked.** The only direct comparison with activation patching, on docstring, gives R² = 0.27. Activation patching "was too computationally expensive to run" on GPT-2 small.
- **The estimate can mislead.** The authors say EAP overestimates the effect where the metric is concave along the path to the corrupted value.
- **Some metrics fail.** The KL divergence has zero gradient at the clean run, so attribution patching "will not work" with it.
- **Imperfect ground truth.** The authors say the published circuits are "imperfect, increasing the noise in this measurement".
- **No seeds or intervals.** None are reported, and no run times are given in the text.

## Use in this repo

- **Our citations check out.** `papers/syntax-positive-controls` says attribution patching ranks each neuron "by its gradient times its change", and that the top 10 are then measured exactly. That matches Equation 2. Measuring the top ones exactly fits the authors' advice to "run EAP first and then apply ACDC". The three authors, 2023 and the arXiv link match.
- **Neurons are our step.** The source scores edges between heads and MLPs. Scoring single neurons, in `papers/syntax-positive-controls` and `openinterpretability/pole_effects.py`, applies the footnote's "also applies to nodes", which the source doesn't test.
- **Our check fits far better.** [`docs/analyses/pole-effects.md`](../analyses/pole-effects.md) measures 60 poles exactly per model. The Spearman correlation with the estimate is 0.97 on GPT-2 small and 0.85 on Qwen2.5-0.5B. Our reading: mean-ablating one pole is a small change, so a linear estimate suits it better than corrupting a whole edge does.
- **The origin isn't cited.** Our papers cite this paper for the method. Our reading: the method's own origin, which this paper credits, would be a more exact citation. The text we have doesn't name it.
- **No question for the maintainer.**
- **Related pages.** [`locating-editing-factual-associations.md`](locating-editing-factual-associations.md) reviews causal tracing, an earlier form of activation patching.
