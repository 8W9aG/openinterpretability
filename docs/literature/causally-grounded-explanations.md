# Causally Grounded Explanations

**Citation.** Ajay Pravin Mahale (Department of Interdisciplinary Engineering, Hochschule Trier). "Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations". arXiv 2603.09988v1, 13 February 2026. [Paper](https://arxiv.org/abs/2603.09988). An MSc thesis, with no venue. The code "will be made public after thesis submission", and no link is given. No funding is stated.

**Status.** Not applicable. It studies six attention heads of one circuit in GPT-2 small, on 50 prompts, with no held-out tests. The registry already audits heads, and has an induction-head control. One point is a lesson for our causal metrics.

**Read.** The abstract page and the v1 HTML, on 2026-11-14. A model read the text and reported it to us, with short quotes, all from the paper's own text. We did not see the figures.

## Summary

**Claim.** A pipeline can link circuit analysis to natural-language explanations. First, it finds the causally important heads by activation patching. Then it writes explanations from templates or with an LLM. Last, it scores their faithfulness with sufficiency and comprehensiveness, as in ERASER.

**Method.**

- **Model and task.** GPT-2 small, through TransformerLens, on indirect-object identification (IOI). There are 50 prompts from 2 templates, with corrupted runs that swap the names. The model gets all 50 right.
- **Circuit.** Each head's effect is the share of the logit difference that patching it restores. The top 6 heads form the circuit, the same for every prompt.
- **Explanations.** A template that fills in the head names and attention weights, or an LLM given the same data. The paper does not name the LLM or give its prompt.
- **Faithfulness.** Sufficiency is the cited heads' share of the circuit's effect. Comprehensiveness is 1 − (the logit difference with the cited heads ablated) / (the clean one). The paper does not say how the heads are ablated.
- **Quality.** A checklist of five 0–1 criteria, such as "names specific heads" and "names the indirect object".

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| The 6 heads' share of the logit difference | 61.4%. The top head, L9H9, holds 17.4% |
| Circuit explanations, 50 prompts | Sufficiency 100.0%, comprehensiveness 22.0% ± 17.3%, F1 36.0% |
| Attention-based baseline | Sufficiency 16.7%, comprehensiveness 26.6%, F1 20.6% |
| Random 6 heads | Sufficiency 46.7%, comprehensiveness 28.8%, F1 35.6% |
| Quality, 30 prompts | LLM 99%, template 60% |
| Model confidence against comprehensiveness | r = 0.009 |

The author reads the low comprehensiveness as backup heads that take over when the circuit is ablated.

## Where the claims go beyond the evidence

- **Sufficiency is 100% by construction.** The explanations cite the circuit's heads, and sufficiency is their share of the circuit's effect. So it is 100% (±0.0) whatever the explanation says. Only comprehensiveness measures anything, and there the circuit's 22.0% is below the random heads' 28.8%.
- **The random baseline nearly ties.** Six random heads get an F1 of 35.6%, against the circuit's 36.0%. The paper says that the circuit beats the attention baseline by 75%, but it does not discuss the random one.
- **"Quality" counts the template's features.** The checklist rewards naming heads, attention percentages and both names, which the LLM was given. So 99% against 60% measures how fully the text repeats its input, not whether it is true.
- **Numbers that don't agree.** The arXiv abstract says that LLM explanations are 64% better, and the HTML abstract and the body say 66%. Table 3's 99% against 60% is 65% higher. Section 3.4 lists a conciseness criterion that Appendix D leaves out.
- **One model, one task, one seed, no intervals.** The author lists these limits, and also that there is no human evaluation.

## Use in this repo

**It doesn't study what the registry studies.** The registry tests a free-text explanation of one unit against adversarial probes. This paper explains a fixed set of heads with text built from their own numbers, and it never tests the text on new inputs. Our head audits (`attention-head-audit`) and the induction-head control already cover heads.

**A lesson for causal metrics: a score defined from the explanation's own parts can't fail.** The paper's sufficiency compares the cited heads with the circuit that they make up, so it is 100% for any explanation. Our sufficiency and necessity scores avoid that. They rank a pole's patch effect against the 32 same-layer neurons whose strength changes most similarly (`docs/PROTOCOL.md` §4, *Causal metrics*). So 0.5 means "no more influential than a matched neuron". The paper's random-heads baseline, which nearly ties its circuit, shows why such a comparison is needed.

**Its comprehensiveness result fits the hydra effect.** Ablating the circuit removes only 22% of the logit difference, which the author puts down to backup heads. The protocol freezes downstream MLP outputs during patching for the same reason (§4).
