# Who Cares About Brackets?

**Citation.** Theo Clark, Alex Roman and Hannes Thurnherr. "Who cares about brackets?". Research report, Apart Research Alignment Jam #10 (Interpretability 3.0), 2023. [PDF](https://webflow.com/files/634e78122252d2e2fc3a9ab9/formUploads/4679d69c-0a9d-4e19-a09a-2638b1b8b133.pdf), 10 pages. Not peer reviewed. The report gives no affiliations and no exact date. Code is in a [Colab notebook](https://colab.research.google.com/drive/1WftRKjZ5sbbVV_RfiNbo56zAeJZH0S5m), which we didn't open.

**Mismatch.** The PDF's metadata title is "Hackathon Report", not the report's title.

**Status.** Not applicable. No code uses it. `papers/syntax-positive-controls` cites it as background, for what earlier work found about brackets in GPT-2 small.

**Read.** On 2026-11-21 this page's author, a model (Claude Opus 5.5), read all ten pages of the PDF as page images. A `web-fetch` subagent first failed to extract the PDF's text. We saw the figures as images, but read no values from their colour scales.

## Summary

**Claim.** The authors found no circuit in GPT-2 small that tells a position it is inside round brackets. Their evidence suggests that the information is spread through the network.

**Method.**

- **Model.** GPT-2 small, through TransformerLens.
- **Proxy.** The probability of `)` as the next token stands for "inside the brackets".
- **Prompts.** Random token strings, 27 sentences written by GPT-4, and the two-token string `()`.
- **Tools.** Attention from later positions to the `(`, patching attention to minus infinity, activation patching with an "inverse" prompt, mean activations over prompts, and cosine similarities.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Attention to `(` (Section 2.1) | Layer 2 head 5, and less so layer 10 head 3, stand out. The authors found "no further evidence" that they matter |
| Blocking the final position's attention to `(` (Section 2.2) | Blocking one head, or every head, made "little difference" |
| `)` straight after `(` (Section 2.3) | Patching attention in layers 0 to 2 brings P(`)`) to "almost 0". A pair, layer 1 head 5 and layer 2 head 1, stands out |
| The same with square brackets (Section 2.3) | The pattern breaks, so it "does not generalise to other forms of 'openness'" |
| Mean activations (Section 2.4) | Patching layers one at a time or in blocks "yielded no clear layers or heads" |

## Where the claims go beyond the evidence

- **A short sprint.** The report comes from a hackathon. It gives no sample sizes for most tests, no seeds and no intervals. Most results are colour maps with no numbers in the text.
- **A proxy, not the concept.** The authors give a whole section to "Predicting a closing bracket is a bad proxy".
- **Heads only.** Every test reads or patches attention. MLP layers and their neurons are never tested.
- **The authors hold back.** They write that they "did not find significant evidence of a circuit responsible for this behaviour". They found "some evidence" that the information is "highly distributed", and are "reluctant to conclude this definitively".

## Use in this repo

- **Our citations check out.** `papers/syntax-positive-controls` cites it three times.
  - The search for a circuit "that tells a position it is inside brackets", in GPT-2 small, matches the abstract.
  - The quote "did not find significant evidence of a circuit responsible for this behaviour" is word for word from the introduction.
  - The "highly distributed" information and the warning about tools match the abstract and the introduction.
  - The three authors, 2023 and the event match the PDF.
- **The neuron fills a gap the report left.** The report tested attention heads only. `papers/syntax-positive-controls` finds a pole in an MLP layer, L2 N676+, that counts open round and square brackets. Our reading: the two results don't clash, because the report never looked at neurons.
- **The same proxy.** `papers/syntax-positive-controls` also measures the closer's probability after an ablation. The report's warning about that proxy applies to it as well.
- **No addendum needed.** That paper is published, and nothing in it needs a correction.
- **No question for the maintainer.**
- **Related pages.** [`universal-neurons.md`](universal-neurons.md) reviews the open-parenthesis suppression neuron that the same paper cites.
