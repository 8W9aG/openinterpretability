# Neuron features among GPT-2's transcoder latents

Shabalin and Piotrowski (arXiv 2610.24521) call a transcoder feature a *neuron feature* when its encoder and its decoder both align with one MLP neuron. In late layers of 13 transcoders on gated models, neuron features are 14% to 59% of active features at a threshold of 0.7 (`docs/literature/neurons-in-disguise.md`). If many of the latents in our learned-basis baseline were neuron features, that baseline would partly compare neurons with neurons (`transcoder-baseline.md`). Issue #3608 measures that share for Dunefsky et al.'s GPT-2 small transcoders, `gpt2-small-transcoders`. GPT-2's MLP is GELU, not gated, and the paper doesn't test it. So the share here is new evidence either way.

## Method

`scripts/neuron_features.py` uses the vectors that `open-interpretability crossref` compares (`openinterp/harness/crossref.py`):

- **A latent** reads with its encoder, with `ln_2` folded in and centred. It writes with its decoder.
- **A neuron** reads with `c_fc`, with `ln_2` folded in. It writes `w_out`.
- **The score.** A latent and a neuron pole score the smaller of the read cosine and the write cosine. Each cosine is oriented as crossref orients it: + means the latent reads what drives the pole and writes what the pole writes. So both sides must align with the same neuron and pole. A latent's score is its best pole's. A latent is a neuron feature at threshold t when its score is at least t.
- **Chance.** Random pairs of read and write directions, 4,096 for each layer, scored the same way. Their 99th percentile is 0.108 to 0.109 at every layer.
- **Unsigned.** A second score uses |cos| on each side and no pole, in case the sign convention hides matches.

The rows are in `neuron-features.jsonl`: one for each of the 40 audited latents, and one for each layer's whole dictionary.

## The 40 audited latents

| Threshold | Neuron features | VERIFIED among them | VERIFIED among the other latents |
| --- | --- | --- | --- |
| 0.6 | 2 of 40 | 1 of 2 | 36 of 38 |
| 0.7 | 1 of 40 | 0 of 1 | 37 of 39 |
| 0.9 | 1 of 40 | 0 of 1 | 37 of 39 |

- **Most latents are not neuron features.** The median score is 0.21, and 37 of 40 score between 0.15 and 0.30. That is above chance (0.11) but far from 0.6. A typical latent leans toward some neuron without being one.
- **The neuron features.** L9/F6460 scores 0.94 with L9/N486+. L11/F6643 scores 0.62 with L11/N2188+. The registry has studied neither neuron.
- **The one neuron feature at 0.7 is one of the three failures.** L9/F6460 was NEEDS_REVISION on lookahead consistency. Its hypothesis is a lookahead one: it fires on the token before a likely money noun. One latent can't say whether neuron features fail more often.
- **Unsigned scores change nothing.** No latent gains more than 0.06, and no count in the table changes.

So the learned-basis baseline is not neurons against neurons. At the paper's 0.7, 39 of the 40 audited latents are not neuron features, and 37 of those 39 verified.

## The whole dictionary

| Layer | Median score | Share at 0.6 | Share at 0.7 | Share at 0.9 |
| --- | --- | --- | --- | --- |
| 0 | 0.18 | 1.8% | 0.8% | 0.2% |
| 1 | 0.19 | 2.1% | 0.9% | 0.0% |
| 2 | 0.15 | 4.4% | 3.9% | 1.0% |
| 3 | 0.22 | 8.6% | 6.9% | 1.2% |
| 4 | 0.19 | 7.5% | 6.9% | 3.6% |
| 5 | 0.20 | 9.3% | 8.7% | 5.0% |
| 6 | 0.20 | 9.4% | 8.8% | 5.8% |
| 7 | 0.21 | 9.9% | 9.5% | 7.3% |
| 8 | 0.21 | 10.3% | 9.9% | 8.0% |
| 9 | 0.22 | 10.6% | 10.0% | 7.8% |
| 10 | 0.19 | 8.7% | 8.1% | 4.8% |
| 11 | 0.18 | 5.7% | 4.3% | 1.4% |

Over all 294,912 latents, 6.6% are neuron features at 0.7. The share rises from under 1% at layers 0–1 to 10% at layers 8–9, then falls at the last layers. The paper's late-layer range on gated models is 14% to 59%. This GELU model's late layers reach 4% to 10%.

## Caveats

- **The dictionary counts include dead latents.** The paper counts active features. About 29% of these latents never fire on 200k tokens (`transcoder-baseline.md`). Dead latents keep their initial weights, which no neuron should match, so the share among live latents is likely higher. The 40 audited latents are all live.
- **One transcoder, one model.** The share may differ for other GPT-2 transcoders and for skip transcoders.
- **Weights only.** The paper says that its definition, by weights alone, may not hold up. This check doesn't compare activations.

## Reproduce

    uv run python scripts/neuron_features.py --out docs/analyses/neuron-features.jsonl

It reads the cached transcoders, and takes about 15 seconds on CPU.
