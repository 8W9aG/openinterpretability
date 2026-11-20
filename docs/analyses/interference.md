# The interference report: two runs

Issue #5230. `open-interpretability interference` compares a changed model with its base, at four levels: each VERIFIED unit's probes and output, each recorded route, and behaviour. Each level also runs against two noise floors: the base against itself, and a matched random change of the same size. A change counts only above both. These are the two runs that the issue's *Done when* names, run on 2026-11-19 on MPS. The rows are in `interference/`.

## pythia-70m against a zero-scale patch

The patch adds a random dense matrix to layer 3's down projection, with its scale at zero. So it must change nothing, and the report must say so.

- **Verdict: no known interference.** The footprint is empty, and the token-level KL is 0 on every side.
- **Units.** All 103 VERIFIED poles pass their probes on the base and on the patched model.
- **Routes.** The 2 VERIFIED answer routes stay VERIFIED on every side. The 1 explanation keeps its coverage of 0.819 on every side.
- **Coverage.** The 103 VERIFIED poles are 0.42% of the model's 24,576 poles, and they carry 0.35% of its activation mass on the corpus sample. The unit and route levels are a lower bound on interference, limited by that coverage.

The rows are in `interference/pythia-70m-zero-patch.jsonl`.

## Qwen2.5-1.5B against Qwen2.5-1.5B-Instruct

This run repeats `base-instruct.md` (#3375) inside the report.

| | Value |
| --- | --- |
| VERIFIED poles tested | 59 |
| Reproduced on the base | 59 |
| Keep on the instruct model | 50 |
| Flips that count above both floors | 8 |
| Token KL, base to instruct (nats) | 0.0377 |
| Token KL, base to a random change of the same size | 0.0075 |
| Median activation correlation, VERIFIED poles | 0.986 (lowest 0.953) |
| Median cosine of the direct logit effect, VERIFIED poles | 0.9997 (lowest 0.996) |
| Neurons in the footprint | 250,820 of 250,880 |

**Table 1.** The summary of the run on 20,000 corpus tokens.

**It reproduces `base-instruct.md`.** The registry has 59 VERIFIED poles of Qwen2.5-1.5B now, against 43 on 2026-11-14. 42 of those 43 still stand VERIFIED. All 42 get the same keep result as the first run: 37 keep, and the same five fail. The verdict is *interference*, and it names 8 poles. One more pole flips on the instruct model but also flips under the random change, so it doesn't count.

**Instruction tuning moves nearly every neuron.** The footprint holds 250,820 of the model's 250,880 neurons: their activation or their output moves more than under a random change of the same size. The behaviour KL is five times the random change's too. So for a whole fine-tune, the footprint says only that the change is everywhere, as it is. The registry studies 268 of those neurons. The footprint is written per layer for such a change, with a row for each studied neuron.

**The random floor is not empty.** The random change flips 3 of the 59 poles. One of them also flips on the instruct model, which is why it is left out of the 8. So a keep test with no floor would count flips that a change of the same size, with no structure, also makes.

The rows are in `interference/qwen2.5-1.5b-instruct.jsonl`.

## Limits

- **No routes for Qwen2.5-1.5B.** The registry has no answer route or explanation of it yet, so the route level ran on pythia-70m only.
- **The traffic map isn't in the footprint.** #5231 is still open. When it lands, its flagged units join the footprint.
- **One sample.** Each run read 20,000 tokens of `corpus.jsonl`. The floors cover nondeterminism and size, not the choice of sample.
- **Run time.** The Qwen run took 1 h 41 min on MPS, most of it in the random floor. The floor now draws its noise once, not on every use.

## Reproduce

```bash
uv run open-interpretability interference --model EleutherAI/pythia-70m \
    --patch <a zero-scale draft> --out docs/analyses/interference/pythia-70m-zero-patch.jsonl --device mps
uv run open-interpretability interference --model Qwen/Qwen2.5-1.5B \
    --changed Qwen/Qwen2.5-1.5B-Instruct --out docs/analyses/interference/qwen2.5-1.5b-instruct.jsonl --device mps
```
