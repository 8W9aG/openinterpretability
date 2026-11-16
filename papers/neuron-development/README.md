# When Pythia-70m's VERIFIED neurons form

Pythia-70m's training run is public at many steps, so we can watch the registry's VERIFIED neurons form. We followed its 9 VERIFIED positive poles back through 20 checkpoints, from step 0 to step 143,000. We measured each one with the review's own code. The loss reaches within 1% of its final value by step 16,000. But at that step, the poles' activations correlate with their final form at a median of only 0.48. They keep changing until the end of training. Four poles' recorded explanations hold without a break from step 16,000 or earlier, one of them from step 2,000. The other five hold only from step 64,000 or later, and two only at the last checkpoint.

**Why we wrote it.** The registry explains each neuron as the finished model has it. If a neuron's behaviour forms early and then stays, one explanation describes most of training. If it keeps changing, an explanation is true only of the checkpoint that it was written for. The registry had not yet looked at any neuron across training. **The question.** When does a VERIFIED neuron pole form during training? Does its explanation hold from then on, and does its formation line up with the loss's plateau? It opens research question 7, *When do neurons form during training?* ([research questions](../../docs/research-questions.md)).

![Training loss settles by step 16,000, but the VERIFIED poles keep changing](media/development.svg)

## Results

1. **The loss settles early.** On the same 300 windows, the loss falls from 11.09 at step 0 to 3.62 at step 16,000, within 1% of its final 3.59. Little changes after step 32,000. This is one model and one training run.
2. **The poles keep changing long after the loss settles.** At step 16,000, the median correlation of a pole's activations with its final activations is 0.48, with a range of 0.29 to 0.74. It is 0.72 at step 64,000 and 0.93 at step 128,000. The correlation with the final checkpoint must reach 1 at the end, so a late rise partly reflects how it is measured (*Caveats*).
3. **Explanations settle at different times.** The recorded probes pass the review's own gates without a break:
   - For four poles, from a step between 2,000 and 16,000: L5/N655+ (from 2,000), L5/N674+ (8,000), L5/N1256+ and L1/N535+ (both 16,000).
   - For the other five, from step 64,000 or later. L4/N566+ holds from 64,000, L2/N1181+ and L4/N751+ from 128,000, and L4/N216+ and L4/N812+ only at the last checkpoint.
4. **Formation does not line up with the loss plateau.** Only two poles reach a correlation of 0.8 with their final form before step 128,000, both at step 64,000. That is four times as many steps as the loss needs to settle.

These results do not show why the poles keep changing, or whether the late changes matter to the model's loss. They are for one small model.

## Method

- **Model and checkpoints.** Pythia-70m (Biderman et al., 2023) at 11 early steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512. It is also read at 9 later steps: 1,000, 2,000, 4,000, 8,000, 16,000, 32,000, 64,000, 128,000 and 143,000. Each is loaded with the registry's own loader at revision `step<N>`, in float32.
- **Poles.** The newest VERIFIED record of each Pythia-70m pole in `wiki/` on 2026-11-16: 9 poles, all positive, in layers 1, 2, 4 and 5.
- **Windows.** 300 windows of 64 tokens from the last 200 documents of the registry's scan corpus (`corpus.jsonl`), the same windows at every step.
- **Measurement.** With the review's code (`submission_unit` and `peak_strength`), each pole's peak strength on each window and on each of its recorded probes.
- **Correlation with the final form.** The Pearson correlation between a pole's 300 window peaks at a step and at step 143,000.
- **Does the explanation hold.** A probe is active at or above 0.25 times the pole's scale at that step. That is the review's line. The scale is the record's `max_activation`, times the step's maximum window peak over the last step's. At step 143,000 it is the review's own line. The explanation holds at a step when the hard negatives' and the holdouts' pass rates reach the record's own gates (specificity 0.8 and sensitivity 0.6).
- **Loss.** The mean next-token loss on the 300 windows.

## How each pole forms

| Pole | Holds without a break from | Correlation 0.5 first reached | Correlation at step 16,000 |
| --- | --- | --- | --- |
| L5/N655+ | 2,000 | 32,000 | 0.38 |
| L5/N674+ | 8,000 | 32,000 | 0.37 |
| L5/N1256+ | 16,000 | 8,000 | 0.63 |
| L1/N535+ | 16,000 | 32,000 | 0.45 |
| L4/N566+ | 64,000 | 4,000 | 0.73 |
| L2/N1181+ | 128,000 | 64,000 | 0.48 |
| L4/N751+ | 128,000 | 64,000 | 0.29 |
| L4/N216+ | 143,000 | 4,000 | 0.74 |
| L4/N812+ | 143,000 | 1,000 | 0.73 |

The two orders differ. The poles whose activations take their final shape early (L4/N812+, L4/N216+, L4/N566+) are the ones whose explanations settle last. The poles whose explanations hold early (L5/N655+, L5/N674+) take their final shape late. So passing the probes and matching the final activations measure different things. A pole can do what its explanation says while much of the rest of its activation still changes. Equally, it can keep its overall shape while the detail that the probes test is still moving.

## Caveats

- **One model, one run.** Pythia publishes one training run of each size, so there is no second seed. The 9 poles are the ones the registry has VERIFIED, not a random sample.
- **The correlation is with the end.** A pole's correlation with its final form reaches 1 at the last step by construction. A pole that drifts steadily looks as if it forms late. Correlations between adjacent checkpoints would separate drift from formation, but they are not measured here.
- **The explanations were written for the last checkpoint.** An earlier form of a pole may have a different good explanation. That the recorded one fails early does not mean the pole had no clean function then.
- **The scale is borrowed.** The line at each step scales the record's maximum by the window maximum's change. A pole whose top activations sit outside the 300 windows gets a less exact line.
- **Few steps late in training.** Between steps 64,000 and 143,000 only one checkpoint is read, so the timing of late changes is coarse.

## Reproduce

```bash
uv run python scripts/neuron_development.py --out papers/neuron-development/results.jsonl
uv run python scripts/paper_figures.py
```

The script downloads the 20 checkpoints from Hugging Face, and ran on an M1 Max.

## Next

- Correlate adjacent checkpoints, to separate a pole's steady drift from its formation.
- Run the same test on OLMo, whose training run is also public, once it has VERIFIED poles.
- Measure the local learning coefficient at the same checkpoints. Then test whether the late changes line up with a change in it, rather than with the loss.

## References

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
