# PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs

**Citation.** Seven authors: Oskar van der Wal (University of Amsterdam), Pietro Lesci (University of Cambridge) and Max Müller-Eberstein (IT University of Copenhagen). Then Naomi Saphra (Harvard University), Hailey Schoelkopf (Anthropic, work done at EleutherAI), Willem Zuidema (University of Amsterdam) and Stella Biderman (EleutherAI). "PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs". [arXiv:2503.09543](https://arxiv.org/abs/2503.09543), version 1 of 12 March 2025, revised as version 2 on 17 May 2025. The arXiv comment says "Published as a conference paper at ICLR 2025". The models are in the Hugging Face collection [EleutherAI/polypythias](https://huggingface.co/collections/EleutherAI/polypythias-67bed6916110c8933e1ea561), for example `EleutherAI/pythia-70m-seed1`. The batch orders are in [EleutherAI/pile-preshuffled-seeds](https://huggingface.co/datasets/EleutherAI/pile-preshuffled-seeds), and the code is in [EleutherAI/pythia](https://github.com/EleutherAI/pythia). StabilityAI gave the compute.

**Mismatch.** None found. Our reference gives the seven authors, 2025 and ICLR 2025, as the arXiv listing does. We didn't check the ICLR page itself.

**Status.** Planned. `papers/neuron-development` reads one training run of pythia-70m, and names the PolyPythias runs of that model as a next step. #5713 holds that work.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). In five fetches it had Sections 1 to 7 and Appendices B, C and E in the source's own words. It saw the figure captions but not the figures. The values in Tables 2 and 3 came through the HTML garbled, so this page uses only the numbers that the text gives.

## Summary

**Claim.** Language-model training is mostly stable across random seeds. Ten runs at each of five sizes show that downstream accuracy, representations and training dynamics agree across seeds. The exceptions are two outlier runs, and the paper shows how to find them from the weights.

**Method.**

- **Models.** Pythia at 14M, 31M, 70M, 160M and 410M parameters. Each size has nine new runs, seeds 1 to 9, beside the original seed 0. That makes 45 new runs and fifty in all.
- **What varies.** Each seed changes "parameter initialisation and batch composition". The data, the 143,000 steps and the hyperparameters stay as in Pythia.
- **Checkpoints.** Each run has Pythia's 154 checkpoints, so about 7,000 are new.
- **Analyses.** Accuracy on seven benchmarks, with agreement between seeds as Cohen's kappa. Gender-bias benchmarks. Minimum-description-length probes on seven linguistic tasks. "Training maps": a hidden Markov model fitted to 14 statistics of the weights over training.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Outlier runs (Section 3), 2 standard deviations on four benchmarks | "only two such combinations: 410M seed 3 and 4". Only these show loss spikes |
| Agreement between seeds (Section 3) | Kappa of about 0.7 near step 1,000, then about 0.5 from step 10,000 |
| Representations across sizes (Section 4) | Mean Pearson r of 0.99 for probe macro-F1 and 0.98 for codelength ratio |
| Training maps (Section 5) | "Only the training map of 410M is predictive of performance (R2=0.99)" |
| Partial runs (Section 5) | "at least 120 k of the 143 k steps are required for a reliable prediction" |

## Where the claims go beyond the evidence

- **Small models only.** The authors say the suite "spans model sizes up to 410 M parameters", for lack of compute. Stability at these sizes may not hold for larger models.
- **No formal outlier test.** The authors say a formal test "would have required a larger sample size (i.e., more tasks)". The two outliers come from a 2-standard-deviation rule.
- **The 410M size is less stable.** The authors say 410M "is less consistent, even when we do not account for the outlier seeds", and leave it for future work.
- **No units.** The probes read whole representations, and the training maps read weight statistics. The paper says nothing about single neurons, or whether units match across seeds.

## Use in this repo

- **Our citations check out.** `papers/neuron-development` says PolyPythias adds "nine more training runs, with other seeds, at five of its sizes", and that pythia-70m is one of them. Section 2 says "For each size, we release 9 additional training runs", and lists 70M. The seven authors, 2025 and ICLR 2025 match.
- **"Its neurons are not the same units" is our own step.** `papers/neuron-development` says that a second run needs its own audited poles. The source doesn't study neurons, so it neither supports nor contradicts this. It follows from the change of initialisation. [`universal-neurons.md`](universal-neurons.md) and [`convergent-learning.md`](convergent-learning.md) review how far neurons match across seeds.
- **The seed-0 run is the same run we read.** The fifty runs count the original Pythia run as seed 0. So the registry's `EleutherAI/pythia-70m` is one of ten runs at its size, and its histories can now be compared with nine others.
- **pythia-70m isn't an outlier size.** The only outlier runs are at 410M. Our reading: the 70M runs make a fair comparison set for the one-run limit in `papers/neuron-development`.
- **Follow-up.** #5713 would repeat the neuron-development measures on the nine other pythia-70m runs. Each run would need its own audited poles first.
- **No question for the maintainer.**
- **Related pages.** [`pythia.md`](pythia.md) reviews the original suite, and [`the-pile.md`](the-pile.md) its data.
