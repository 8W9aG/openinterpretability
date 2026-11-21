# Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling

**Citation.** Thirteen authors, most of them at EleutherAI: Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien and Eric Hallahan. Then Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika and Oskar van der Wal. "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling". *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, pages 2397–2430. [PMLR](https://proceedings.mlr.press/v202/biderman23a.html). The preprint is [arXiv:2304.01373](https://arxiv.org/abs/2304.01373), version 1 of 3 April 2023, revised as version 2 on 31 May 2023.

**Mismatch.** Section 2.2 says the Pile has 300B tokens, and Section 2.4 says it has 334B tokens with the GPT-NeoX tokenizer. The models train on about 300B tokens either way.

**Status.** Used. pythia-70m is a configured registry model. `papers/neuron-development` reads its training checkpoints, and `papers/why-neurons-rest` checks its findings against them.

**Read.** On 2026-11-20 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the PMLR page, the arXiv abstract page and the HTML of version 2. It had the main text and Appendices A to F in the source's own words, in three fetches. The rest of Appendix G, which holds benchmark tables, wasn't read.

## Summary

**Claim.** A suite of 16 models from 70M to 12B parameters, trained on public data in the same order, has 154 checkpoints each. It lets researchers study how models develop over training and scale.

**Method.**

- **Models.** Eight sizes, each trained on the Pile and on a near-deduplicated Pile, so 16 models. Every model sees the same data in the same order.
- **Training.** About 300B tokens over 143,000 steps, at 1,024 sequences of 2,048 tokens per step.
- **Checkpoints.** Step 0, then steps 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512, then every 1,000 steps to 143,000: 154 in all.
- **Case studies.** Swapping gendered pronouns late in training, memorisation as a function of training order, and the effect of a term's frequency on few-shot accuracy.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Pythia-70m (Table 1) | 6 layers, width 512, 8 heads, 18,915,328 non-embedding parameters |
| Checkpoints per model (Section 2.4, Table 2) | 154, against 30 for GPT-Neo and 1 for GPT-2 |
| Gender-pronoun intervention (Section 3.1) | Stereotypical accuracy falls at every scale. At 6.9B the bias flips to anti-stereotypical |
| Memorisation (Section 3.2) | A Poisson model fits it, so "training order has little impact on memorization" |
| Term frequency (Section 3.3) | The link between frequency and accuracy appears only in larger models, after about 65,000 steps |

## Where the claims go beyond the evidence

- **One seed per model.** The authors say that 25 seeds "would cost approximately 10 million USD". So a finding at one size could be particular to its run. PolyPythias later added seeds.
- **The suite was retrained.** Appendix B says the first release, now `pythia_v0`, was retrained with changes to batch size, checkpoints and learning rate, and the models were renamed. The authors reran their analyses.
- **Not all data is seen.** The authors warn that "some data in the Pile may not be seen by the standard Pythia models".
- **Case studies, not tests.** The three studies are each one experiment, with few intervals.

## Use in this repo

- **Our citations check out.**
  - `papers/neuron-development` says Pythia's runs are public at many steps, and that the suite was built to study development over training and scale. Both match the abstract. The 20 checkpoints it reads are all among the 154, and 143,000 is the last.
  - `papers/why-neurons-rest` says Pythia publishes the checkpoints of its training runs. That matches Section 2.4.
  - The thirteen authors, ICML 2023, PMLR 202 and pages 2397–2430 match the PMLR page.
- **One seed is a limit on our checkpoint studies.** A neuron's history in pythia-70m is the history of one run. `papers/neuron-development` notes that PolyPythias adds nine more runs at pythia-70m's size. Our reading: repeating its measures on those runs would show which histories are particular to one seed.
- **The registry reads version 1.** The registry loads `EleutherAI/pythia-70m`, the retrained suite. Our reading: a finding from the old `pythia_v0` weights wouldn't apply.
- **Parallel attention and MLP.** Pythia computes attention and the MLP in parallel from the same input, not one after the other. So in Pythia, a neuron's input doesn't include its own layer's attention output, unlike GPT-2. Our reading: comparisons of neurons across the two families should keep that in mind.
- **No question for the maintainer.**
- **Related pages.** [`the-pile.md`](the-pile.md) reviews the training data. [`gpt-2.md`](gpt-2.md) reviews the registry's other small model, which has one checkpoint.
