# Training Verifiers to Solve Math Word Problems

**Citation.** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun and Łukasz Kaiser. Then Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse and John Schulman. All are at OpenAI. Cobbe and Kosaraju are marked as equal contributors. "Training Verifiers to Solve Math Word Problems". [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), version 1 of 27 October 2021, revised as version 2 on 18 November 2021. Preprint, no venue given. The dataset, GSM8K, is at [github.com/openai/grade-school-math](https://github.com/openai/grade-school-math). The paper states no licence for it. The repository's LICENSE file is the MIT licence, "Copyright (c) 2021 OpenAI". The Hugging Face copy, `openai/gsm8k`, also gives `mit` on its card at commit `740312add88f781978c0658806c59bc2815b9866`. The text names no funding. Appendix A says that contractors from Upwork wrote the first problems, and that Surge AI ran the larger collection. A few-shot prompted 175B GPT-3 model wrote seed questions.

**Status.** Used. `open-interpretability corpus --dataset gsm8k` builds a corpus from GSM8K (`openinterpretability/corpus_presets.py`). `openinterpretability/interference/tuning.py` fine-tunes on it as one narrow task, and `papers/sequential-tuning/` (#5125) reports that run.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the text in two passes, the second from where the first stopped. It had the source's text for the abstract, Sections 1 to 6, the acknowledgements and the references. It also had Appendices A, B (with Table 1), C, E and F. The tool allows only short verbatim quotes, so most of that text came back paraphrased. It did not see the figures. The solve rates of the main results are in Figures 2, 5, 6, 7 and 8, which it could not read. Appendix D's example solutions are images and were not read either.

## Summary

**Claim.** The paper introduces GSM8K, 8.5K grade-school maths word problems written by people. Large language models solve these problems poorly, even after fine-tuning. A verifier that ranks 100 sampled solutions gives about the gain of a model 30 times larger.

**Method.**

- **Dataset (Section 2, Appendix A).** 8.5K problems, split into 7.5K for training and 1K for testing. Each problem takes 2 to 8 steps of basic arithmetic. In the authors' words, "A bright middle school student should be able to solve every problem." Each solution is written in natural language, with calculator annotations (Appendix C).
- **Quality checks.** Other workers solved every problem again. Problems where they disagreed were corrected or discarded. A second check on a subset found disagreement on 1.7% of problems. The authors read this as the share with breaking errors or ambiguities, and they say a larger share could have subtle errors.
- **Models.** GPT-3 models of 6B and 175B parameters, with 3B and 12B in some runs.
- **Fine-tuning baseline (Section 4.1).** Cross-entropy on the training solutions for 20 epochs, then greedy decoding on the test set.
- **Verifiers (Section 4.2).** A generator is fine-tuned for 2 epochs and samples 100 solutions for each training problem. Each solution is labelled correct when its final answer matches. A verifier, a language model with a scalar head, learns to predict that label. By default it scores after every token. At test time, the generator samples 100 solutions at temperature 0.7, and the verifier's top-ranked solution is the answer.
- **Metric.** The test solve rate. test@N is the share of problems solved at least once in N samples.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 6B, fine-tuned without the intermediate steps (Section 4.1) | The solve rate falls from 20.6% to 5.2% |
| 6B with verification, against 175B fine-tuned, full training set (Section 6) | Verification "slightly outperforms" the 175B model, about the gain of a 30 times larger model |
| Extrapolation of the fine-tuning trend (Section 4.1) | About $10^{16}$ parameters would be needed for 80% with the full training set. This is an extrapolation, not a measurement |
| Number of samples ranked, 6B verifier (Section 5.1) | The solve rate rises up to 400 samples, then falls |
| Second agreement check on a subset (Section 2) | 1.7% of problems still disagree |

The main solve rates of fine-tuning and verification at each training-set size are in Figures 2 and 5, which this page did not read. So it gives none of them.

## Where the claims go beyond the evidence

- **One model family.** Every result is on GPT-3 models. The 30 times figure compares two sizes of that family on one test set of 1K problems.
- **Few runs, no intervals.** Figures 2 and 5 show the mean and standard deviation over 3 runs. The 175B verification result in Figure 5 is one run. The text gives no intervals or tests.
- **An outcome label.** The verifier learns from whether the final answer matches. The authors note that a correct answer can come from flawed reasoning, so some positive labels are wrong.
- **The extrapolation.** The $10^{16}$ figure assumes that a log-linear trend continues far past the measured sizes. The authors present it as an estimate.
- **Label noise in the test set.** The authors state the 1.7% disagreement and the chance of more subtle errors themselves.

## Use in this repo

- **GSM8K is a corpus source, kept out of the default blend.** `corpus_presets.py` reads the `main` split of `openai/gsm8k`, its question and answer fields. `blends.py` leaves it out of `mixed`, because it is "one benchmark's short word problems", and open-web-math gives the blend its maths. The paper's description supports that reason: the problems are short, and their style is narrow by design.
- **The sequential-tuning paper's claims match the source.** It calls GSM8K "a grade-school maths problem, answered with the dataset's worked solution". The paper describes grade-school problems with step-by-step natural-language solutions. It also gives the licence as MIT, from the card, and both the card and the GitHub repository say MIT. Its fine-tune reads the training split and holds out the test split, as the paper splits them.
- **The test split may not be held out from Qwen2.5's own training.** GSM8K's test set has been public since 2021. The sequential-tuning paper's held-out loss on GSM8K, 1.15 falling to 0.57 nats on the base model, shows that the stage learned the task. A model that saw the test set in pretraining would have a lower starting loss, but the stage's change would still show. So the result there does not depend on a clean split. No issue follows.
- **The verifier has no counterpart in the registry.** A verifier ranks sampled outputs by a learned score of correctness. The registry's judges score an explanation against a unit's activations, not a model's output, and they don't train on labels. The paper says nothing about units, explanations or nulls, so it gives no test that the registry could adopt.
- **Label noise is small beside the stage's effect.** The 1.7% of problems with breaking errors are about 128 of the 7.5K training problems, if the rate holds over the training split. That noise is too small to explain why 22 explanations first fail at the GSM8K stage on the base model.
