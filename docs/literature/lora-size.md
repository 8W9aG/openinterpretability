# Predicting LoRA Size for Any Dataset

**Citation.** Engram. "Predicting LoRA size for any dataset". *Engram blog*, 24 November 2026. [Post](https://engram.com/blog/lora-size). The byline names the company only, and no individual authors. The acknowledgements thank Henry Smith and Jacob Portes for feedback. It is a blog post with no peer review, and there is no arXiv version. The post links no code and releases no data. Several of its datasets are internal or synthetic, for example Engram's own emails and research notes, and "a synthetic law firm's workspace". It states no licence, no funding and no tool disclosure. The post's suggested citation key is `engram2026loramemorize`.

**Status.** Not applicable. The repo trains adapters only to audit their directions (`papers/lora-units/`), and no code, skill or protocol here chooses an adapter's rank or sizes a weight change. The post measures how many bits an adapter stores, not which units change, so it doesn't bear on a verdict. Two of its ideas carry over as context, and *Use in this repo* gives them.

**Read.** On 2026-11-28 a `web-fetch` subagent read the post for this page's author, a model (Claude Opus 5.5). It returned the text of the whole page in one fetch. That covers the main sections, Appendix A (the theory), Appendix B (the setup), the references and the citation block. Quotes were capped at 125 characters, so the rest came back paraphrased. The sidenote bodies (footnotes 1–27) did not render, so this page can't tell which cited work supports which claim. The charts did not render either. Of the real-data points in Figures 8 and 9, only one is in the text. The recruiting sentence between Figure 8 and the main result is not part of the research.

## Summary

**Claim.** The rank that a LoRA needs for a dataset can be predicted. Measure the dataset and the adapter in one unit, bits under the base model. A low-rank adapter's capacity then follows from the shape of the adapted matrix, with diminishing returns as the rank rises.

**Method.**

- **Bits of a dataset.** A dataset's size is its code length under the base model, $-\sum_i \log_2 p_\theta(x_i \mid x_{<i})$. Datasets longer than the context are extended with YaRN to 2 million tokens and cut into windows of that length. The post calls the result "a slight overestimate".
- **Synthetic data of an exact size.** Random bits are decoded through the base model with arithmetic coding. So each sample has a code length of about $N$ bits under the base model, and it carries only noise that the model can't predict.
- **The theory (Appendix A).** A quadratic expansion of the log-likelihood, with the Hessian approximated by the negative Fisher, and the Fisher factored as a Kronecker product. The whitened update is then treated as a Gaussian matrix $Z$ with i.i.d. entries, and truncated to rank $r$ by Eckart–Young. The equation below gives the capacity that follows.
- **Measuring capacity.** The adapter trains on a synthetic dataset, and the post fits $\mathrm{bits}(T) = \mathrm{bits}_\infty T^c / (T^c + \tau^c)$ to runs of 1,000 steps or more. The asymptote $\mathrm{bits}_\infty$ is the capacity. Forgetting is the bits lost on samples from the base model.
- **Model and training.** Qwen3.5-0.8B. AdamW with a short warm-up, cosine decay, gradient clipping and no weight decay. LoRA's $\alpha$ is fixed at 256 for every rank. The post gives no learning rate, batch size or seed.
- **Matrices.** The theory and the synthetic curves use `v_proj` (512 × 1,024). The depth study adapts `gate_proj`, `up_proj` and `down_proj` in 1 to 24 layers at rank 4. The real-data check adapts layer 23's MLP, or every linear layer.
- **Real data.** Thirteen datasets from 22,000 to 27 million tokens, among them SWE-bench (Django), MATH-500, GSM8K, Tulu 3, No Robots, OpenThoughts3, MMMLU and enwik8. For each, the post predicts a rank from the synthetic curve and trains ranks from half to twice that. The measured rank is the smallest rank that stores 95% of the dataset.

The capacity in bits of an $m \times n$ matrix at rank $r$ is then

$$
C(r) \approx \frac{1}{2 \ln 2} \, \mathbb{E} \sum_{i \le r} \sigma_i(Z)^2 \approx \frac{(\sqrt{m} + \sqrt{n})^2 \, r}{2 \ln 2},
$$

where $\sigma_i(Z)$ is the $i$-th singular value of $Z$. The second form holds at low rank. At full rank the capacity is $mn / (2 \ln 2)$ bits, and a Marchenko–Pastur form gives the whole curve.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Capacity of `v_proj` | Theory, 512 × 1,024 | About 378,000 bits at full rank: 0.48 bits per parameter at full rank, and 1.3–1.4 at low rank |
| Bits per LoRA parameter at low rank | Theory | 1.40 for `v_proj`, 1.32 for `up_proj` (1,024 × 3,584) |
| Measured against predicted capacity | `v_proj`, rank 128, synthetic data (Figure 3) | 23.9 KB stored and 23.9 KB predicted, 0.97 bits per parameter |
| Capacity per parameter against depth | MLPs at rank 4, synthetic data (Figure 4) | 1.5 bits per parameter in one layer, 3.3 in 24 layers, against 1.30 from the theory |
| Forgetting against learning | `v_proj`, rank 512, 16,000 steps (Figure 7) | 45.8 KB learned and 45.8 KB forgotten, a ratio of 1.00 |
| Predicted against measured rank | Engram's own emails, 22,000 tokens, layer 23's MLP (Figure 8) | 3.45 predicted, 4.26 measured |

The post says that the equality of forgetting and learning holds "across all settings examined in the post". It shows the numbers for one run only.

**Where the claims go beyond the evidence.**

- **One real-data point.** The post calls its rank predictions "extremely accurate", but the text gives the numbers for one dataset only. There, the measured rank is 23% above the predicted one. The "within 5%" claim is about the bits of a dataset, not about the rank.
- **The theory is for noise.** The derivation assumes that the data comes from the base model, so the adapter stores only noise. The authors say so, and call the result a "conservative" worst case. Real data has structure that the base model already predicts in part, so the measured rank on real data tests the theory only loosely.
- **The theory misses the depth effect.** With 24 layers, the measured capacity is 3.3 bits per parameter, against 1.30 from the theory. The authors state that the theory "cannot explain these super-additive effects yet".
- **One model, no seeds.** Every run uses Qwen3.5-0.8B. The post gives no seeds, error bars or repeat runs, and no learning rate. The authors expect, but don't test, transfer to other base models.
- **Forgetting at one setting.** The ratio of 1.00 is from one run. Forgetting is measured on samples from the base model, so it is a loss of likelihood, not a loss of any skill.
- **No code and partly private data.** Nobody outside Engram can rerun the internal datasets, and no code is released.

## Use in this repo

- **Bits are not units.** The post counts how much an adapter stores. The repo asks which units an adapter or an edit changes, and whether their explanations still pass. A capacity in bits says nothing about which neurons carry the stored bits. Our rank-1 adapter on Qwen2.5-0.5B wrote dense directions that put no extra weight on the neurons with passing explanations. Only 1 of 20 of its directions passed the review, against 12 of 20 native neurons (`papers/lora-units/`). So the post's measure can't stand in for an audit.
- **A budget for edits, as context only.** The post's low-rank formula gives a worst-case capacity for a rank-one change. Applied by this page's author to pythia-70m's MLP output matrix (2,048 × 512), it gives about 3,300 bits. A ROME edit of one fact changes that matrix with one rank-one update (`papers/rome-keep/`). A single fact needs far fewer bits than that, so capacity is not what limits an edit. ROME doesn't train by gradient on samples, so the post's theory does not strictly apply to it. That step is ours, not the post's.
- **Forgetting equal to learning.** If a fine-tune loses as many bits of the base model as it stores, then a larger fine-tune costs more of the base model's behaviour. Our sequence of narrow fine-tunes fits that direction only loosely. The first stage alone cost the base model 66 of 362 passing explanations, about as many as instruction tuning cost (62) (`papers/sequential-tuning/`). The post's ratio is in likelihood on the base model's own samples, so it can't predict which explanations fail.
- **It agrees with LoRA Without Regret.** That post says that LoRA matches full fine-tuning only when the adapter's capacity exceeds the information in the data (`lora-without-regret.md`). This post gives a way to measure both sides of that comparison, with a theory for noise and a measurement for real data. *LoRA Learns Less and Forgets Less* (`lora-learns-less-forgets-less.md`) finds that LoRA forgets less than full fine-tuning on benchmarks. A bits-for-bits rule would put that down to LoRA learning less.
- **The cited works.**
  - *How much do language models memorize?* (arXiv:2505.24832) measures a language model's capacity in bits per parameter. It is the direct source for the post's units, so #7320 reviews it.
  - *Learning to reason in 13 parameters* (arXiv:2602.04118) trains adapters far smaller than rank 1. It bears on `papers/lora-units/` and #7022, so #7321 reviews it.
  - *MLPs are Hebbians* (arXiv:2607.10034) is about how MLPs store associations. It may bear on how single neurons store facts, so #7322 reviews it.
  - *Pre-training under infinite compute* (arXiv:2509.14786) is about data efficiency in pre-training. It has no bearing on units or verdicts, so it needs no issue.
