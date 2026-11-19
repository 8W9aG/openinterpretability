# LoRA Without Regret

**Citation.** John Schulman and Thinking Machines Lab. "LoRA Without Regret". *Thinking Machines Lab: Connectionism* (blog), 29 September 2025. [Post](https://thinkingmachines.ai/blog/lora/), DOI [10.64434/tml.20250929](https://doi.org/10.64434/tml.20250929). The byline says "John Schulman in collaboration with others at Thinking Machines", and the post names no other author. It is a blog post with no peer review. The post gives no code, data or checkpoint links. It states no funding. The acknowledgements thank Dan Alexander Biderman, Weizhu Chen, Daniel Han and Sadhika Malladi for feedback on a draft. The site's menu links to Tinker, a product of the lab, but the post's own text does not name it.

**Status.** Planned. #5125 would use it, through a LoRA arm of its narrow fine-tunes. The post says which LoRA setting matches full fine-tuning, so it sets the arms that such a run should compare. No code, skill or protocol in this repo depends on it. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the post's HTML with `curl` and removed the markup with a scratch script. The reader was this page's author, a model (Claude Opus 5.5). We read all of the text in the source's own words. That covers every section from the introduction to the citation, the sidenotes (the post's footnotes), the figure captions and the one table. The equations came through as KaTeX text, and we read them from that. The nine figures are SVG images, and we did not see them. So for each figure we have only its caption and the numbers that the text states. The text names only Figures 2, 3 and 6. This page numbers the others by their order in the post, which agrees with those three. We did not read the papers that the post cites.

## Summary

**Claim.** LoRA learns with the same sample efficiency as full fine-tuning (FullFT), and reaches the same final loss, in a "low-regret regime". That regime needs two conditions. LoRA must be on all layers, and on the MLP or MoE layers above all. And the adapter must have more capacity than the information in the training data. Reinforcement learning (RL) with policy gradients needs so little capacity that rank 1 matches FullFT.

**Method.**

- **The update.** W′ = W + (α/r)·BA, with rank r, α = 32 and the Hugging Face `peft` defaults. A is uniform with scale 1/√d_in, B starts at zero, and A and B share one learning rate.
- **Models.** Llama 3 models and Qwen3 models, one of them the MoE model Qwen3-30B-A3B-Base. The layer tests use Llama-3.1-8B. The RL tests use Llama-3.1-8B base and Qwen3-8b-base.
- **Supervised data.** One epoch of Tulu3 (instruction following) and a subset of OpenThoughts3 (reasoning). The metric is log loss, not sampled evaluations.
- **RL data.** MATH, GSM8K and DeepMath, with a reward for a right answer. The algorithm is a policy gradient with an importance-sampling term and GRPO-like group centring.
- **Sweeps.** Rank from 1 to 512, against FullFT. A learning-rate sweep for each condition, with a constant schedule and no warm-up. Each curve is the minimum over learning rates at each step.
- **Layers.** LoRA on attention only, MLP only, or all layers. For the MoE model, each expert has its own LoRA, with rank equal to the total rank divided by 8, the number of active experts.
- **Theory.** An argument from the invariances of Adam for why the 1/r factor makes the best learning rate nearly independent of rank. An information argument for RL. A count of FLOPs per pass.

**Findings, as the source reports them.** All numbers are from the post's text, captions or table.

| Setting | Result |
| --- | --- |
| Supervised, Tulu3 and OpenThoughts3, rank sweep (Figure 1) | FullFT and high-rank LoRA have similar curves. Lower ranks leave the minimum-loss curve when the adapter runs out of capacity |
| Same, 1B model (Figure 1 caption) | High-rank LoRA beats FullFT on one dataset and loses on the other |
| Best learning rate, Tulu3 (Figure 2) | FullFT's is lower by a factor of 10 than high-rank LoRA's. Between rank 4 and rank 512 it changes by less than a factor of 2. It is lower at rank 1 |
| Batch size, 10,000 examples of OpenThoughts3 (Figure 3) | LoRA's gap to FullFT grows with batch size, independent of rank. At batch size 32 the gap is smaller and shrinks over time |
| Layers, Llama-3.1-8B and Qwen3-30B-A3B-Base (Figure 4) | Attention-only LoRA does worse than MLP-only LoRA. Attention adds nothing beyond MLP-only |
| Parameter count, Llama-3.1-8B (table) | MLP r = 256: 0.49B. Attention r = 256: 0.25B. All r = 256: 0.70B. MLP r = 128: 0.24B |
| Matched parameters (text) | Attention-only at rank 256 does worse than MLP-only at rank 128 |
| Layers, OpenThoughts3 subset at rank 256, and RL on MATH (Figure 5) | Attention-only does worse than MLP-only, and MLP-only is similar to MLP plus attention |
| RL on MATH and GSM8K, Llama-3.1-8B (Figure 6) | LoRA reaches the same peak as FullFT at ranks as low as 1, with a wider range of good learning rates |
| RL on DeepMath, Qwen3-8b-base, 8192 tokens (Figures 7, 8) | With the best learning rate for each, LoRAs of all sizes and FullFT train in an "almost identical way". The same holds on AIME 2024 and AIME 2025 |
| Information in the MATH run (text) | About 10,000 problems × 32 samples gives 320,000 bits. Rank-1 LoRA on Llama-3.1-8B has 3M parameters |
| DeepSeek-R1-Zero (text) | 10,400 steps × 32 questions × 16 samples gives 5.3M episodes, so 5.3M bits |
| Fit of the best learning rate, 14 Llama and Qwen models on Tulu3 (text) | LoRA multiplier 9.8 over FullFT. LoRA has the same dependence on hidden size as FullFT (LoRA pow = 0) |
| Short runs (text) | "Preliminary evidence" for a multiplier near 15×, under about 100 steps, on "anecdotal evidence" |
| End of training, Tulu3 and OpenThoughts (text) | B has a larger spectral norm than A |
| FLOPs per pass (text) | LoRA uses 2N² + 6NR multiply-adds against 3N² for FullFT, slightly more than ⅔ |
| Hyperparameters (text) | The authors could not improve on the `peft` defaults with α = 32 |

## Where the claims go beyond the evidence

- **"Same performance" means same loss or reward.** The post reads log loss and RL reward only. It does not compare the weights, the representations or behaviour off the training distribution. Shuttleworth et al. find different weight changes at equal accuracy ([`lora-illusion-of-equivalence.md`](lora-illusion-of-equivalence.md)). So equal loss does not show that LoRA and FullFT learn the same thing.
- **No seeds, intervals or tests.** Each condition appears to be one run. The post gives no seeds, no error bars and no test for any gap. The RL result holds only "within the precision limits afforded by the noisiness of RL", in the post's own words.
- **The curves are a minimum over the learning-rate sweep.** Each curve takes, at each step, the best of many runs. That favours every arm in the same way, but it is a lower envelope, not one run. The post does not say whether the loss is on held-out data. With one epoch, each batch is new when the model reads it, which partly answers this.
- **The 1B result disagrees in sign between datasets.** High-rank LoRA beats FullFT on one dataset and loses on the other. The post puts this down to "random variation", with no test.
- **Most numbers live in figures we could not read.** The text gives few numbers for the loss gaps, so the size of each gap rests on the figures.
- **"Consistently 10×" has exceptions in the post.** The best learning rate is lower at rank 1, and short runs want about 15×. The fit gives 9.8 over 14 models, which the post does not name.
- **The information argument compares bits with parameters.** It counts at most one bit per RL episode and compares that with the parameter count. It does not show that rank-1 LoRA stores the information, only that its parameter count is not the limit. The post's rank-1 count (3M) is below the R1-Zero count (5.3M), so that prediction needs a rank above 1.
- **One sidenote drops a word.** The QLoRA sidenote says that "LoRA performed worse than MLP or MLP+attention". It appears to mean attention-only LoRA.
- **No code or data.** The post gives no code, configurations or loss values to check the plots against.

The authors give several limits themselves. They say that the batch-size gap "may not matter as much in practice". They say that one bit per episode "may only apply narrowly to policy gradient algorithms". They say they lack a theory for the 10× ratio, and that their own derivation does not match it. Their open questions say that they cannot "yet make accurate forecasts" of the regime. The FLOPs count omits attention, and they say so.

## Use in this repo

- **The repo has no LoRA or fine-tuning code.** `git grep` for LoRA in `openinterpretability/` and `scripts/` finds only `kv_lora_rank` and `q_lora_rank`. These are attention settings of a small test model in `scripts/moe_experts_check.py`, not adapters. [`fixing-models.md`](fixing-models.md) also notes that the repo has no fine-tuning or weight-edit code.
- **The low-regret setting moves the registry's own weights.** Hu et al. freeze the MLP in every experiment ([`lora-low-rank-adaptation.md`](lora-low-rank-adaptation.md)), so a pole's own W_gate, W_up and W_down stay fixed. This post says that LoRA matches FullFT only when it also adapts the MLP. So a merged adapter that matches FullFT changes the weights of each MLP pole. Results from attention-only adapters do not tell whether a VERIFIED explanation keeps under such an adapter. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) gives the same layer result for LLaMA2-7B.
- **The post's table gives a matched-parameter design for #5125.** A LoRA arm of #5125 could train three adapters on one narrow task: attention-only at rank 256, MLP-only at rank 128, and all layers. On Llama-3.1-8B the first two have about the same parameter count (0.25B and 0.24B). Under attention-only, any pole that fails does so through its inputs, because its own weights are frozen. So the gap between arms shows the share of failures due to a pole's own weights. That matches the Houlsby comment already on #5125 ([`parameter-efficient-transfer-learning.md`](parameter-efficient-transfer-learning.md)). This fits as a comment on #5125.
- **Equal loss is not equal units.** [`papers/neuron-development/README.md`](../../papers/neuron-development/README.md) follows 86 VERIFIED poles of Pythia-70m. The loss is within 1% of its final value by step 16,000. There, the poles correlate with their final form at a median of only 0.62. Adjacent checkpoints after that step correlate at a median of 0.78 to 0.82. So the post's result, that LoRA and FullFT reach the same loss, does not predict that they leave the same poles in place. A keep share for LoRA and for FullFT at matched loss would test that directly.
- **The base-instruct numbers set the scale.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) reports that 252 of 311 poles keep on Qwen2.5-0.5B (81%, 95% CI 76–85%), and 38 of 43 on Qwen2.5-1.5B (88%, 76–95%). The analysis does not record how Qwen tuned these models, so they are not a LoRA result. They are the value that a LoRA arm and a FullFT arm would be compared with.
- **A rank-1 adapter gives each pole one number.** This point is our inference from the post's update rule, not a claim of the post. At rank 1, the change to W_up is (α/r)·b·aᵀ. So neuron i's input row moves by b_i times one shared direction a, and W_gate behaves the same way. For W_down, neuron i's output column moves by a_i times one shared direction b. The adapter thus gives each pole a single loading, readable with no SVD. The post says that rank 1 is enough for RL. A rank-1 RL adapter would let a run test whether |b_i| predicts which poles fail, against a shuffled-pole null. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) makes the general point that all neurons of a layer move in one shared r-dimensional subspace. The open review #4864 (rank-1 LoRAs that encode reasoning signals) bears on the same unit.
- **Record the post's hyperparameters with any LoRA run.** B starts at zero, so a merged adapter at step 0 is the base model exactly, and every pole keeps. The post says that the effective learning rate grows as B grows. So a keep test at several steps would see change start slowly under LoRA and not under FullFT. Record α, the 1/r scale, the initialisation and the learning rate, as the LoRA+ comment on #5125 asks ([`lora-plus.md`](lora-plus.md)).
- **No bearing on #5133 or #5154 beyond the above.** Both read the instruct change Δ, which did not come from a known adapter. The post reads no weight change, so it adds no test to either.
- **A question for the maintainer.** Should a LoRA arm of #5125 compare with FullFT at matched steps or at matched loss? The post matches by steps, with the best learning rate for each arm. The repo's question is about units, and the neuron-development result shows that units move at fixed loss. This choice belongs to the design of #5125, so this page records it as a question and opens no issue.
