# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**Citation.** Julian Minder (EPFL, ETH Zürich) and Clément Dumas (ENS Paris-Saclay, Université Paris-Saclay) are equal first authors, in random order. The other authors are Caden Juang (Northeastern University), Bilal Chughtai and Neel Nanda. The text gives no affiliation for Chughtai and Nanda. "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning". [arXiv:2504.02922](https://arxiv.org/abs/2504.02922), version 4 of 20 February 2026. Version 1 is of 3 April 2025. Venue: the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), as the paper's footer and the arXiv comment state. Code: [jkminder/science-of-finetuning](https://github.com/jkminder/science-of-finetuning) and a fork of [dictionary_learning](https://github.com/jkminder/dictionary_learning). The crosscoders are on the Hugging Face organisation [science-of-finetuning](https://huggingface.co/science-of-finetuning), and the training runs are on [Weights & Biases](https://wandb.ai/jkminder/chat-crosscoders). The work was part of the ML Alignment & Theory Scholars (MATS) program. The text names no other funding.

**Status.** Planned. #3763 would use it: a crosscoder for Qwen2.5-1.5B and its instruct model, beside the neuron-level result in [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md). The paper gives that issue its loss (BatchTopK, not L1) and a check for false model-only latents (Latent Scaling).

**Read.** On 2026-11-19 we downloaded the arXiv PDF of version 4 and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own words the abstract, sections 1 to 5, the contributions and the acknowledgements. We also read Appendices A to N. They hold the glossary, the loss definitions and the derivation of Latent Scaling. They also hold the Llama, medical and reasoning replications, and the training details. We read Tables 1 to 4 as text. We did not see the figures as images. Where a figure's numbers came out of the text extraction, we could not always tell which bar each number labels. So this page leaves out numbers that appear only in figures. We read the qualitative latents of Appendix N (Tables 5 and 6, Figures 22 to 33) only in part, and the references not in full.

## Summary

**Claim.** A crosscoder trained with an L1 sparsity loss labels many latents as chat-only, but most of them also exist in the base model. Two artifacts of the loss cause this: Complete Shrinkage and Latent Decoupling. A per-latent regression, Latent Scaling, detects both. A crosscoder trained with BatchTopK has few such artifacts. Its chat-only latents are causally relevant and interpretable, for example refusal, false information from the user and questions about the model itself.

**Method.**

- **Models and site.** Gemma 2 2B and Gemma 2 2B it, residual stream after layer 13 of 26. The replications use Llama 3.2 1B and Llama 3.1 8B with their chat models. Two narrow fine-tunes follow: Gemma 2 2B it against Meditron3 (medical), and DeepSeek-R1-Distill-Qwen-1.5B against Nemotron-Research-Reasoning-Qwen-1.5B (RL on reasoning).
- **Crosscoder.** One shared encoder over both models' activations, and one decoder vector per latent for each model. The L1 crosscoder weights each latent's activation by the sum of its two decoder norms. The BatchTopK crosscoder keeps the top k·n scaled activations over a batch of n inputs. At inference, a threshold learned from training replaces the top-k.
- **Training.** Expansion factor 32, so 73,728 latents. 100M tokens of FineWeb web text and of LMSYS-Chat chat text. Both crosscoders reach an L0 of about 100. The two crosscoders need different initial decoder norms to converge: 0.05 for L1 and 1.0 for BatchTopK.
- **Model-only latents.** The relative norm difference Δnorm compares a latent's two decoder norms. A latent is chat-only for Δnorm in 0.9–1.0, base-only for 0–0.1 and shared for 0.4–0.6.
- **Latent Scaling.** For a chat-only latent j, a closed-form least-squares fit finds the scale β that best maps f_j(x)·d_j^chat onto a target. The ratio ν_j = β^base / β^chat is near 0 for a chat-specific latent and near 1 for a shared one. With the reconstruction error as the target, the ratio ν^ε detects Complete Shrinkage. With the reconstruction as the target, ν^r detects Latent Decoupling. A latent is chat-specific when ν^r < 0.5 and ν^ε < 0.2. The fits use 50M training tokens.
- **Causal test.** The paper adds the chat decoder's part of a set of latents to the base activation at layer 13. Then it runs the rest of the chat model and measures the KL divergence from the true chat model. The data is 512 chat interactions, with user turns from the dataset of Ding et al. (2023) and responses from the chat model. The paper reports full responses and response tokens 2 to 10, with 95% confidence intervals. The sets are the top and bottom halves of 3,176 latents, ranked by Δnorm or by Latent Scaling.
- **Interpretability.** An automated pipeline (Paulo et al.) explains each latent with Llama 3.3 70B. A detection score then tests the explanation against similar non-activating text, found by sentence embeddings. Manual reading and steering cover a set of BatchTopK chat-only latents.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Latents by Δnorm, Gemma 2 2B (Table 4) | L1: 1,437 base-only, 3,176 chat-only, 53,569 shared. BatchTopK: 5 base-only, 134 chat-only, 62,373 shared |
| Training fit on validation data (Tables 2, 3) | L1: total FVE 84.4%, L0 101.0, 8.1% dead. BatchTopK: total FVE 87.6%, L0 99.48, 12.0% dead |
| False fine-tuning-only latents, ν > 0.6 (Table 1) | Gemma chat: L1 2,132 of 3,176 (67.1%), BatchTopK 1 of 134 (0.7%) |
| Same, other pairs (Table 1) | Llama 3.1 8B: L1 49.5%, BatchTopK 13.4% of 97. Llama 3.2 1B: L1 58.1%, BatchTopK 11.8% of 17 |
| Same, narrow fine-tunes (Table 1) | Medical: L1 235 of 246 (95.5%). Reasoning: L1 58 of 59 (98.3%). BatchTopK finds no fine-tuning-only latents in either |
| L1 chat-only latents with high ν^ε (section 3.1) | Error ratios up to about 0.5, "indicating substantial Complete Shrinkage" |
| Twin latents, L1 (Appendix F) | 109 pairs of a chat-only and a base-only latent with decoder cosine above 0.9. 60% of the pairs fire mostly in different contexts. BatchTopK has no such pairs |
| Δnorm against ν, BatchTopK (section 3.1) | Pearson 0.73 for ν^r and 0.87 for ν^ε, p < 0.01 |
| Full replacement, both crosscoders (section 3.2) | KL falls by 59% over all tokens and by 78% over the first 9 tokens, against the base activation |
| Reconstruction error (section 3.2) | Over full responses, the chat error term alone reduces KL slightly more than the chat reconstruction. Over the first 9 tokens, it does "more than twice worse" |
| Top against bottom half by Δnorm, BatchTopK (section 3.2) | 50% against 6% KL reduction, first 9 tokens |
| Top half by Δnorm, L1 (section 3.2) | "nearly identically or worse" than the bottom half. Latent Scaling's top half nearly matches BatchTopK |
| Template tokens (section 3.3) | 40% of the chat-only latents fire mostly on chat-template tokens |
| Chat activation on template tokens only (section 3.3) | KL 0.239 on full responses and 0.507 on the first 9 tokens. That is equal to or slightly better than the top half of chat-specific latents |
| Shared latents, decoder cosine above 0.9 (Appendix L) | 90% in the L1 crosscoder, 61% in BatchTopK |
| Independent L1 crosscoder of Kissane et al. (Appendix J) | 16,384 latents: 265 chat-only, 14,652 shared, 98 base-only. 17.7% of chat-only latents fall in the central 95% of the shared ν^r distribution |
| Compute (Appendix M) | About 180 H100 GPU-hours for all experiments. Llama 3.1 8B costs about 150–200% more |

The paper also reports that the chat model's activations have larger norms than the base model's (Figure 1, footnote 3).

## Where the claims go beyond the evidence

**One crosscoder per setting, and no seeds.** Each comparison trains one L1 and one BatchTopK crosscoder. The paper reports no second seed, so the L1 and BatchTopK gap has no measure of training noise. Only the KL results have intervals.

**The false-attribution rates use other thresholds than the main text.** Section 3.1 calls a latent chat-specific when ν^r < 0.5 and ν^ε < 0.2. Table 1 calls a latent false when "ν > 0.6", and the text does not say which ν. The paper gives no reason for any of these cut-offs.

**A rate of 0.0% on zero latents.** On the medical and reasoning fine-tunes, BatchTopK finds no fine-tuning-only latents at all. So "0 false fine-tuning-only latents" says nothing about its precision there. The appendix still counts these pairs as support for "the generality" of the result.

**BatchTopK may find fewer model-only latents by design.** Appendix B.3 states that BatchTopK has a bias toward shared latents. A variant without that bias found more chat-only latents, but with more Latent Decoupling. The comparison of 134 latents against 3,176 therefore shows a different trade-off, not only fewer artifacts.

**No ground truth.** The paper has no model with planted chat-specific features. The authors check Latent Scaling against Δnorm, against MSE gains and against KL, but not against a known answer. A latent with low ν can still be a shift of an old feature to new inputs. The authors state this limit themselves.

**The crosscoders do not match in every respect.** They match on L0, but BatchTopK explains more variance (87.6% against 84.4%) and has more dead latents (12.0% against 8.1%). The initial decoder norms differ by a factor of 20. Any of these could change the count of chat-only latents.

**"Highly interpretable" rests mostly on chosen examples.** The refusal, false-information and emotion latents are hand-picked from the chat-only set. The automated scores show that L1 latents with artifacts score lower. They do not show that chat-only latents beat shared latents or a random baseline.

**Small errors in the glossary.** Appendix A defines "chat-specific latents" twice, with two meanings. It also gives ε^base(x) = h^chat(x) − h^base(x), which disagrees with the definition in section 2.1.

**One layer, residual stream only.** All the Gemma results are at layer 13. The authors state that results at other layers, at other widths and at other sparsities are not known. They also state that the reconstruction error still carries much of the chat behaviour. Their related-work section cites findings that tuning changes upper layers most, but the paper does not test depth.

The authors replicate on two Llama sizes, on an independent L1 crosscoder and on two narrow fine-tunes. They release code, weights and training logs.

## Use in this repo

**The repo's base-instruct comparison has no dictionary, so it has none of these artifacts.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) compares each VERIFIED pole with the same index in the instruct model. Neuron *i* is the same position in both models, so no loss can make a false model-only unit. On Qwen2.5-1.5B, 38 of 43 hypotheses keep, at a median correlation of 0.983. On Qwen2.5-0.5B, 252 of 311 keep, at a median correlation of 0.974. The paper studies dictionary latents in the residual stream, and it says nothing about single neurons.

**The paper is the method that #3763 needs.** #3763 asks for a crosscoder on Qwen2.5-1.5B and its instruct model. The paper gives four decisions for it. Use BatchTopK, not the L1 loss of Lindsey et al. Calibrate model-only latents with Latent Scaling (ν^r and ν^ε), not with Δnorm alone. Train on web and chat text with the chat template. Report the reconstruction error's own KL beside the latents'. This belongs in a comment on #3763. The paper's budget is a warning: 100M tokens, 73,728 latents and about 180 H100 GPU-hours for a 2B model. The repo's Qwen2.5-0.5B transcoders used 13.5M tokens on a laptop ([`docs/dictionaries.md`](../dictionaries.md)).

**The repo's own trainer already uses a top-k loss.** `scripts/train_transcoders.py` trains `TopKTranscoder` with an auxiliary loss on dead latents. Then it converts each layer to a JumpReLU with one threshold. That is close to the paper's BatchTopK and its inference threshold (Appendix B.2). A crosscoder for #3763 could reuse that trainer's top-k code and its JumpReLU conversion.

**The chat template is where the paper finds the change.** 40% of the BatchTopK chat-only latents fire mostly on template tokens. The chat activation on template tokens alone gives as low a KL as the top half of chat-specific latents. `scripts/base_instruct.py --chat` reads a pole only over the probe's own characters, "so a template token can't fire it". So the repo's chat run does not look where the paper finds most of the change. A pole that is quiet on the probe text could still change on the template tokens.

**This adds a measure to #5126.** #5126 asks whether instruction tuning moves VERIFIED poles more on chat-formatted text than on plain text. The paper suggests a third reading: each pole on the template tokens only, in both models. This should be a comment on #5126, not a new issue.

**The paper does not bear on #5127's question of depth.** #5127 asks whether the change in the middle third of Qwen2.5-0.5B's layers is a change of the whole layer. The paper studies only the middle layer of each model, so it gives no evidence on where in depth tuning acts. Its note on larger activation norms in the chat model fits a whole-layer change of scale. The repo's median scale ratio of 0.98 on 1.5B and 0.96 on 0.5B is for MLP neurons, not for the residual stream.

**For research question 3, the paper is a caution about dictionary units.** [`docs/research-questions.md`](../research-questions.md) §3 compares native neurons with learned dictionaries at equal effort. The paper shows that a dictionary's loss can make units that mean nothing about the model. In a dictionary, whether a unit "exists" in a model depends on the training objective. A native neuron is a fixed part of the model, so the question does not arise. The paper also finds that the reconstruction error carries at least as much of the chat behaviour as the latents, over full responses. A coverage comparison of dictionaries and neurons should count that error.

**A neuron-level form of the paper's causal test is missing.** The repo correlates each pole's activation across the two models. It does not ask which units carry the behaviour difference. The paper patches chat latents into the base model's activations and reads the KL from the chat model. The same test works on MLP neurons, with the same index in both models and no dictionary. We propose it as a follow-up issue below.

**Evidence for the fix campaign, not a design.** The paper's patch measures how much of a fine-tune's change in output a set of units carries. It also reports that steering the chat model with refusal latents gives a refusal first, then an answer. The same KL patch could tell whether a fix to the #4781 " days" error acts through the VERIFIED poles or elsewhere. This is our inference. [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code.

**Crosscoders could also compare checkpoints.** [`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md) and `open-interpretability sisters` match neurons across models by activation. A crosscoder over two Pythia checkpoints could show features that move between neurons during training. The paper tests only base and fine-tuned pairs, so this is our inference and not a proposal.

**The paper's Latent Decoupling is a form of the superposition that `explain_groups.py` records.** In Latent Decoupling, one concept is one latent in one model and a mix of latents in the other. The repo's superposition entities (`openinterpretability/explain_groups.py`) are sparse directions over one layer's neurons. A pole whose function moves into such a direction after tuning would look, at its own index, like a pole that fails. The repo cannot yet tell those two cases apart.

### Follow-up proposed

- **Which MLP neurons carry the base-to-instruct difference?** Patch the instruct model's MLP neuron activations into Qwen2.5-0.5B at one layer, for the VERIFIED poles and for as many random poles. Read the KL from the instruct model on chat responses, as section 3.2 of the paper does. Report all tokens and tokens 2 to 10. Area: research questions. Priority: low. It is part of #4782 and complements #5118 and #5133.
