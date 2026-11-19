# Extracting Latent Steering Vectors from Pretrained Language Models

**Citation.** Nishant Subramani (Allen Institute for Artificial Intelligence), Nivedita Suresh (Arrive Bio) and Matthew E. Peters (Allen Institute for Artificial Intelligence). "Extracting Latent Steering Vectors from Pretrained Language Models". arXiv 2205.05124, version 1, 10 May 2022. [Paper](https://arxiv.org/abs/2205.05124), [code](https://github.com/nishantsubramani/steering_vectors). The arXiv comment says "Accepted to ACL2022 Findings". The ACL Anthology page gives Findings of the Association for Computational Linguistics: ACL 2022, pages 566–581, DOI 10.18653/v1/2022.findings-acl.48. The paper uses public datasets: Cornell Movie Dialogs, NLTK Gutenberg, Gigaword, WikiText-103, Yelp, STS-B and StylePTB. We found no funding statement. One small mismatch: the HTML puts Suresh's name beside the email address `matthewp@allenai.org`, and Peters's name beside the Arrive Bio address.

**Status.** Planned. The fix campaign that #4784 prepares for would use it, if the maintainer chooses steering as one way to test a fix. No repo code depends on it. The closest code, `openinterpretability/harness/steering.py` and the `neuron-steering-tests` skill, clamps one unit, and the paper optimises a free vector.

**Read.** On 2026-11-18. We read the abstract page and the ACL Anthology page with the `WebFetch` tool. We then downloaded the v1 HTML and converted it to plain text with a script. So we have the full text in the source's own words: Sections 1 to 9, the references and Appendix A with Tables 1 to 12. We did not see the figures as images. The text gives no numbers for Figure 3 (Yelp sentiment transfer), Figure 4 (STS-B by layer) or Figure 6 (the memorization control). We did not read the PDF or the code.

## Summary

**Claim.** A frozen GPT-2 can be made to write a chosen sentence by adding one fixed vector to its hidden states. The authors find such a vector for each sentence by gradient descent. Differences of these vectors move the sentiment of a sentence, and their cosine similarity tracks sentence similarity.

**Method.**

- **Model.** "We use GPT2 as our language model and use its 117M parameter model size." It has 12 layers and hidden size 768.
- **The vector.** For each target sentence, a vector `z_steer` starts from a Xavier-normal draw. Adam then maximises the log probability of the sentence, with the model frozen. Table 6 gives at most 500 steps and learning rates in [0.01, 1.0].
- **Injection.** The vector goes into the embedding layer, after a self-attention layer, after a feed-forward layer, before the LM head, or several of these. The authors add it at the first timestep only or at all timesteps.
- **Decoding.** The model starts from a beginning-of-sentence token. Decoding is greedy, "without assuming a true length", and stops at end-of-sentence or 1024 tokens.
- **Recovery data.** Four domains: movies, books, news and wiki. Sentences go into 8 length bins in words, from 5–10 to 40–128, with 8 sentences per bin. The books subset has 64 sentences, so we infer 256 sentences in all. The metric is BLEU-4 with smoothing, between the target and the decoded text.
- **Sentiment transfer.** The offset is the mean steering vector of 100 target-style sentences minus the mean of 100 source-style sentences. A test sentence's vector gets λ times this offset. A fine-tuned RoBERTa-base classifier scores accuracy, and BLEU-4 against the input scores content ("self-BLEU"). The vector goes after the 7th self-attention layer, at the first timestep.
- **Similarity.** On the STS-B test set, Spearman rank correlation compares the cosine similarity of two sentences' vectors with the human score.
- **Controls.** A "gibberish" fold of 64 sentences of uniform random tokens, and a "shuffled" fold of books sentences with their tokens shuffled.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Recovery, layer 6 after self-attention, all timesteps (Table 1) | BLEU-4 100.0 |
| Recovery, layer 6 after self-attention, first timestep only (Table 1) | BLEU-4 99.80 |
| Recovery, layer 7 after feed-forward, all or first timestep (Table 1) | 100.0 and 99.25 |
| Recovery, embedding layer or LM head, all timesteps (Table 1) | 33.99 and 6.72 |
| Recovery by layer, all timesteps, lr 1 (Table 7) | Self-attention at layers 2–6 and 8 gives 100.00. Self-attention at layer 0 gives 33.25, and at layer 11 gives 86.04. Feed-forward at layer 11 gives 6.29 |
| 8 seeds for each of the 64 books sentences | "for most sentences (63 of 64) all initializations recover the target sentence perfectly" |
| The mean of the 8 seed vectors for each sentence | BLEU-4 99.4 |
| Vector dimension 192, 384, 576, 768 (Table 4) | BLEU-4 43.43, 83.29, 93.93, 100.00 |
| k vectors joined to the input embeddings as a prompt, k = 1, 5, 10, 20, 50 (Table 5) | BLEU-4 81.7, 94.3, 98.7, 98.6, 98.5 |
| STS-B test, layer-7 feed-forward and self-attention (Table 3) | Spearman 52.08 and 52.04. Mean-pooled BERT gives 47.29, GloVe 42.53 and GPT-2 25.92. SBERT-NLI-base gives 77.03 |
| Samples from a per-dimension Gaussian fit to 4000 Yelp vectors (Section 6.2) | Of 24 samples, "5 samples lead to fully-formed sentences" |
| Yelp sentiment transfer (Figure 3) | Only in the text's words: it "performs comparably to Shen et al. (2017)". The figure's numbers are not in the text |
| StylePTB, 19 tasks, λ = 0.25, BLEU-1 (Table 10) | For example, AEM 0.774 against 0.263 for GPT2-finetune, and ARR 0.721 against 0.897 for `Retrieve-Edit` |

The authors also note limits in their own text. Fluency "is often challenging for higher values of λ", and outputs "repeat individual words or phrases". Negative-to-positive transfer is "qualitatively more fluent and accurate" than the other direction. For each sentence, "there exists a radius around it" where nearby vectors decode the same sentence. The ethics statement says the method "does not guarantee removal of toxic content".

## Where the claims go beyond the evidence

**"Extracting" is fitting.** The authors optimise each vector for up to 500 steps against one sentence, with 768 free numbers. Perfect recovery shows that a mid-layer vector can force a sentence. It does not show that the sentence was "already encoded within the model", as the abstract supposes. A comparison with an untrained or random-weight GPT-2 would test that. The paper has none.

**The memorization null has no numbers in the text.** The gibberish and shuffled folds are the paper's only null. Their results are only in Figure 6. The text says recovery is "highest for books, then shuffled, and lastly gibberish", but gives no BLEU values. So we cannot say how large the gap is.

**"Middle layers" is a wide plateau.** The text says recovery is best "in the middle of the network (layers 6 or 7 of 12)". Table 7 shows 100.00 at self-attention layers 2–6 and 8, and at feed-forward layers 3, 4, 5 and 7. The appendix says so itself: "any layer other than the first or last layer achieves nearly perfect recovery". So the result does not single out a layer.

**Configurations are chosen on the test set.** Figure 4 sweeps the layer on "the test split of STS-B". The text then takes "our two best performing configurations" and reports them on the same split in Table 3. The Yelp experiment uses layer 7 after self-attention, and the recovery study uses layer 6. The text does not say how λ = 0.25 was chosen for StylePTB.

**One small model, English only.** All results use GPT-2 117M. The text says the approach "can be directly applied to any transformer-based autoregressive language model decoder", but tests no other model.

**No intervals.** The seed study uses 8 seeds for 64 sentences. We found no confidence intervals, error bars or significance tests. The recovery set is about 256 sentences, and each Yelp offset uses 100 sentences for each class.

**Small slips in the text.** Algorithm 1 updates the vector by adding `lr` times the gradient of the loss, which is ascent. The text says gradient descent, so the sign is probably a typo. Section 3.2 says "sequences up to 128 tokens", but the bins count words from the NLTK word tokenizer.

**Credit.** The authors give a seed study, a dimension sweep, a prompt-based comparison and a memorization control. They state the fluency failure, the weaker positive-to-negative direction and the sampling failure themselves.

## Use in this repo

**Our steering is a different and narrower intervention.** `openinterpretability/harness/steering.py` clamps one unit's activation at 1, 2 and 4 times its own `max_activation`. It uses five fixed neutral prompts, greedy decoding and 24 new tokens. The direction is the unit's own write, and nothing is optimised. The paper adds an optimised 768-number vector. Its direction is free, so it is not a statement about any unit. The paper's results apply to our tests only as a warning about free directions.

**The paper supports our caution about large clamps.** The skill and the harness docstring call drift "only at 4×" off-distribution and weak evidence. The paper finds the same failure at high λ: fluency drops and words repeat. Its "radius" finding also fits. Many nearby vectors decode the same text, so one steered output does not identify one direction.

**A steering success does not locate a cause.** In GPT-2 117M, almost any layer from 1 to 10 can carry a vector that forces a whole sentence (Table 7). So a forced output says little about where the model computes it. This matches the argument of [Does Localization Inform Editing?](does-localization-inform-editing.md): a successful intervention at a site is weak evidence that the behaviour lives there.

**What the paper predicts for the #4781 copying failure.** EleutherAI/pythia-70m answers " days" to "A week has seven days and a year has twelve", with " months" at rank 4. The VERIFIED route copies the first clause's unit through layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months" (#5015). The paper predicts that a vector optimised on this one prompt would make " months" the answer at most middle layers. But that vector fits one sentence, as in Section 3, so it is not a fix. It need not move the other five shortlisted prompts in `docs/analyses/hallucination-shortlist.jsonl`. The paper's offset method is the form that could carry over. It takes a mean difference over two labelled sets, and it worked with 100 examples for each class. This is a prediction from the paper, not a measurement. How to use it is for the fix campaign, which the maintainer designs.

**What a steering test of a fix would need from the paper.** The paper scores transfer by two numbers together: the style classifier's accuracy and BLEU against the input. A fix tested by steering would need the same pair. That is, the target answer on the prompt, and how little the other text changes. Registry coverage (`openinterpretability/coverage.py`) could measure the second part on other prompts. The paper's gibberish fold is also a model for a null. A steering effect means little unless a control direction of the same size fails.

**Gap in our steering samples.** `steering_samples` records baseline, ablated and clamped text, but no control. A clamp along a random direction with the same norm as the unit's write would show how much drift any push of that size causes. The `neuron-steering-tests` skill could then compare the unit's drift against that control. This page proposes a follow-up issue for it.

**Question for the maintainer.** #3209 asks whether steering predictions should become a gate. The paper gives no pre-registered prediction format, and its own claims rest on curves and examples. Should a steering gate require a same-norm random-direction null, as above? That is a protocol decision, so this page records it as a question only.
