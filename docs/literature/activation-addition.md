# Steering Language Models With Activation Engineering

**Citation.** Alexander Matt Turner (independent researcher), Lisa Thiergart (MIRI), Gavin Leech (University of Bristol), David Udell (independent researcher), Juan J. Vazquez (Arb Research), Ulisse Mini (MATS) and Monte MacDiarmid (Anthropic). "Steering Language Models With Activation Engineering". arXiv 2308.10248. Version 1 is from 20 August 2023. We read version 5, from 10 October 2024. [Paper](https://arxiv.org/abs/2308.10248), [code](https://zenodo.org/records/13879423). The abstract page gives no venue and no journal reference, so this is a preprint with no venue given. The acknowledgements thank "an anonymous ICML reviewer", but the text names no accepted venue. We saw no funding statement. The text lists one Nvidia RTX A5000 GPU and TransformerLens 1.4.0.

**Status.** Used. The `neuron-steering-tests` skill, `docs/PROTOCOL.md` §4 and `openinterpretability/core/steering.py` cite this paper for the steering samples. But the repo clamps one unit, and does not add a contrast-pair vector, so the repo uses the paper's idea and not its method.

**Read.** On 2026-11-18. WebFetch read the abstract page and the v5 HTML, but it returned only summaries. So we downloaded the v5 HTML and converted it to text with a short script that keeps the math alt text. We read that text ourselves, from the abstract to the end of Appendix H, with Tables 1 to 13 as text. All numbers on this page come from that text. We did not see the figures (Figures 1 to 14), so we have no numbers that are only in a plot. We did not read the PDF, the notebook or the code.

## Summary

**Claim.** The difference between a model's activations on two contrasting prompts, such as "Love" and "Hate", is a steering vector. Adding it to the residual stream at inference changes the topic or sentiment of the output, and keeps off-target performance. The method uses no optimization and as few as one prompt pair.

**Method.**

- **Algorithm (Algorithm 1).** Right-pad the two prompts `p+` and `p-` to the same token length. Run each one forward and record the residual stream input to layer `l`. Take the difference and multiply it by an injection coefficient `c`. Add the result to the user prompt's residual stream at layer `l`, starting from the first token ("front" alignment). The authors mask the modified positions, so that only next-token predictions from other positions count.
- **Hyperparameters.** The text says that `c` typically has "a magnitude less than 15". The grid search is "usually between c∈[3,20] and l∈[6,24]", and "This section does not have complete statistics".
- **Models.** GPT-2-XL for perplexity, token probabilities, layer sweeps, topics and ConceptNet. GPT-J-6B and Llama-1-13B for replication. OPT-6.7B and Llama-3-8B for toxicity and sentiment.
- **Running example.** The wedding vector: `p+` = " weddings", `p-` = a space, `l` = 16, `c` = 1, on OpenWebText.
- **Toxicity.** RealToxicityPrompts, a random subset of n = 1,000, sampled 5 times, scored by the Perspective API. Baselines come from Pei et al. (2023), plus FUDGE and Air-Decoding.
- **Sentiment.** Stanford IMDb, n = 1,000. The model continues each review with the opposite sentiment. SiEBERT classifies the output. Disfluency is conditional perplexity under davinci-002. Relevance is the cosine between all-MiniLM-L6-v2 embeddings of the prompt and the continuation.
- **Statistics.** A one-sample t-test against the second-best method, with p < 0.05 shown in bold. Generations use seed 0. The authors checked the qualitative results again on seeds 1 and 2. They chose the sampling settings before the experiments: temperature 1.0, frequency penalty 1.0 and top_p 0.3.
- **Controls.** A random vector with the same per-position norm as the Anger−Calm vector (Appendix G). Prompting with " weddings" instead of steering (Appendix B, Table 9). The embedding part of the vector alone, added at layer 20 (Appendix B).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| GPT-2-XL, wedding vector, layer sweep (§4.1.3) | Effective "at the very first layer", rising "until layer 6", then declining. "For the optimal injection site, we see 90% success in topic steering (compared to a 2% baseline)" |
| GPT-2-XL, perplexity ratio, ActAdd vs prompting (Table 9) | Wedding-related text 0.875 vs 0.890. Unrelated text 0.994 vs 1.132 |
| GPT-2-XL, token log-probability shift (Table 11) | Largest increases: " bride" 1.623, " br" 1.139, " wedding" 1.027. Largest decrease: "Image" −0.370 |
| GPT-2-XL, topics, `c` = 2 (§4.2) | "a large boost in relevance (5-20%) on all topics", except "art". GPT-3.5 scores 100 completions per topic |
| RealToxicityPrompts, OPT-6.7B (Table 3) | Toxicity: unsteered 0.134, ActAdd 0.112, PREADD-D 0.122, prompting 0.200. Disfluency 8.9 unsteered, 13.8 with ActAdd |
| RealToxicityPrompts, Llama-3-8B (Table 3) | Toxicity 0.114 unsteered, 0.108 with ActAdd. Disfluency 6.3 to 6.7. Relevance 0.391 to 0.365 |
| IMDb, negative to positive (Table 4) | Llama-3: 0.417 unsteered, 0.669 with ActAdd. OPT: 0.445 unsteered, 0.564 with ActAdd. PREADD-S-OPT 0.624, FUDGE 0.551 |
| IMDb, positive to negative (Table 4) | OPT: 0.175 unsteered, 0.432 with ActAdd (disfluency 24.2). PREADD-S-OPT 0.631 (disfluency 68.4) |
| GPT-2-XL, ConceptNet, n = 29,774 (§4.5) | "a negligible impact on off-target answer probabilities". The number is only in Figure 5 |
| GPT-2-XL, Anger−Calm, `l` = 20, `c` = +10 (Appendix F) | The vector is "nearly ten times the norm of the underlying forward pass". About "90% of the residual stream is determined by the steering vector" |
| GPT-2-XL, random vector at the norm of `c` = +10 Anger−Calm (Appendix G) | "a noticeable shift in the outputs", but still coherent. The anger vector "changes the output tokens less than the random vector does" (Figure 10, KL) |
| Llama-1-13B replication (Appendix E.2) | The same qualitative effect, with "a notable failure to replicate" on the Eiffel Tower, anger and harm examples |

The authors state their limits. The user must choose `c` and `l`. Some vectors fail, and Table 7 shows three. They "have not examined ActAdd's potential impact on reasoning". The method needs access to activations. They "doubt that activation engineering can e.g. teach a model a new skill". They also say that "the Love vector alone is ineffective compared to the Love − Hate vector".

## Where the claims go beyond the evidence

**The best sentiment result depends on the base model.** The best negative-to-positive score, 0.669, is ActAdd on Llama-3. The baselines that it beats run on OPT or GPT-2. On OPT, ActAdd gives 0.564, below PREADD-S-OPT at 0.624. The authors also found an unsteered OPT score in this direction that disagrees with the PREADD paper.

**The authors copy the baseline numbers, and do not measure them again.** Most baseline rows come from other papers, shown in gray. The text says that PREADD used the discontinued davinci-001 for fluency and that ActAdd used davinci-002. The sampling settings differ too. So the disfluency columns may not be on one scale.

**The detoxification gain on Llama-3 is small.** Toxicity goes from 0.114 to 0.108, about 5%. The fluency cost is small too. Five repeated samples of one subset give the p-values. The text reports no intervals.

**The authors select the qualitative results.** Table 6 shows "the subjectively most compelling completion pair out of the first three". The authors left out 1 of 12 vectors because its completions were "unusually unimpressive". They state this selection openly.

**Some numbers in the text disagree with the tables.** Appendix E.3 gives "a 21% absolute increase in positive classification" for OPT. Table 4 gives 0.445 to 0.564, which is 11.9 points. Section 3 says "intervening at the middle layers is most effective". But the wedding sweep peaks at layer 6, and the running example uses layer 16.

**The random control is in one appendix, on one vector.** Appendix G compares one random vector with the Anger−Calm vector, over "dozens of prompts". The toxicity and sentiment tables have no random-vector row.

**The interpretability claims are wider than the tests.** The authors say that steering "establish[es] causality, at least in the limited set of contexts examined". The text then claims "stronger, experimental evidence of feature linearity". Appendix H concludes "axis-alignment" from a non-monotonic curve on one prompt. The authors call one part of that argument "complicated and undemonstrated".

**Credit.** The authors chose the sampling settings in advance, and a blinded outside researcher checked them. The text gives the seeds, the failed vectors and the failed Llama-1 replications.

## Use in this repo

**What the repo does now.** Each audit of a neuron pole or dictionary feature records steering samples (`openinterpretability/harness/steering.py`). There are five fixed neutral prompts, greedy decoding and 24 new tokens. The pole is ablated, or clamped at ×1, ×2 and ×4 its signed `max_activation`, at every position. `docs/PROTOCOL.md` §4 cites this paper for the rule "Coherent drift at ×1–×2 says what the unit makes the model write". The samples are reported, not gated.

**The repo's intervention is not ActAdd.** ActAdd adds a difference of two residual streams, from the first token, and masks the modified positions. The repo sets one neuron's activation to a fixed value at every position, generated tokens included. So the paper's numbers do not transfer to the samples. Its general result does transfer: adding a direction changes generated text in a predictable way.

**The paper's scale argues for caution about the ×4 rule.** On GPT-2-XL, the effective Anger−Calm vector is nearly ten times the residual norm, and the output stays coherent. So coherent text at a large intervention does not show that the intervention is on-distribution. The repo's multiples are in units of the pole's own activation, not of the residual norm. That is a different scale, and the repo does not measure the ratio.

**The samples have no null, and the paper shows why one helps.** Appendix G compares a steering vector with a random vector of the same norm, by next-token KL. The repo's steering samples have no random-unit or random-direction row. So "coherent drift" has nothing to compare with. A matched control on the same five prompts would give one. This is the follow-up below.

**What the paper says about a fix for an answer like #4781.** The #4781 route is VERIFIED, at 93.8% prompt coverage. It copies " days" from the first clause through the copy heads L3/H3, L4/H0 and L4/H7, and no verified unit supplies " months" (#5015). The paper tests topic, sentiment and toxicity, not one factual answer. Its nearest test is the Eiffel Tower vector, at layer 24 of GPT-2-XL. That vector moves the tower's location in the completion, but it fails to replicate on Llama-1-13B. The paper also tests side effects on ConceptNet, through a top-K probability. It does not test the neighbour prompts that one edited answer could break. For #4781 those prompts are the other five in `docs/analyses/hallucination-shortlist.jsonl`. This is evidence for the fix campaign, which the maintainer designs. It is not a design.

**Question for the maintainer.** Should the steering samples also record an additive, contrast-pair mode next to the clamp? It would match the method that the protocol cites. #3209 already asks whether steering should become a gate. This page records the question and opens no issue for it.
