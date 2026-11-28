# Enhancing Automated Interpretability with Output-Centric Feature Descriptions

**Citation.** Yoav Gur-Arieh and Roy Mayan (equal contribution), Chen Agassy, Atticus Geiger and Mor Geva. The affiliations are the Pr(Ai)²R Group and the Blavatnik School of Computer Science and AI, Tel Aviv University. The text that we read doesn't map each author to an affiliation. "Enhancing Automated Interpretability with Output-Centric Feature Descriptions". [arXiv:2501.08319](https://arxiv.org/abs/2501.08319) [cs.CL]. Version 1 is of 14 January 2025, and version 2, the one we read, of 29 May 2025. The arXiv comment says "Accepted to ACL 2025 Main Conference". The licence is CC BY 4.0. Code and the generated descriptions are at [yoavgur/Feature-Descriptions](https://github.com/yoavgur/Feature-Descriptions). The funders are the Gemma 2 Academic Research Program at Google, the Edmond J. Safra Center for Bioinformatics at Tel Aviv University, Open Philanthropy and the Israel Science Foundation (grant 1083/24). The authors say that ChatGPT helped write some helper functions.

**Status.** Planned. #3211 asks each explanation to say what its pole does to the output and to score that side by steering, and it cites this paper. The proposer's prompt already has this paper's VocabProj evidence, through the Jacobian and logit lenses. It doesn't have TokenChange evidence, and #7334 would add it.

**Read.** On 2026-11-28 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in three fetches by offset. That covers Sections 1 to 8, the limitations, the acknowledgements, the references and Appendices A to H. Quotes were capped at 125 characters, so the rest came back paraphrased. The figures are images, and we saw only their captions. So the prompts in Figures 4, 7, 8 and 11–13, and the human scores in Figure 5, are unknown to us. We did not read the PDF.

## Summary

**Claim.** Descriptions written from a feature's top-activating inputs (MaxAct) say what turns a feature on, but often miss what the feature does to the output. Two cheap describers that read the output, VocabProj and TokenChange, describe the output effect better. An ensemble of input and output describers does best on both kinds of score.

**Method.**

- **Models and features.** Gemma-2 2B (Gemma Scope SAEs), Llama-3.1 8B (Llama Scope SAEs), GPT-2 small (OpenAI SAEs), and MLP features of Llama-3.1 8B Instruct described by Transluce. There are 12,160 features in all, 40 a layer for each SAE and 80 a layer for the Llama Instruct MLP. The paper never runs a separate test on single neurons. The Llama Instruct MLP set is the only set not called an SAE, and it is not compared with SAE features of the same model.
- **The describers.** GPT-4o mini writes each description.
  - MaxAct uses top-activating examples, and reuses Neuronpedia's descriptions. MaxAct++ is Transluce's pipeline, for the Llama Instruct MLP only.
  - VocabProj projects the feature's decoder vector through the final LayerNorm and the unembedding, and gives the describer the top and bottom 50 tokens.
  - TokenChange clamps the feature on 32 random Pile prompts of 32 tokens. It gives the describer the 20 tokens whose mean logits rise most and the 20 that fall most.
  - Ensembles join the raw evidence (Ensemble Raw) or the finished descriptions (Ensemble Concat).
- **The input score.** Gemini 1.5 Pro writes 5 activating and 5 neutral sentences from the description. The description is faithful if the feature's mean peak activation is higher on the activating set.
- **The output score.** The feature is clamped on three short prompts, at values set to give a next-token KL of 0.25 and 0.5, both positive and negative. That gives 12 texts. GPT-4o mini sees the description and three sets of texts, one from this feature and two from random features, and must pick the right set. Chance is 1/3.
- **Human checks.** Three graduate students on 100 features gave the judge a winning rate of 1 ($p$ = 0.03) in an alternative-annotator test. Six students scored 100 Gemma features, but those results are only in a figure that we did not see.

**Findings, as the source reports them.**

| Result | Setting | Value (input / output score, with 95% intervals) |
| --- | --- | --- |
| MaxAct | Gemma-2 residual SAE (Table 1) | 56.6 ± 2.2 / 49.2 ± 2.2 |
| VocabProj | Same | 50.1 ± 2.2 / 56.5 ± 2.2 |
| TokenChange | Same | 44.7 ± 2.2 / 54.9 ± 2.2 |
| Ensemble Raw, all three | Same | 66.6 ± 2.1 / 64.9 ± 2.1 |
| MaxAct | Gemma-2 MLP SAE (Table 1) | 50.4 ± 2.2 / 35.1 ± 2.1 |
| VocabProj, TokenChange | Same | 20.9 / 37.2, and 22.3 / 40.3 |
| MaxAct, MaxAct++ | Llama-3.1 8B Instruct MLP (Table 1) | 85.6 / 36.9, and 89.8 / 39.0 |
| VocabProj, TokenChange | Same | 71.2 / 45.8, and 74.0 / 43.8 |
| MaxAct, VocabProj | GPT-2 small MLP SAE (Table 7) | 39.7 / 34.0, and 6.3 / 38.3 |
| Dead features revived | 1,850 Gemma-2 SAE features that Neuronpedia calls dead (Section 6.2) | 9.1% of MLP features (85) and 62% of residual features (491) |
| Cost of the describers | All MLP features of Gemma-2 2B (Table 4) | MaxAct $3.9 \times 10^{16}$ FLOPs, VocabProj $2.8 \times 10^{14}$, TokenChange $9.9 \times 10^{13}$ |

The paper says that MLP features are harder to steer. Its reason: "This might be explained by the MLP layers introducing gradual changes to the residual stream."

**Where the claims go beyond the evidence.**

- **Neurons are barely tested.** The abstract speaks of features in general, but nearly every set is SAE features. The one set of MLP units, in Llama-3.1 8B Instruct, uses Transluce's descriptions and has no SAE set of the same model beside it. On it, the output describers raise the output score by 7 to 9 points, but lose 12 to 15 points on input.
- **The output gain is small on MLP features.** On Gemma-2's MLP SAE, MaxAct scores 35.1 on output, against a chance of 33.3, and the best single output describer reaches 40.3. On GPT-2's MLP SAE, VocabProj scores 6.3 on input. The paper says that GPT-2's results are noisier.
- **The exceptions to the headline.** On Llama's residual SAE, TokenChange (72.0) and MaxAct (71.8) tie on output. On the Llama Instruct MLP, MaxAct++ beats every ensemble on input (89.8 against 86.2). The paper notes the second.
- **The dead-feature counts don't add up.** If the 1,850 features split evenly between MLP and residual, 491 residual revivals are 53%, not 62%. The paper gives no reconciliation.
- **The judges are models.** GPT-4o mini writes the descriptions and also judges the output score, and Gemini writes the input test sentences. The human check is one test on 100 features. The authors call the output score "fairly noisy" and say that the binary input threshold "may oversimplify".
- **No seeds.** The paper gives intervals for Table 1 but doesn't say how it computed them, and it reports no seeds. The authors say that the output describers depend on the prompt, can't describe positional features, and don't tell promotion from suppression.

## Use in this repo

- **VocabProj is already evidence for the proposer.** The proposer's prompt lists the tokens that each pole promotes and suppresses, through the Jacobian lens and the logit lens (`format_lens` in `openinterpretability/agents/proposer.py`). The logit lens is close to VocabProj, read from a pole's output weights. The Jacobian lens carries it through the later layers first. The `vocab-projection-lookahead` skill tells the proposer how to read the two.
- **TokenChange is not.** No evidence in the proposer's prompt comes from clamping a pole and reading the change in logits. The `neuron-steering-tests` skill clamps a pole and reads the generated text, but it is a check after the hypothesis is written, not evidence before it. The paper finds TokenChange the cheapest describer, and on most sets as good as VocabProj on output. #7334 would add its top and bottom tokens to the proposer's evidence, and compare explanations with and without them.
- **The verdict tests the input side only.** The review's test prompts ask whether a pole fires, so a VERIFIED explanation says what turns the pole on. Only a lookahead claim also predicts an output. The paper's MLP result suggests what this misses: with MaxAct, Gemma's MLP SAE features score 35.1 on output, about chance. #3211 holds the output-side score, and we commented there with this paper's protocol and numbers.
- **The output score is close to our steering checks.** Both clamp a unit and read generated text. The paper sets the clamp by a target KL, gives the judge two random features as foils, and reports chance at 1/3. Those are three choices that #3211's design can copy. The judge, GPT-4o mini, has one human check on 100 features, a smaller check than the repo runs on its own judges (`explanation-simulation-scoring`).
- **Neurons against SAE features.** The registry audits neurons. The paper's one neuron-like set scores far higher on input than any SAE set, at 85.6 against 30.3 to 56.6 for MaxAct. But its descriptions come from another pipeline, so the comparison is confounded. It is not evidence for or against the neuron basis (`neurons-as-an-interpretable-basis.md`).
