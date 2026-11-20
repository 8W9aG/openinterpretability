# Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics

**Citation.** Yaniv Nikankin, Anja Reusch, Aaron Mueller and Yonatan Belinkov, of the Technion. Mueller is also at Northeastern University. "Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics". [arXiv:2410.21272](https://arxiv.org/abs/2410.21272), version 2 of 20 May 2025. Version 1 is of 28 October 2024. Preprint, no venue given. The code is at [github.com/technion-cs-nlp/llm-arithmetic-heuristics](https://github.com/technion-cs-nlp/llm-arithmetic-heuristics). Funding: the Israel Science Foundation (448/20), the Azrieli Foundation, an Open Philanthropy AI Alignment grant, the ERC (Control-LM 101165402) and fellowships for two authors. The text gives no disclosure of tool use.

**Mismatch.** The text gives an average faithfulness of 0.96, and Table 1's values average to 0.95. Table 3's "Average" column isn't the mean of its operator columns for any model, and no weighting is described. Section 2.1 allows operands up to 1,000, and the experiments use 0 to 300.

**Status.** Planned. #5220 names it as background for phase 5(b), a compiled addition module. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in three fetches: Sections 1 to 8, the references and Appendices A to J. We didn't see any figure as an image. So the neuron-level faithfulness curves and the ablation results of Figures 7 and 8 appear here only as the text describes them. We didn't read version 1 or the code.

## Summary

**Claim.** Language models don't do arithmetic with an algorithm or by memory. A sparse set of MLP neurons each fires on a simple pattern of the operands, such as a range or a remainder. Each one nudges the logits of matching answers. Together these heuristics produce the answer.

**Method.**

- **Models.** Llama3-8B in the main text. Llama3-70B, Pythia-6.9B and GPT-J in Appendix I.
- **Prompts.** `op1 op op2 =` with no examples, for +, −, × and integer ÷, with operands from 0 to 300. Answers must be one token. 100 prompts for each operator find the circuit, and 100 more evaluate it.
- **The circuit.** Activation patching from counterfactual prompts. Faithfulness compares the circuit's normalised answer logit with the full model's, with the rest mean-ablated.
- **Heuristic neurons.** Each neuron's activation is mapped over the grid of operand pairs. A neuron implements a heuristic when its top prompts overlap the heuristic's prompts by at least 0.6. The heuristic types are ranges, remainders, digit patterns, equal operands and, for division, sets of results.
- **Training.** Pythia-6.9B checkpoints from step 23K to 143K.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Llama3-8B circuit faithfulness (Table 1) | + 0.97 with 6 heads, − 0.98 with 6, × 0.90 with 20, ÷ 0.96 with 6. All MLPs are in the circuit |
| Neurons that carry the circuit (Section 3) | the top 200 neurons of each layer from 16 on, about 1.5% of MLP neurons |
| Neurons classified as a heuristic (Section 4.1) | 91% of those top neurons |
| Knockout of a prompt's heuristic neurons (Section 4.2) | accuracy falls from about 95% to about 66%, "not ... a complete accuracy drop" |
| Wrong answers (Section 4.3) | wrong prompts have at least as many heuristic neurons as right ones. The authors put errors down to weak promotion of the right logit |
| Over training, Pythia-6.9B (Section 5) | the final heuristics explain 79% of the heuristic contribution at each earlier checkpoint. Ablating 25 heuristic neurons per layer gives near-zero accuracy at every checkpoint |
| Accuracy (Table 3, "Average" column) | Llama3-8B 0.95, Llama3-70B 0.88, Pythia-6.9B 0.43, GPT-J 0.37 |

## Where the claims go beyond the evidence

- **The 91% has no null.** A neuron with a broad activation pattern could overlap some heuristic among many types by chance. The paper doesn't classify random neurons, or neurons of an untrained model, against the same list. The authors say they estimated false labels "manually," with no number.
- **The heuristic list was chosen after looking.** The types and the 0.6 threshold come from the authors. The threshold was set for a "Pareto-optimal balance."
- **No seeds, intervals or tests.** The text says "significantly" in two places with no test. The failure analysis uses 50 right and 50 wrong prompts.
- **One model for most results.** The 91% and the neuron-level circuit are for Llama3-8B. The training analysis is Pythia-6.9B only. The other models are "similar" in figures.
- **Knockouts leave much intact.** Removing a prompt's heuristic neurons leaves about 66% accuracy. The authors credit low recall and redundancy. That also means the bag is not a complete account.

The authors state limits themselves. The heuristic types are defined by hand. They studied only tokenizers that keep numbers as multi-digit tokens. Some heuristics have low recall.

## Use in this repo

- **The issue's question: one circuit or many units?** Many units. In Llama3-8B the answer comes from about 1.5% of MLP neurons in the later layers, most of them simple heuristics. Removing a prompt's heuristic neurons costs about 29 points of accuracy, not all of it. A fix that targets one circuit would leave the rest of the bag acting.
- **A heuristic neuron is a registry-style explanation.** "Fires when the result is between 150 and 180" is a rule about inputs that a registry probe set can test. These neurons are the kind of unit the registry already explains. The paper's 0.6 overlap rule is weaker than the registry's verdict, which needs probes and causal checks.
- **The tokenizer splits the registry's models.** We checked: Pythia-70m and GPT-2 keep `226` as one token, as the paper's models do. Qwen2.5-0.5B splits it into `2`, `2`, `6`. The paper names single-digit tokenizers as a limit, so its picture may not hold for Qwen.
- **Size bears on the "absent" failure.** #5220 calls Pythia-70m's "Two plus two equals" → " the" an *absent* route, with " four" at rank 27. The paper's smallest model, Pythia-6.9B, reaches only 0.43. Our reading, not the paper's: a 70M model may have no bag of heuristics for arithmetic at all. That fits #5220's choice to add capacity for this failure, not to fix a route.
- **For phase 5(b), the bag is what a compiled core replaces.** A compiled addition core is one algorithm. The read adapter would sit where the paper finds the operands copied by a few heads to the last position. #5220 predicts that the core generalises and the interfaces fail. This paper gives a reason: the model's own number features are heuristic, so a read adapter must decode the operands from them.
- **Questions for the maintainer, recorded on this page only.**
  - Should phase 5(b) run on Qwen, whose digits are single tokens, or on a model whose numbers are whole tokens, as in the paper?
  - Should the registry test the paper's heuristic types as explanations of arithmetic neurons, with a random-neuron null?
- **Related pages.** [`interchange-intervention-training.md`](interchange-intervention-training.md) is a way to train a read adapter to a stated algorithm. [`tracr.md`](tracr.md) and [`alta.md`](alta.md) compile the core. [`ffn-key-value-memories.md`](ffn-key-value-memories.md) is the key and value view of a neuron that the paper's logit-lens test uses.
