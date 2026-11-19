# Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking

**Citation.** Nikhil Prakash (Northeastern University), Tamar Rott Shaham (MIT CSAIL), Tal Haklay (Technion – IIT), Yonatan Belinkov (Technion – IIT) and David Bau (Northeastern University). "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking". [arXiv:2402.14811](https://arxiv.org/abs/2402.14811), version 1 of 22 February 2024, the only version. Venue: the Twelfth International Conference on Learning Representations (ICLR 2024), as the page header and the arXiv comment state. Code, data and the fully fine-tuned model are at [finetuning.baulab.info](https://finetuning.baulab.info). Funding: an Open Philanthropy AI Alignment grant (Prakash, Haklay, Belinkov, Bau). Haklay and Belinkov also had support from the Israel Science Foundation (grant 448/20) and an Azrieli Foundation fellowship. Rott Shaham had partial support from the Zuckerman STEM Leadership Program and the Viterbi Fellowship. The Center for AI Safety gave compute.

**Status.** Planned. #5140 would use its cross-model activation patching (CMAP). That issue patches the instruct model's MLP neuron activations into Qwen2.5-0.5B on the same input, which is CMAP on neurons in place of heads.

**Read.** On 2026-11-19 we downloaded the arXiv PDF of version 1 and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own words the abstract, sections 1 to 9 and Appendices A to J, with Tables 1 and A1 to A13. We read the reference list only in part. We did not see the figures as images. The text extraction gives the axes and legends of Figures 1, 3, 4 and A1 to A9, but not their bar heights. So this page leaves out every number that is only in a figure, such as the accuracies of Figure 3 and Figure 4.

## Summary

**Claim.** Fine-tuning Llama-7B on arithmetic improves its entity tracking, but it does not change the mechanism. The circuit of attention heads that does entity tracking in the base model also does it in three fine-tuned models. Each part of that circuit has the same function in all four models. The gain comes mostly from better positional information and a better read of the right object.

**Method.**

- **Models.** Llama-7B and three fine-tunes of it. Vicuna-7B learns from ShareGPT conversations. Goat-7B learns synthetic arithmetic through LoRA. FLoat-7B learns the same arithmetic data without LoRA. The paper does not say whether Vicuna-7B used LoRA or full tuning.
- **Task.** Adapted from Kim and Schuster (2023). Each prompt names 7 boxes, each with one single-token object, for example "The apple is in box F, … Box F contains the". The model must give the object. Chance is 0.14. The authors put the object before the box label, so that the query does not copy the context's form.
- **Units.** A node is an attention head at one token position. The paper leaves all MLP layers in the computation and does not study them.
- **Circuit discovery.** Path patching on 300 examples. Each noise prompt has a random query, labels and objects. The score of a path is (p_patch − p_org)/p_org, and the most negative paths join the circuit. The authors take 90 heads in four groups, then prune to 72 by a greedy minimality test that drops heads below 1% contribution.
- **Faithfulness.** Accuracy over 500 examples, with every head outside the circuit mean-ablated. Faithfulness is the circuit's accuracy over the full model's. The baseline is 10 random circuits with the same number of heads at each position.
- **Function.** Desiderata-based Component Masking (DCM, from Davies et al., 2023) trains a sparse binary mask over each group's heads. It trains one mask for each of three counterfactual tasks: object, label and position. It uses 1,000 training and 500 test examples, λ = 0.01, two epochs, Adam and batch size 32. Activation patching then tests each masked subset in every model, ten times on new samples, with the standard deviation.
- **CMAP.** The same input goes through two models. The outputs of chosen heads in the fine-tuned model replace those heads' outputs in Llama-7B, and the authors measure Llama-7B's accuracy.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Full-model accuracy on entity tracking, 500 tasks (Table 1) | Llama-7B 0.66, Vicuna-7B 0.67, Goat-7B 0.82, FLoat-7B 0.82 |
| Llama-7B circuit (72 heads) in each model (Table 1) | Accuracy 0.66, 0.65, 0.73, 0.72. Faithfulness 1.00, 0.97, 0.89, 0.88 |
| Random circuits of the same size (Table 1) | Accuracy 0.00 to 0.01 in all four models |
| Llama-7B circuit groups (Table A2) | A: 40 heads, Value Fetcher. B: 7, Position Transmitter. C: 20, Position Detector. D: 5, Structure Reader |
| Heads that DCM assigns a function (section 5.2) | All of Group A encode the object's value. 71.43% of Group B and 70.0% of Group C encode its position. Group D has no clear function |
| Goat-7B and FLoat-7B circuits (Appendices E, F) | 200 heads before pruning and 175 after, in the same four groups and positions |
| Goat-7B circuit in each model (Table A5) | Faithfulness 1.17 on Llama-7B, 1.13 on Vicuna-7B, 0.99 on Goat-7B, 0.96 on FLoat-7B |
| FLoat-7B circuit in each model (Table A9) | Faithfulness 1.05, 1.04, 0.93 and 1.0 |
| Llama-7B heads also in the Goat-7B circuit (Table A6) | A 27 of 40, B 6 of 7, C 16 of 20, D 5 of 5 |
| Llama-7B heads also in the FLoat-7B circuit (Table A10) | A 27 of 40, B 7 of 7, C 15 of 20, D 5 of 5 |
| Completeness, Llama-7B circuit, random K (Table A1) | Incompleteness 0.1 ± 0.06 over 10 subsets. Group A 0.19, B 0.21, C 0.02, D 0.12 |
| Completeness, Goat-7B and FLoat-7B, Group A (Tables A4, A8) | Incompleteness 0.2 and 0.43. The other groups are 0.01 to 0.12 |
| Value Fetcher heads in fine-tuned models (section 5.3) | Goat-7B "can achieve a performance improvement of 20%" over Llama-7B |
| CMAP, fine-tuned heads into Llama-7B (section 6.2) | Value Fetcher heads give "the full fine-tuned models' performance". Position Transmitter heads give "a substantial increase" |
| Position desiderata on Position Transmitter heads (Table A11) | Accuracy 0.16 to 0.38 over 11 changes to the prompt. Extra boxes before the right segment give 0.2 |

## Where the claims go beyond the evidence

**One task, one base model, and a general title.** All results are for one box-and-object template on Llama-7B. The title and the ethics statement speak of fine-tuning in general. The authors state this limit in section 7: whether "such mechanism invariance is typical will require experience with further tasks on more models".

**The "same circuit" is a circuit of heads only.** Every MLP layer stays in the model during each test, and the authors ablate only heads. So the paper cannot show that the MLPs compute the same thing after tuning. A gain inside the MLPs would appear as "enhanced" head outputs.

**"Approximately a superset" means three quarters.** By Tables A6 and A10, 54 of the 72 Llama-7B nodes are in each fine-tuned circuit. In Group A, 13 of 40 are not. The tables call the share of Llama-7B heads "precision". That is the recall of the base circuit, so the names are the reverse of what a reader expects.

**The larger circuits come partly from a number the authors chose.** Path patching gives no threshold, so the authors fix the starting size by hand: 90 heads for Llama-7B and 200 for each fine-tuned model. Minimality only removes heads. So the claim that fine-tuning "is inserting additional components" rests partly on the larger start.

**No intervals on the main table.** Table 1 gives accuracies over 500 examples with no interval or seed. Circuit discovery and the DCM masks each run once. Only Figure 3, the position desiderata and the random completeness tests report a spread.

**CMAP has no control.** The paper patches fine-tuned head outputs into Llama-7B and finds a gain. It does not patch random heads across the models, or the same heads from Llama-7B into itself. It also does not patch in the other direction. So the gain is not compared with a null.

**The mean in mean ablation is not defined.** The text says that heads outside the circuit are mean-ablated. It does not say which inputs give the mean. Faithfulness above 1 (1.17 in Table A5) shows that the choice matters.

**Small errors.** The abstract says "Patch Patching" for path patching. Appendix E4 cites Table A10 for the Goat-7B overlap, which is Table A6. Appendix F4, on FLoat-7B, says the heads "remain highly influential in the Goat-7B circuit".

The authors release code, data and the FLoat-7B model. They test both LoRA and full tuning, and they give the full head lists for all three circuits.

## Use in this repo

**The repo's base-instruct test is the neuron form of the paper's Table 1.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) takes each VERIFIED pole of a Qwen2.5 base model, keeps its index, and tests its probes on the instruct model. On Qwen2.5-1.5B, 38 of 43 keep (88%), at a median correlation of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%), at a median of 0.974. The paper keeps the base circuit's graph and finds faithfulness of 0.88 to 0.97 in the fine-tuned models. Both rest on one fact: a fine-tune shares the architecture and the start, so unit *i* is the same position in both models. Both find that most units keep their role.

**The repo cannot see "enhanced" units, and its rows suggest none.** The keep test sets each pole's scale by the ratio of its instruct maximum to its base maximum. That removes any change in a pole's range before the test reads the probes. The rows in `base-instruct.jsonl` and `base-instruct-0.5b.jsonl` hold this `ratio` per pole. For this page we read them. The median ratio is 0.983 for the 38 poles that keep on 1.5B, and 0.956 for the 252 on 0.5B. Only 15 of 38 and 86 of 252 are above 1. So instruction tuning does not make the kept poles larger, unlike the paper's Value Fetcher heads. Range is not task accuracy, and chat tuning is not arithmetic tuning, so this is no test of the paper's claim.

**The paper speaks to heads, and the repo also explains heads.** The registry claims attention heads (for example pythia-70m L1/H3), scans them in `openinterpretability/harness/heads.py`, and plans them in `openinterpretability/planner_heads.py`. A head's strength there is |W_O(z − z̄)|. The paper's units are a head at one token position, and its function labels come from counterfactual tasks, not from top activations. The paper says nothing about MLP neurons, which are most of the registry.

**The answer routes already have the paper's random baseline.** The VERIFIED route of #4781 (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) has 32 units, neurons and heads. Its sufficiency is 0.62 and its patching score is 0.75, against 8 null sets of like units with a null factor of 3. That is the paper's comparison of a circuit with 10 random circuits of the same size. [`docs/analyses/copy-route-separation.md`](../analyses/copy-route-separation.md) reads the same 32 units on 956 copies. None of them separates right copies from wrong ones beyond its null. The paper's transfer test has no counterpart in the repo yet, because every answer route is on pythia-70m, which has no fine-tuned partner here.

**A route found in a base model could be tested on its instruct model, as Table 1 does.** Take a VERIFIED answer route of Qwen2.5-0.5B and keep its units. Then run the route's own gates on Qwen2.5-0.5B-Instruct with no new search. That asks the paper's question with the repo's measures. This page proposes it as a follow-up issue. It gathers evidence only and fixes nothing.

**CMAP is the method of #5140, and the paper's gaps are the controls #5140 needs.** #5140 patches instruct activations of VERIFIED poles into the base model and reads the KL from the instruct model. The paper adds two lessons. First, activations of two models can be compatible. The paper finds that they are, though "they could have been using completely different sub-spaces and/or norms". Second, the paper reports no null. So #5140 should patch random poles, as it plans. It should also patch the instruct activations of the same poles from a different prompt. That tells a general shift of the tuned activations apart from information about the prompt. The repo's `ActivationPatcher` (`openinterpretability/harness/patcher.py`) holds one runner and patches between two inputs of that runner. Cross-model patching needs a source runner beside the target runner. These points belong in a comment on #5140, not in a new issue.

**For the fix work, the paper is one data point.** [`docs/literature/fixing-models.md`](fixing-models.md) collects evidence on fixing a model's behaviour, such as the #4781 " days" copy. In this paper, a fine-tune that raised accuracy by 16 points kept the base circuit and improved what its parts carry. If that holds in small models, an explanation of the base route is a first guess for a tuned model's route. The paper gives no evidence for a fine-tune that must remove a behaviour, which is what a wrong copy needs. The design of any fix is the maintainer's.

**Neighbouring reviews.** [`crosscoder-chat-tuning.md`](crosscoder-chat-tuning.md) finds that most chat-only crosscoder latents also exist in the base model. That is the same "keep, not replace" pattern, in the residual stream. [`implicit-inference-forgetting.md`](implicit-inference-forgetting.md) argues that fine-tuning changes which task the model infers more than what it can do. The present paper is the circuit-level evidence closest to that view, on one task.
