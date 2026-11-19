# Fast Model Editing at Scale (MEND)

**Citation.** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn and Christopher D. Manning (Stanford University). "Fast Model Editing at Scale". arXiv 2110.11309v2, 13 June 2022 (v1 21 October 2021). [Paper](https://arxiv.org/abs/2110.11309). ICLR 2022: the PDF header says "Published as a conference paper at ICLR 2022", and the arXiv comment agrees. Code and data: [project site](https://sites.google.com/view/mend-editing), which we did not open. Funding: a Knight-Hennessy graduate fellowship for Mitchell. Finn and Manning are CIFAR fellows. The paper gives no tool disclosure.

**Status.** Not applicable for now. No code, skill or open issue in this repo edits weights. The fix campaign could use the paper's tests of reliability, generality and locality, but the maintainer designs that campaign, and it has no issue yet.

**Read.** On 2026-11-18. We read the arXiv abstract page through a web-fetch model. We read the v2 PDF ourselves, as text from `pdftotext`. That text covers the whole paper: Sections 1–6, the acknowledgements, the ethics and reproducibility statements, and Appendices A–G. The figures came through as captions only, so we did not see Figures 1–4. We did not read the reference list in detail, and we did not open the code.

## Summary

**Claim.** Fine-tuning on one edited example overfits, and earlier learned editors do not scale. MEND trains small networks that map the fine-tuning gradient of one input and label to a weight edit. The edit changes the output on the input and on its rephrasings, and leaves unrelated inputs about the same. MEND trains on one GPU in under a day, even for models with more than 10 billion parameters.

**Method.**

- **The three properties.** An edit is *reliable* if the edited model gives the label on the edit input. It is *general* if the model also gives the label on inputs from the edit's "equivalence neighborhood" (rephrasings). It is *local* if the KL divergence between the old and new model on unrelated inputs is small.
- **Gradient decomposition.** For one example, the gradient of an MLP weight matrix has rank 1. It is the outer product of the output gradient δ and the layer input u. MEND maps (u, δ) to a "pseudoactivation" and a "pseudodelta". Their outer product, summed over tokens, is the edit.
- **Editor network.** An MLP with one hidden layer, low-rank weights and residual links, set to the identity at the start. One set of editor weights serves each weight shape. A per-layer scale, offset and step size let layers differ.
- **Training.** The loss is 0.1 × L_e + L_loc. L_e is the negative log-likelihood of the label on a rephrasing. L_loc is the KL divergence on a randomly drawn unrelated input. The base model does not change during training.
- **Which weights change.** The MLP weights of the last 3 blocks for GPT and BERT models (6 matrices). The MLP weights of the last 2 encoder and decoder blocks for BART and T5 (8 matrices). All editors edit the same weights.
- **Models.** GPT-Neo 2.7B, GPT-J 6B, T5-XL 2.8B and T5-XXL 11B. At small scale: BERT-base, BART-base and distilGPT-2.
- **Data.** zsRE question answering, with back-translated rephrasings. FEVER fact checking, with random binary labels. A Wikitext set of about 68k pairs, whose labels are 10-token samples from distilGPT-2. Its "rephrasing" is the passage with up to half its prefix removed.
- **Metrics.** Edit success (ES) is the accuracy on the edit input and its neighbourhood. Drawdown (DD) is the rise in perplexity, or the fall in accuracy, on other data.
- **Baselines.** Fine-tuning (FT) up to 100 steps. FT+KL, an "oracle" that has the training set at test time. Editable Neural Networks (ENN, a re-implementation). KnowledgeEditor (KE). A caching editor in Appendix G.

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| GPT-Neo 2.7B, Wikitext (Table 3) | MEND ES 0.81, perplexity DD 0.057. FT 0.55 and 0.195. KE 0.00 |
| GPT-J 6B, Wikitext (Table 3) | MEND ES 0.88, DD 0.031. FT 0.80 and 0.125. KE 0.01 |
| T5-XXL 11B, zsRE (Table 3) | MEND ES 0.89. FT 0.87. KE 0.04. Accuracy DD below 0.001 for all |
| distilGPT-2, Wikitext (Table 4) | ENN ES 0.93, DD 0.094. MEND 0.86 and 0.225. FT 0.29 and 0.938 |
| BERT-base FEVER, BART-base zsRE (Table 4) | ENN, KE and MEND all have ES 0.95 or more |
| 25 edits at once, BART-base zsRE (Table 5) | MEND ES 0.89, ENN 0.35 |
| 125 edits at once (Table 5) | MEND ES 0.67, DD 0.012. ENN 0.11 and 0.006 |
| No identity start, distilGPT-2 (Table 6) | ES falls from 0.86 to 0.27 |
| No input normalisation, distilGPT-2 (Table 6) | ES falls from 0.86 to 0.02 |
| Edit attention, not MLP, GPT-J (Table 9) | ES 0.54, against 0.88 for MLP |
| Caching editor, Wikitext (Table 11) | ES 0.001 at the threshold that matches MEND's drawdown |

Tables 2 and 10 show failures on a T5 model. After the edit "Opel makes the Astra", the model also says Opel for the F-150 and the Golf. After "Michael Jackson recorded Thriller", it says Michael Jackson for "Back in Black". The paper calls this "overgeneralization". It also shows "undergeneralization", where a rephrasing keeps the old answer.

## Where the claims go beyond the evidence

- **One seed and no intervals.** Appendix C says "seed 0 for all experiments". No table has an interval or a test. Some gaps between editors are 0.01 or 0.02 in ES, and one seed cannot separate them.
- **Selection on the reported split.** "All reported performance numbers are on the validation set". Early stopping uses "a subset of 500 validation examples". So the stopping rule saw part of the data that the tables report.
- **Most labels are not real errors.** zsRE labels are other plausible answers from a BART model, and FEVER labels are coin flips. Wikitext labels are distilGPT-2 samples, and fewer than 1% agree with the base model's own output. Appendix C.4 says that MEND "performs reliably" on real errors, but gives no number for it.
- **The generality test is weak.** Back-translated rephrasings and prefix removal are close to the edit input. The authors say themselves that these neighbourhoods do not test what the edit implies.
- **The locality test is weak.** The locality inputs are random, so they are seldom related to the edit. The authors say that such negatives are "not challenging enough", and Table 10 shows the cost.
- **"Only approach at 10B" covers four baselines.** ENN did not run at that scale because of memory. The claim compares MEND with FT, FT+KL and KE only.
- **Numbers disagree inside the paper.** Section 5.3 says that MEND applies 25 edits "with 96% edit success". Table 5 gives 0.89 for 25 edits. Table 4 gives MEND on zsRE as ES 0.98 and DD 0.002. Table 6 gives "> 0.99" and 0.001 for the same setting. Table 11 labels the zsRE-hard drawdown as perplexity, but zsRE is an accuracy task.
- **The paper fixes the edited layers in advance.** The paper always edits the last 2 or 3 blocks. It does not test whether the right layer depends on the edit. It explains why MLPs edit better than attention only as a hypothesis.

The authors state the locality and generality limits in Section 6, and they show the failure cases in Table 10.

## Use in this repo

**The three properties map onto tests that the repo already runs.** For the " days" prompt of #4781, reliability is the answer on "A week has seven days and a year has twelve". Generality is the answer on the same failure in other words. Locality is the model's behaviour elsewhere. `docs/analyses/hallucination-shortlist.jsonl` has 35 prompts with one right answer. The script keeps six of them, and all six copy the first clause's unit.

**The shortlist has the hard negatives that the paper says it lacked.** In some shortlist prompts, copying the first unit is right. "A spider has eight legs and a dog has four" gives " legs" (0.55). "A car has four wheels and a bike has two" gives " wheels" (0.41). A change that stops the copy could also break these prompts. That is the overgeneralisation in the paper's Table 10. A random sample of unrelated text, like the paper's locality inputs, would seldom contain such a prompt.

**Coverage could measure drawdown, but only for the MLP side.** `openinterpretability/coverage.py` computes the mean next-token loss on held-out windows (`mean_loss`). The same loss before and after a change is the paper's perplexity drawdown. Coverage itself measures something else: how much of the loss the VERIFIED poles keep. A change to a weight could move the coverage of the poles that it touches.

**A MEND edit falls on the units that are active at the edit token.** This follows from the paper's Eq. 2, but the paper does not discuss neurons. For an MLP output matrix, u is the vector of neuron activations at the token. So a rank-1 edit changes the write of each neuron in proportion to its activation there. Thus such an edit at the answer position touches the poles active there. The registry can name those poles, and a VERIFIED explanation predicts where else they fire. That prediction tells where the edit could leak.

**The paper's layer choice does not match the #4781 route.** MEND edits the MLPs of the last blocks. Pythia-70m has 6 layers. The #4781 route starts at layer-0 count-word and time-unit neurons, and the copy heads L3/H3, L4/H0 and L4/H7 carry " days". Appendix E finds that MLP edits beat attention edits, but on facts, where the paper has no route. The paper gives no evidence on edits to a copy route.

**The method itself needs an edit dataset.** MEND trains its editor on about 68k edit pairs before it can make one edit. This repo has one explained failure and five related prompts. So the method does not apply as it stands. Its evaluation terms carry over without the training set.

**Questions for the maintainer.** These are not issues, because the fix campaign is the maintainer's.

1. Should the shortlist prompts that copying answers right, such as " legs" and " wheels", count as locality cases for any fix of #4781?
2. Should the held-out loss of `coverage.py` be the drawdown measure, and on which corpus?
3. Should a fix report results on more than one seed, given that this paper used seed 0 only?
