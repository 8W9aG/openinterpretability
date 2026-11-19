# Learning Transformer Programs

**Citation.** Dan Friedman, Alexander Wettig and Danqi Chen, Department of Computer Science and Princeton Language and Intelligence, Princeton University. "Learning Transformer Programs". arXiv 2306.01128 [cs.LG]. Versions: v1 1 June 2023, v2 31 October 2023. We read v2. [Paper](https://arxiv.org/abs/2306.01128). [Code and example programs](https://github.com/princeton-nlp/TransformerPrograms). The v2 PDF says "37th Conference on Neural Information Processing Systems (NeurIPS 2023)". Issue #4844 says the paper was an oral, and the PDF does not say this, so we did not check it. The data are public sets: CoNLL-2003 through Hugging Face Datasets, TREC, MR, Subj and AG News. Funding: the National Science Foundation (IIS-2211779) and a Sloan Research Fellowship.

**Status.** Not applicable. The paper trains a changed architecture whose units are discrete modules, not MLP neurons, and the harness can't load it. Its evidence goes to open issues as comments: #5149 and #3584 on positive controls, #5144 on token-set rules, and #3446 on reading weights.

**Read.** We read the arXiv v2 PDF on 2026-11-19, as text that `pdftotext` took from it. We also read the arXiv abstract page for the version history. We have the source's own text for the abstract, §1 to §6, the references and Appendices A to C. We did not see the figures as images. The bars of Figure 10 and the plots in Figures 6 to 8 and 13 came through only as axis labels. So the only numbers from them here are numbers that the text states. The code listings in Figures 4, 6, 7, 8, 9 and 13 came through as text. We did not read the code or the released programs. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** The authors constrain a transformer and train it by gradient descent. Then a fixed procedure turns its weights into an equivalent Python program in the style of RASP. They call these models Transformer Programs. They say the programs perform "on par with standard Transformers of comparable size" and are easy to interpret.

**Method.**

- **Disentangled residual stream (§3.1).** The residual stream holds a fixed set of named categorical variables, each one-hot in its own orthogonal subspace. Each module writes one new variable by concatenation, so the width is (2 + L × H) × k. Here k is the variable cardinality.
- **Reading.** Each projection is a one-hot gate over variables, so a head reads exactly one key, one query and one value variable.
- **Categorical attention (§3.2, A.1).** A head has a binary predicate matrix whose rows each have one non-zero entry, so each query value matches one key value. Attention is hard. With no match, the query attends to the first token. With several, it attends to the closest match.
- **Training.** The gates and predicate rows are categorical distributions, trained through the Gumbel-Softmax with a temperature that falls from 3.0 to 0.01. After training, the maximum-likelihood weights give the discrete model, and a fixed mapping writes it as Python.
- **Extensions (§3.3, A.2).** Factored categorical word embeddings for large vocabularies. Numerical attention, which sums a value and so counts, like RASP's `selector_width`. Feed-forward modules that read ℓ = 2 variables through a one-hidden-layer MLP and output one categorical variable. The authors decompile each one to a lookup table by listing all k² inputs.
- **Tasks.** An in-context learning task with two attention-only layers. The seven RASP tasks of Weiss et al. (2021), at length 8 (16 for Dyck). CoNLL-2003 named entity recognition. Four text classification sets.
- **Baselines.** Standard transformers of similar size with a grid search, a unigram tagger for NER, and a multinomial Naive Bayes bag of words for classification.
- **Seeds and selection.** Five seeds for the in-context task and the RASP tasks, three for NER and classification. The paper reports the best seed.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| In-context learning, 2 layers, 1 head each (§4.1) | "Perfect accuracy on a held-out test set". Head 1 copies the previous token. Head 2 is an induction head that matches a letter to the number after it |
| RASP tasks, best of grid (Table 1) | Reverse 99.79, Histogram 100.0, Double histogram 98.40, Sort 99.83, Most-Freq 75.69, Dyck-1 99.30, Dyck-2 99.09 (token-level accuracy) |
| Model size on RASP (§4.2) | Weiss et al. solve Sort and Reverse with two layers and one head. The best learned models use three layers and eight heads |
| Sort program (§4.2) | A two-layer, four-head model above 99% compiles to about 300 lines without the classifier. One head looks for a 1 at early positions and a 4 at later ones |
| Longer inputs and larger vocabulary (Figure 10 caption) | Both model types degrade. Transformer Programs degrade more, with "over a 50% accuracy gap" to the standard transformer in the hardest setting |
| Only categorical variables (Table 3) | Reverse 99.75, Hist 83.20, 2-Hist 55.89, Sort 99.99, Most Freq 64.13, Dyck-1 99.43, Dyck-2 99.17 |
| Program length, full and pruned (Table 4) | From 324 and 160 lines (Hist) to 9,975 and 892 lines (Dyck-1) |
| NER, CoNLL-2003 (Table 2) | F1: unigram 66.2, standard transformer 70.5, Transformer Program 76.1 |
| Text classification (Table 6) | TREC 85.6 against 88.6 for the standard transformer. MR 77.9 against 77.2, Subj 93.0 against 92.3, AG 90.8 against 91.7 |
| Reverse with hand-set components, 3 seeds (Table 5) | All random: 23.2, 23.6, 24.1. Attention reads, predicates and MLP reads set: 99.9, 99.9, 80.8. Only the MLP read random: 95.1, 94.1, 95.3. Layer-1 predicate random: 99.1, 83.9, 78.2. Each other attention component random: 13.7 to 44.1 |

The authors state the limits themselves (§6, C.3):

- Programs "can struggle to learn on longer inputs, and to learn parsimonious solutions". They think the main problem is optimisation, because a program with perfect accuracy exists for many tasks.
- Hard attention is less expressive than standard attention, and the paper uses a small set of modules.
- "The learned programs can still be complicated and non-intuitive." For complex tasks a program is a set of feature functions, not one algorithm.

## Where the claims go beyond the evidence

- **"Easy to interpret" is not measured.** The paper has no user study, no task for readers and no metric of interpretability. It shows hand-picked parts of programs. The pruned programs are 160 to 892 lines long (Table 4). The authors grant this in §6.
- **"On par" holds only at small scale.** Table 1 gives no standard-transformer numbers. Figure 10 shows a gap of over 50% in the hardest setting, and the caption says Transformer Programs "deteriorate more dramatically".
- **The NER baseline may be weak.** The standard transformer's F1 of 70.5 is only 4.3 points above a unigram tagger. Its recall, 62.7, is lower than the program's 71.8. The paper does not say why a tuned transformer falls so far below the program.
- **One model was chosen on test loss.** Appendix B.1 trains five in-context models and picks "the one with the lowest test loss". It then reports perfect test accuracy. The RASP and NER runs choose on validation data, as they should.
- **No intervals, and best seeds only.** Tables 1, 2, 3 and 6 report one best run. Only Table 5 gives all three seeds, and they spread from 80.8 to 99.9 in one row.
- **Table 5 does not fully support its own text.** Appendix C.3 says the model fails when even one attention component starts at random. But with the layer-1 predicate random, one seed reaches 99.1. The other two reach 83.9 and 78.2, which is partial, not failure.
- **A small inconsistency.** §4.3 keeps sentences of at most 32 words, and Appendix B.3 keeps at most 30 tokens. It does not change a result.
- **Credit.** The authors report the scaling failure, the long programs and the drop on longer inputs themselves. Table 5 is a clear test of where optimisation fails. The code and the example programs are public.

## Use in this repo

- **No neuron here has a role.** The paper's units are modules: a head with a predicate, or an MLP module that maps two variables to one through argmax. The hidden units inside each MLP module are not decompiled. Only the module's lookup table is. So, as with Tracr (`tracr.md`), the paper sets no single neuron's role. The repo's neuron controls write or route one neuron instead: the 12 plants of `docs/analyses/ground-truth-calibration.md`, and #5066's gradient routing.
- **The model does not load in the harness.** It is PyTorch, but it has a concatenated residual stream, gates over named variables, hard attention and argmax MLPs. `NeuronHookRunner` (`openinterpretability/harness/hooks.py`) needs a Hugging Face layout. #3584 hit the same problem with Tracr and InterpBench.
- **Evidence for #5149: training does not find the human program.** #5149 trains a GPT-2-shaped model with RASP attention supervision, as a positive control for head audits. This paper trains without supervision, and the learned Reverse and Sort use three layers and eight heads where Weiss et al. need two and one. In Table 5, Reverse learns the general solution only when the authors set its attention reads and predicates by hand. So a trained head's ground truth must come from supervision or from the decompiled weights, never from the task alone. That supports #5149's choice of supervision. It goes on #5149 as a comment.
- **A decompiled program is ground truth for the training range only.** Programs degrade on longer inputs (§4.2, Figure 10). So a program read from weights can describe the model exactly on its inputs and still not hold beyond them. The syntax paper cites a later work by the same first author on this risk: Friedman et al. (2024), *Interpretability illusions in the generalization of simplified models* (`papers/syntax-positive-controls/README.md`). We did not read it.
- **Reading weights is exact here only because training constrained them (#3446).** Decompiling needs no forward pass. It works because each variable is discrete and named, and each MLP module has k² inputs to list. A GPT-2 neuron reads a continuous 768-dimensional input with no named variables. On the registry's poles, ROTATE's kurtosis read from weights does not predict VERIFIED (`docs/analyses/weight-kurtosis.md`). On 518 test poles, the AUC is 0.51 (0.45–0.57) for the output side and 0.54 (0.48–0.60) for the input side. The paper gives no method for weights that training did not constrain.
- **The NER program splits features as #5144 does.** In Figure 7, the top features for B-LOC against B-ORG are word-embedding variables and attention features. The embedding variables are token sets: Var3.V03 holds "at", "In" and "Saturday". The attention head `attn_0_3` copies that variable from the previous token, which is a context rule. #5144 counts which VERIFIED explanations are token sets and which need context. This paper gives a learned example of both kinds, and only the token-set kind can be read from one unit's input weights. It goes on #5144 as a comment.
- **Evidence for #4782: the same task gives different programs.** Table 5's fully set row reaches 99.9, 99.9 and 80.8 over three seeds. The learned Reverse and Sort programs differ from the hand-written ones. So a function does not fix its mechanism, even in a model built to be read. This fits `papers/neuron-development/`, where Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. The paper does not compare programs across seeds, so it shows that the mechanism can vary, not how much.
- **Weight-space programming (#3381, #4471).** Tracr writes weights from a program, #4471 has an agent write them, and this paper reads a program out of trained weights. Table 5 also shows a mixed case. With the attention and the MLP reads set by hand and the MLP internals trained, Reverse reaches 99.9 on two of three seeds. That is evidence about writing part of a model and training the rest. It is not a fix method, and the fix campaign stays the maintainer's.
- **No single-neuron content.** The paper has no neuron explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **Proposed follow-up: none.** #5149 and #3584 already hold the positive-control ideas, and a learned program would need a new runner for a model with no neurons to audit. We propose comments on #5149, #5144 and #3446 with this page's evidence.
