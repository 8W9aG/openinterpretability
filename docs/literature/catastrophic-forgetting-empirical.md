# An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks

**Citation.** Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville and Yoshua Bengio. "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks". arXiv:1312.6211 [stat.ML], version 3, 4 March 2015. Version 1 dates from 21 December 2013, and version 2 from 6 January 2014. [arXiv](https://arxiv.org/abs/1312.6211). Code: [github.com/goodfeli/forgetting](https://github.com/goodfeli/forgetting). Preprint, no venue given on arXiv. The code repository calls the paper an "ICLR submission", but neither source says that ICLR accepted it. The paper gives no affiliations, only e-mail addresses. Four are at the Université de Montréal (`iro.umontreal.ca`, `umontreal.ca`), and Da Xiao's is at `bupt.edu.cn`. Funding and tools: NSERC, Compute Canada and Calcul Québec gave compute, and Goodfellow had the 2013 Google Fellowship in Deep Learning. The experiments use Theano and Pylearn2. The data are MNIST and the Amazon reviews of Blitzer et al. (2007).

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it, together with [`complementary-learning-systems.md`](complementary-learning-systems.md). The paper measures forgetting in small classifiers, not units. So this page reports evidence and records questions only, because the campaign to fix models is the maintainer's.

**Read.** On 2026-11-19 we downloaded version 3 of the PDF from arXiv with `curl` and read its text, extracted with `pdftotext`. We read all of it in the source's own text: the abstract, sections 1 to 5, the acknowledgments and the references. A model (claude-opus-5-5) read it, not a person. We also rendered pages 5 to 7 as images and looked at the six figures. The text gives almost no numbers, and the results are curves in Figures 1, 3 and 5 and bars in Figures 2, 4 and 6. This page takes no numbers from the figures. It takes only their direction, their axis titles and their legends. We did not read the code.

## Summary

**Claim.** When a network trains on one task and then on a second, dropout is the best training algorithm in all three task pairs. It is best on the new task, best on the old task and best along the trade-off between them. The best activation function changes with the task pair, so the authors advise cross-validating it.

**Method.**

- **Networks.** Each network has two hidden layers and a softmax output layer. Two algorithms (stochastic gradient descent and dropout) and four activation functions (logistic sigmoid, rectified linear, hard local winner-take-all and maxout) make eight conditions.
- **Hyperparameters.** Random search draws 25 hyperparameter sets for each condition and task pair. The search covers the max-norm constraint, the initialisation, the initial biases, the learning-rate and momentum schedules and the layer sizes. The authors set the dropout probabilities to 0.5 for hidden units and 0.2 for inputs. The initial-bias ranges differ by activation function, and the paper gives the reasons.
- **Training.** A network trains on the old task until its validation error has not improved for 100 epochs. The run then restores the best parameters. It then trains on the new task until the error on the union of both validation sets has not improved for 100 epochs.
- **Measure.** After each pass through the new task's training set, the authors record the test error on both tasks. A *possibilities frontier* traces the lower-left edge of all these points from all 25 runs. Its axes are the old task's test error and the new task's test error, on log scales.
- **Task pairs.** *Input reformatting*: MNIST, then MNIST with a different permutation of the pixels. *Similar tasks*: sentiment of Amazon reviews in two product categories. The figure title names them as Kitchen, then DVD. *Dissimilar tasks*: two MNIST classes, then Amazon sentiment. The figure title names the classes as 2 and 9, and the new task as DVD reviews. PCA reduces the Amazon data from 5,000 features to MNIST's 784.
- **Optimal size.** Figures 2, 4 and 6 compare the size, in parameters, of the best network with and without dropout.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| A two-layer rectifier network on MNIST, 25 random-search runs for each algorithm | The best network with dropout had "56.48% more parameters" than the best without it |
| Input reformatting | Dropout "improved the two-task validation set performance for all models" |
| Input reformatting, the best network's first-layer weights | Their "apparent semantics do not noticeably change" after the new task. The authors conclude that the higher layers adapted to the new input format |
| Similar tasks | Dropout improved the two-task validation performance "for all models". Hard LWTA with SGD is "the worst of all eight methods" |
| Dissimilar tasks | Dropout improved performance "for all models", but it "reduced the size of the optimal model for most of the activation functions" |
| Input reformatting and similar tasks, optimal size | Dropout "never decreased the size of the optimal model" |
| Activation functions | Their ranking "is very problem dependent". Logistic sigmoid is "the worst under some conditions but the best under other conditions" |
| Maxout with dropout | "the only method that appears on the lower-left frontier" for all three task pairs |
| Hard LWTA | Best on the new task in the dissimilar pair, but "never the leftmost point" in any pair |

The text gives no error rates for any condition. The frontiers in Figures 1, 3 and 5 hold all the results, so this page gives only their direction.

## Where the claims go beyond the evidence

- **"Always best" rests on small data sets.** The abstract says that "it is always best to train using the dropout algorithm". The discussion adds "at least on the relatively small datasets we used". The networks have two hidden layers, and the tasks are MNIST and one sentiment data set.
- **One seed and no spread.** Each hyperparameter set trains once. The paper reports no seeds, no repeated runs, no intervals and no tests, so it can't show which frontier gaps exceed the noise.
- **The frontier selects on test error.** Each frontier is the lower envelope of test-error points from 25 runs and every epoch. That chooses the best run and epoch by the same test errors that it reports. A method with more spread across runs can gain from the envelope alone.
- **The capacity account fails on one pair, as the authors state.** They suggest that dropout resists forgetting because it allows larger networks. On the dissimilar pair dropout shrank the optimal network, and the authors say that they "do not yet have an explanation".
- **The results are in figures only.** The text states which method wins but gives no error rates. A reader must read the log-scale curves, and the authors note that the log scale "sometimes makes the convex regions apear non-convex".
- **Labels within the paper disagree.** Section 4.3 introduces the dissimilar pair as "semantically similar". Section 4.2 sends the reader to Figure 6, but the similar-task sizes are in Figure 4. Figure 2's chart title says "similar tasks", but its caption says input reformatting. The text doesn't name the two Amazon categories or the two MNIST classes, and only the figure titles do.
- **The first-layer result is an impression.** The claim that the first-layer weights kept their meaning comes from looking at the weights of one network. The paper gives no measure of the change.
- **Credit.** The authors chose hyperparameters by random search to keep the comparison fair, and they say why. They give the same search to SGD and dropout, and they release the code.

## Use in this repo

- **The registry's nearest measure finds little change after a fine-tune.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED hypotheses again after instruction tuning. On Qwen2.5-1.5B, 38 of 43 keep (88%, 95% CI 76–95%), and activations correlate at a median r of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%). That is a measure of units, where this paper measures task error. The two measures answer different questions, and nothing here links them.
- **Where a fine-tune acts differs between the paper and our data.** The paper reports that first-layer weights kept their meaning while higher layers adapted. In `base-instruct.md`, Qwen2.5-0.5B keeps 104 of 112 poles in its first third, 63 of 92 in the middle and 85 of 107 in the last third. So the early layers change least there too, but the middle, not the last third, changes most. In Pythia-70m's pretraining, layer 0's poles also settle first ([`papers/neuron-development/`](../../papers/neuron-development/README.md), Table 2: median correlation 0.90 at step 16,000, against 0.46 to 0.72 in the other layers). Pretraining is not a second task, so this agreement is weak.
- **The dropout result doesn't transfer to the registry's models.** Pythia has no dropout ([`papers/why-neurons-rest/`](../../papers/why-neurons-rest/README.md)). Any fine-tune of it that the campaign runs would choose its own regularisation. Whether dropout or a larger network belongs in a fix of the #4781 error is the maintainer's question.
- **The frontier is a way to report a fix's side effects.** A fix trades the new behaviour against the old, and the paper plots that trade-off as one curve. [`complementary-learning-systems.md`](complementary-learning-systems.md) names the old behaviour to keep: shortlist prompts where copying is right, such as "A dog has four" → " legs". Whether a fix should report such a frontier is the maintainer's question. We open no issue for it.
- **Its warnings apply to our own comparisons.** A frontier built from the best of 25 runs and every epoch selects on its own test data. The registry's comparisons of methods should keep that selection apart from the data they report on.
- **No single-unit content.** The paper studies the error of whole networks. Apart from one look at first-layer weights, it says nothing about units, explanations, probes or random baselines. Nothing in it bears on a pole's explanation or on our verdicts.
- **No follow-up issue.** The questions that the paper raises for this repo are the fix campaign's, which is the maintainer's. `base-instruct.md` already measures the stability of units under a fine-tune.
