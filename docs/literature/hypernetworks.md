# HyperNetworks

**Citation.** David Ha, Andrew Dai and Quoc V. Le, Google Brain. "HyperNetworks". arXiv 1609.09106, v4 of 1 December 2016. v1 is of 27 September 2016. [Paper](https://arxiv.org/abs/1609.09106). The issue gives the venue as ICLR 2017. The v4 PDF names no venue, so we did not check the venue in the source. Ha did the work in the Google Brain Residency program. The paper links a TensorFlow reference implementation and an MNIST notebook at `blog.otoro.net/2016/09/28/hyper-networks/`. We did not open that page. The paper gives no funding statement.

**Status.** Planned. #4782 would use it in its strand on weight-space programming and fast weights, and #3381 (weight-space programming) would cite it. No code, skill or protocol in this repo depends on it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** We read the v4 PDF on 2026-11-19. We downloaded it with `curl` and converted it with `pdftotext`. `pdftotext` reported a damaged cross-reference table but gave the text of all 29 pages. We read all of it in the source's own text: Sections 1 to 5, the references and Appendices A.1 to A.6. We read the tables as extracted text. In Tables 2 to 5 the footnote marks merge with the model names. For example, "Wide Residual Network 40-12" is WRN 40-1 with footnote 2. That footnote says "our implementation". We did not see the figures as images, so we have only their captions and the text about them. We did not open the linked code.

## Summary

**Claim.** A small network, the hypernetwork, can make the weights of a larger main network. Both networks train together by backpropagation. A static hypernetwork makes the kernels of a convolutional network from a small embedding per layer. It needs far fewer parameters and loses a little accuracy. A dynamic hypernetwork changes the weights of an LSTM at each time step. That "HyperLSTM" scores near the best published results on language modelling, handwriting and translation.

**Method.**

- **Static hypernetwork (Section 3.1).** A two-layer linear network maps a learned layer embedding `z_j` to the kernel of layer `j`. The learned parameters are the hypernetwork's weights and the embeddings. Larger kernels tile basic 16-channel kernels, each from its own embedding.
- **Dynamic hypernetwork (Section 3.2).** A small LSTM, the HyperLSTM cell, reads the input and the main LSTM's hidden state at each step. It writes embeddings `z`. The full form makes each weight matrix a combination of `N_z` matrices, which costs `N_z` times the memory. The form that they use instead scales each row of a fixed matrix by one element of a vector `d(z)` (Equation 7). The authors say that this row scaling equals an element-wise product with the matrix's output. The embeddings also make the bias.
- **Data and baselines.**
  - MNIST and CIFAR-10 for the static form, against a normal ConvNet and Wide Residual Networks.
  - Character-level Penn Treebank, enwik8, the IAM online handwriting database and WMT'14 English to French for the dynamic form.
  - The baselines are LSTMs of several sizes and depths, Layer Norm LSTMs and published results. The authors implemented many of the baselines themselves.
- **Metrics.** Test error, bits per character, validation log-loss in nats, and BLEU.
- **Interpretation.** Figures 4 and 7 plot the norm of the change to four of the main LSTM's weight matrices under each generated character or pen stroke. Figure 5 compares histograms of the LSTM's cell output.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| MNIST, hypernetwork makes the 7x7x16x16 second kernel, `N_z` = 4 | Test accuracy 99.24% against 99.28% for the normal ConvNet |
| Same, size | 12,544 kernel weights from a 4-number embedding and a hypernetwork of 4,240 parameters (Table 7 gives 4,244) |
| CIFAR-10, WRN 40-1, authors' implementation, against the hypernetwork version | Test error 6.73% with 0.563 M parameters, against 8.02% with 0.097 M |
| CIFAR-10, WRN 40-2, same comparison | 5.66% with 2.236 M, against 7.23% with 0.148 M |
| Penn Treebank, 1000 units, test bits per character | LSTM 1.312 (4.25 M), Layer Norm LSTM 1.267 (4.26 M), HyperLSTM 1.265 (4.91 M), Layer Norm HyperLSTM 1.250 (4.92 M) |
| Penn Treebank, two layers | 2-Layer LSTM 1.281 (12.26 M), 2-Layer Norm HyperLSTM 1.219 (14.41 M) |
| enwik8, test bits per character | Layer Norm LSTM 1800 units 1.402 (14.82 M), HyperLSTM 1800 units 1.391 (18.71 M), Layer Norm HyperLSTM 2048 units 1.340 (26.54 M) |
| IAM handwriting, validation log-loss | LSTM 900 units −1,055, Layer Norm LSTM 900 units −1,096, HyperLSTM 900 units −1,162, Layer Norm HyperLSTM 900 units −1,067 |
| WMT'14 En→Fr, newstest2014, single model | GNMT LSTM 38.95 BLEU, GNMT HyperLSTM 40.03 BLEU. Log perplexity 1.027 against 0.993 |
| MNIST, fully connected 784-256-10, hypernetwork reads pixel and unit coordinates (Appendix A.1) | Best accuracy 93.5%, against 98.5% for the normal network |

The authors also report two observations without a number. The weights change most between words and at brackets, and least inside words such as "Europeans". Most of the time, the HyperLSTM's cell output sits in tanh's saturated range, unlike that of the Layer Norm LSTM.

## Where the claims go beyond the evidence

- **One run per setting, with no seeds or intervals.** No table gives a seed count, a spread or a test. On Penn Treebank the HyperLSTM beats the Layer Norm LSTM by 0.002 bits per character. That is too small to rank the two methods without repeats.
- **The parameter counts differ.** Each HyperLSTM has more parameters than the baseline of the same width, for example 4.91 M against 4.26 M on Penn Treebank. The 1250-unit LSTM (6.57 M) controls for size on that task only, and it scores 1.306. On enwik8 the best result uses 26.54 M parameters against the 14.82 M of the Layer Norm LSTM.
- **Settings chosen on validation data, and handwriting has no test set.** The hyperparameters came from a "non-extensive grid search on the validation set". For handwriting, the authors use the Graves (2013) split, which has no test set. They say themselves that they "may be somewhat over fitting to the validation set" there.
- **The handwriting quality is a judgement.** The authors say that their samples look more coherent. They also say that such judgements are subjective, and they leave the choice to the reader.
- **The weight-change plots are one sample each.** The claim that phrases look more deterministic where the weights change little rests on one generated passage. No measure ties the weight change to any property of the text.
- **Small inconsistencies.** The MNIST hypernetwork has 4,240 parameters in the text and 4,244 in Table 7. Appendix A.1 says 784 × 256 weights and then a 786 × 256 matrix. The CIFAR-10 cost is "∼1.25–1.5%", but Table 2 gives 1.29 and 1.57 points.
- **The best single translation model is one run.** The GNMT HyperLSTM is one model trained once, against the published GNMT number. The authors call it the best single-model result. They state that they did not use the reinforcement learning or quantization of the GNMT paper.

The authors state some limits themselves. The static form loses accuracy on CIFAR-10. Layer norm does not combine well with HyperLSTM on handwriting. The coordinate-based hypernetwork does poorly on MNIST, which is why they use embeddings.

## Use in this repo

- **No configured model is a hypernetwork.** Training wrote the weights of GPT-2, Pythia and Qwen2.5 directly, and no network makes them. So the paper's method does not apply to any unit in the registry. Its relevance is to the strand of #4782 on weights that a network writes, and to #3381.
- **The paper's row scaling is the form of a gated MLP.** Equation 7 scales each row of a weight matrix by an input-dependent number. That is a product of the matrix's output with a vector that the input makes. Qwen2.5's SwiGLU MLP has this form: the gate's activation multiplies the up projection's read, neuron by neuron. So each Qwen neuron's effective up-row changes from token to token. In the paper's terms, part of each Qwen neuron's weights is dynamic.
- **That limits what a weight-only check can prove.** `openinterpretability/harness/dead.py` proves from the weights alone that a pole can't fire. It does this for an ungated MLP by bounding the pre-activation on the norm's ball. For a gated MLP its docstring says that "the sign of the product is free on a ball". So the proof works only when a read is exactly zero and the MLP has no bias. The paper's dynamic weights explain the same limit: a product of two reads has no fixed sign from the weights alone.
- **Weight-only reading has not predicted VERIFIED here.** `docs/analyses/weight-kurtosis.md` (#3226) tested ROTATE's vocabulary-projection kurtosis on 1,724 poles. On 518 test poles the output kurtosis scores an AUC of 0.51 (0.45–0.57), and the input kurtosis 0.54 (0.48–0.60). Both intervals include 0.5. The page adds that no analysis here tests ROTATE itself, the full decomposition of #3446.
- **Can reading weights check an explanation without forward passes?** The paper does not attempt this. Its only interpretation of the weights, Figures 4 and 7, needs a forward pass, because the weights depend on the input. The repo's evidence agrees so far. Weights can prove a narrow negative (`dead.py`) and rank connections (the virtual weights of `openinterpretability/harness/wiring.py`). They have not checked a positive claim about what a pole means.
- **Compressible weights are not evidence for units.** The static hypernetwork makes 12,544 kernel weights from 4 numbers at almost the same accuracy. That shows that trained weights can have low-dimensional structure. It says nothing about whether that structure lines up with single units. `docs/analyses/spd-baseline.md` asks that question of GPT-2's weights. There, SPD keeps whichever basis it starts from, neurons or random directions.
- **No single-neuron content.** The paper gives no unit explanations, probes or random baselines for units. Figure 5 reports saturation over a whole layer, not over single cells. So nothing in it bears on a pole's verdict.
- **No follow-up issue.** The weight-only question already has its tests in this repo, #3226 and #3446. The paper adds no new test that the registry could run.
- **Question for the maintainer.** Should a weight-only reading of a Qwen pole treat the gate as a dynamic scale on the up row? If so, the reading needs the gate's range over a corpus, which is a forward pass. This bears on the scope of any weight-only check under #3446 and #3381. The page records it and opens no issue.
