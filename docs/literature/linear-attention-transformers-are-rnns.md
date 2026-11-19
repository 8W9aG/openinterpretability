# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

**Citation.** Angelos Katharopoulos (Idiap Research Institute and EPFL), Apoorv Vyas (Idiap and EPFL), Nikolaos Pappas (University of Washington) and François Fleuret (EPFL and University of Geneva). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention". Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119. arXiv 2006.16236: v1 29 June 2020, v2 30 June 2020, v3 31 August 2020. [Paper](https://arxiv.org/abs/2006.16236). Code, a PyTorch library with about 200 lines of CUDA, is at [linear-transformers.com](https://linear-transformers.com/). The paper says that the authors did the work at Idiap. The Swiss National Science Foundation funded the first three authors. The paper gives no statement about tools or help from an LLM. The abstract page gives no journal reference, but its comment and the PDF both name ICML 2020.

**Status.** Not applicable. The registry explores four models, and all of them use softmax attention. Softmax attention has no finite state of the paper's kind. The paper's state is also written by activations, so it gives no way to check an explanation from weights alone.

**Read.** We read the arXiv v3 PDF on 2026-11-19. We extracted its text with `pdftotext` and read all of it. That is Sections 1 to 5, Algorithm 1, Tables 1 to 5 and Supplementary Sections A to D. The model that wrote this page read the text itself, with no subagent. We did not see the figures as images. So the loss curves of Figures 1, 2 and 5 are here only as their captions and the body text describe them. Some equations came out of the extraction with their layout broken. We read them against the text around them.

## Summary

**Claim.** Self-attention can be written with a kernel feature map φ. Then a layer computes the sums over keys once and reuses them for each query. This makes attention linear in the sequence length. With a causal mask, the sums are a state that each new token updates, so the layer is a recurrent network. Generation is then much faster than with softmax attention, at similar quality.

**Method.**

- **Linear attention.** Softmax attention weights each value by exp(qᵀk/√D). The paper replaces this with φ(q)ᵀφ(k), with φ(x) = elu(x) + 1, applied to each row. By associativity, φ(Q)(φ(K)ᵀV) costs O(N) in place of O(N²).
- **Causal form.** With a mask, output i is φ(Qᵢ)ᵀSᵢ / φ(Qᵢ)ᵀZᵢ. Here Sᵢ is the running sum of φ(Kⱼ)Vⱼᵀ and Zᵢ the running sum of φ(Kⱼ), over j ≤ i.
- **The RNN.** Section 3.4 writes the layer as a recurrent network with two hidden states. They are the attention memory s and the normaliser memory z. Each step adds φ(xᵢW_K)(xᵢW_V)ᵀ to s and φ(xᵢW_K) to z. The step's cost and memory stay constant.
- **Gradients.** Supplementary Section A derives the gradients as cumulative sums, forward for the queries and backward for the keys and values. So training also needs only linear time and constant memory in N.
- **Baselines.** Softmax attention, from the default PyTorch code, and Reformer with 1, 4 or 8 hashing rounds (lsh-1, lsh-4, lsh-8). Supplementary Section C adds "stateful-softmax", a softmax model that keeps all earlier keys and values between steps.
- **Tasks.**
  - A copy task: sequences of up to 128 symbols from an alphabet of 10, with 4 layers and 8 heads.
  - A timing test of one attention layer at lengths 2⁹ to 2¹⁶.
  - MNIST image generation, pixel by pixel: 8 layers, 8 heads, 250 epochs.
  - CIFAR-10 image generation: 16 layers, 7 days on one GPU for each model.
  - Speech recognition on the 80-hour WSJ set with a CTC loss, which is not autoregressive: 9 layers, 6 heads.
- **Metrics.** Bits per dimension and images per second for images. Phoneme error rate (PER) and seconds per epoch for speech.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Copy task, cross-entropy against gradient steps (Figure 2) | Linear "converges smoothly" and reaches the same loss as softmax. Reformer stays higher |
| One layer, GTX 1080 Ti with 11 GB, lengths 2⁹ to 2¹⁶ | The longest length that fits is 4,096 for softmax and 16,384 for lsh-4 and lsh-8. Linear is faster and uses less memory at every length |
| MNIST, bits/dim and images/s | Softmax 0.621 and 0.45. Stateful-softmax 0.621 and 7.56. LSH-1 0.745 and 0.68. LSH-4 0.676 and 0.27. Linear 0.644 and 142.8 (317×) |
| CIFAR-10, 7 days each, bits/dim and images/s | Softmax 3.47 and 0.004. Stateful-softmax 3.47 and 0.32. LSH-1 3.39 and 0.015. LSH-4 3.51 and 0.005. Linear 3.40 and 17.85 (4,462×) |
| CIFAR-10, epochs in the 7 days | Linear completes 3 times more epochs than softmax |
| WSJ speech, PER and seconds per epoch | Bi-LSTM 10.94 and 1,047. Softmax 5.12 and 2,711. LSH-4 9.33 and 2,250. Linear 8.08 and 824 |
| One MNIST image, batch 1, seconds on CPU and GPU | Softmax 72.6 and 10.2. Stateful-softmax 7.4 and 10.4. Linear 5.5 and 7.3 |
| One CIFAR-10 image, batch 1, seconds on CPU and GPU | Softmax 8,651.4 and 300.1. Stateful-softmax 71.9 and 70.4. Linear 45.1 and 61.3 |

## Where the claims go beyond the evidence

- **The title is wider than the method.** Section 3.4 says that the RNN form holds "in theory even" for softmax attention. But Section 3.2.1 says that the feature map of the exponential kernel is infinite-dimensional, so exact softmax can't be made linear. A finite state exists only for a kernel with a finite feature map. The trained models all use elu + 1, not softmax.
- **The paper tests no language model.** The abstract and related work speak of "comparable performance in terms of validation perplexity". The experiments are image generation, speech recognition and a copy task. No experiment models text.
- **"Similar performance" does not hold for speech.** Linear has a PER of 8.08 against 5.12 for softmax. The authors say themselves that softmax is better in this task.
- **The CIFAR-10 quality gain comes from more epochs.** Each model trained for 7 days, and linear completed three times more epochs. The authors say that this test is about speed, not the final score.
- **The 4,000× speed-up is a throughput figure at large batches.** Against stateful-softmax, the throughput gap is about 56× (17.85 against 0.32). With one image at a time on a GPU, linear takes 61.3 s and stateful-softmax 70.4 s.
- **One of the stated speed-ups mixes devices.** Section C.2 says that linear is "almost 6.6× faster" than softmax for one CIFAR-10 image. Table 5 gives 4.9× on the GPU and 191.8× on the CPU. The 6.6 matches linear on the CPU (45.1 s) against softmax on the GPU (300.1 s).
- **One run per setting.** No result has seeds, intervals or a test. Within a task, all models share one optimiser and learning rate, and the paper reports no tuning for each model.

The authors state two limits themselves. They add the stateful-softmax baseline themselves, which makes the speed claims fairer. They also report the speech result where softmax wins.

## Use in this repo

- **The registry's heads are softmax heads, so the RNN form does not carry over.** The explored models are Qwen2.5-0.5B, Qwen2.5-1.5B, GPT-2 and Pythia-70m (`pyproject.toml`). All of them use softmax attention. By the paper's own Section 3.2.1, no finite state Sᵢ exists for them. Their equivalent of a state is the cache of all earlier keys and values, which is the paper's stateful-softmax baseline.
- **Activations write the state, not weights.** Sᵢ is a sum of φ(k)vᵀ over the tokens of one context. To read it, a forward pass must compute each k and v. So the paper gives no way to check a unit's explanation without forward passes, which is the question in #4838's Bears-on line. What stays fixed across contexts is W_Q, W_K and W_V, as in a softmax head.
- **The repo already reads heads from weights, and one analysis measures the limit.** The weights reading in `harness/wiring.py` is virtual weights: earlier heads' OV circuits and later heads' Q, K and V against a unit's read or write. `docs/PROTOCOL.md` says that these numbers rank units but have no absolute scale. The review reads them after the proposer answers. `docs/analyses/copy-suppression-l4h1.md` shows the gap in one case. Pythia-70m's L4/H1 scores 42.0% on the OV test from weights alone. But on ordinary text, copy suppression explains 1.8% of the head's effect. A weight reading found the signature, and only forward passes showed that the signature does little.
- **The head harness needs the attention pattern, which linear attention never forms.** `harness/heads.py` reads previous-token, duplicate-token and induction scores, and the tokens that a peak attends to, from the N×N weights. `NeuronHookRunner.attention_patterns` asks the model for them and raises an error when it returns none. In the paper's equation 9, the N×N weights are never formed. A head in such a layer would need φ(qᵢ)ᵀφ(kⱼ) computed by the harness. This matters only if the registry adds a model with linear-attention layers. #3331 proposes Qwen3.5-0.8B, whose family mixes Gated DeltaNet linear attention with full attention (as `models-rewrite-own-weights.md` reports). The paper does not cover gated or delta-rule forms. The registry adds no new model before every model meets its target, so this is not work for now.
- **It is the base for the fast-weight pages, not one of them.** The paper does not use the term "fast weights". The link between Sᵢ and a fast-weight matrix written by outer products is the claim of Schlag, Irie and Schmidhuber (2021), #4839. That page is the right place to discuss whether the view helps to read units.
- **No single-neuron content.** The paper explains no unit, probe or head. It reports no null and no random baseline. Nothing in it bears on a pole's verdict, on ROTATE (#3446) or on weight-space programming (#3381, #4471).
