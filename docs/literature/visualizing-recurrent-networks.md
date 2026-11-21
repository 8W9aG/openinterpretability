# Visualizing and Understanding Recurrent Networks

**Citation.** Andrej Karpathy, Justin Johnson and Li Fei-Fei, Department of Computer Science, Stanford University. "Visualizing and Understanding Recurrent Networks". [arXiv:1506.02078](https://arxiv.org/abs/1506.02078), version 1 of 5 June 2015, revised as version 2 on 17 November 2015. The page header of version 2 says "Under review as a conference paper at ICLR 2016". The text names no later venue, so this page treats it as a preprint. The paper links no code.

**Mismatch.** Figure 1 says "The test set has 300,000 characters", but Section 4.4 counts "330K test set characters". The text also sets the 20-gram model against 1.077 on *War and Peace*, the GRU's best. On the Linux kernel it uses 0.84, the LSTM's best.

**Status.** Not applicable. No code uses it. `papers/syntax-positive-controls` cites it for the expectation that code models hold syntax units.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It had the whole text in the source's own words, in two fetches. The page's author then read pages 5 and 6 of the PDF as images, to see Figure 2 and Table 2.

## Summary

**Claim.** Character-level LSTMs hold some cells with clear jobs, such as tracking line length, quotes or brackets. The LSTM's edge over n-gram models comes from characters that need long-range context.

**Method.**

- **Models.** Vanilla RNNs, LSTMs and GRUs with 1 to 3 layers, at four sizes from about 50K to 1.3M parameters.
- **Data.** *War and Peace*, 3,258,246 characters. The Linux kernel source, with its files shuffled into one 6,206,996-character file.
- **Training.** Next-character prediction, truncated backpropagation over 100 steps.
- **Analyses.** Cell activations read by eye, gate saturation, comparisons with n-gram and n-NN baselines, and errors sorted by oracles.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Interpretable cells (Section 4.2, Figure 2) | Some cells "turn on inside quotes, the parenthesis after if statements, inside strings or comments". One rises "with increasing strength as the indentation of a block of code increases" |
| Range of a quote cell (Section 4.2) | "∼230 characters in the quote detection cell example in Figure 2", longer than the 100-step training window |
| Test cross-entropy, best LSTM (Figure 1) | 1.082 on *War and Peace* and 0.840 on the Linux kernel |
| 20-gram baseline (Table 2) | 1.195 on *War and Peace* and 0.889 on the Linux kernel |
| Errors of the best LSTM on *War and Peace* (Section 4.4) | 42% of test characters. An n-gram oracle removes 18% of them |

## Where the claims go beyond the evidence

- **Cells found by eye.** The paper shows "Several examples" of interpretable cells and gives no count or share of them. Figure 2 notes that "A large portion of cells are not easily interpretable".
- **No seeds.** Each model is one run. Bootstrap errors are below 0.004, but they measure the test sample, not the training.
- **Unstated datasets.** The text doesn't say which model each named cell comes from. Figure 2's examples show that the quote cell reads prose from *War and Peace*, and the code cells read C.
- **No causal test.** No cell is ablated or clamped. Whether the model uses a cell is not tested.

## Use in this repo

- **Our citations check out.** `papers/syntax-positive-controls` cites it four times.
  - LSTM cells trained on the Linux kernel source, with the two quoted phrases, match Section 4.2 word for word.
  - "a quote cell that holds for about 230 characters" matches the text.
  - Our paper calls the step from these cells to GPT-2 an assumption. The source studies recurrent models only, so that holds.
  - The three authors, 2015 and the arXiv link match.
- **The quote cell reads prose.** Figure 2's quote cell is on *War and Peace*, not on code. `papers/syntax-positive-controls` already says their cell and its pole "measure different things on different data". So the paper needs no addendum.
- **Our test does what this paper didn't.** The source reads cells by eye. `papers/syntax-positive-controls` ablates L2 N676+ and measures the model's next closer. Our reading: that is the causal step that the source never takes.
- **No question for the maintainer.**
- **Related pages.** [`who-cares-about-brackets.md`](who-cares-about-brackets.md) and [`simplified-model-illusions.md`](simplified-model-illusions.md) review two later studies of brackets in transformers.
