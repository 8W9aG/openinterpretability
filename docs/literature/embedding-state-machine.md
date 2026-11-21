# Embedding a State Machine in a Neural Network

**Citation.** Edward (Ted) Benson. "Embedding a state machine in a neural network". Blog post, edwardbenson.com, 2 November 2026. [Post](https://edwardbenson.com/2026/11/embedding-a-state-machine-in-a-neural-network). No affiliation given. "All rights reserved". The post links no code. The site says that the blog is "human authored". It is the third post in a series, after posts on boolean circuits and on a latch, which we did not read.

**Status.** Not applicable. The post designs a state machine from hand-wired binary units, but it puts the design in no model, gives no code and measures nothing. The repo's ground-truth calibration already plants known units in GPT-2 (#3201). One idea carries over, in *Use in this repo*.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the whole post as extracted text, in the post's own words. That includes three pseudocode blocks and the text labels of an interactive widget. We could not run the widget, and we saw no images. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A finite state machine can be built from neurons that output a hard 0 or 1. Each state is a latch, and each transition is a boolean circuit, "S and Condition". One signal resets the old state's latch and sets the new one's.

**Method.** A worked example, a tea maker with three states (Waiting, Steeping, Ready), one-hot encoded. Four input signals drive it: hot water, tea present, timer done and cup emptied. The post assumes a "region of the network where every neuron outputs a hard `0` or `1`". It does not say which network, or whether the weights are set by hand or trained.

**Findings, as the source reports them.** The post reports no measurement.

| Setting | Result |
| --- | --- |
| Tea example | Three states, one-hot: [1, 0, 0], [0, 1, 0], [0, 0, 1] |
| Starting state | A special rule fires when every state is off, with "a delay of one update cycle" |
| Self-transitions | Need special handling, because the latch puts reset before set |
| Race conditions | Open: "I'm not sure here." The post proposes equal cost per rule and a global order |

## Where the claims go beyond the evidence

- **No host model.** The post never names the network. Real MLP neurons in a language model are not hard binary units, so the design's main assumption is untested.
- **No evidence that it runs.** The only demonstration is an interactive widget, with no code and no recorded output.
- **No side effects.** The post aims to "attach a control circuit to personalities in the LLM" later, but measures nothing about a host model.

The post states its open problems itself: race conditions, the starting state and self-transitions. It calls the earlier posts "hastily done".

## Use in this repo

- **Planted units already exist here.** [`docs/analyses/ground-truth-calibration.md`](../analyses/ground-truth-calibration.md) rewrites one GPT-2 neuron's input weights as a probe for a known label rule. Then it checks whether the pipeline's explanation names that rule. The post's units would be the same kind of test, if someone built them into a real model.
- **A multi-unit plant.** The idea that carries over: a planted circuit with state, not one planted neuron. The calibration's gate kept 7 of 12 plants, and none of the three rules that need context. One example is "inside quotes", at precision 0.59 and recall 0.50. A probe written into one neuron could not hold that state. A latch holds its value over tokens, so a planted latch is one way to build such a rule. This is our step, not the post's.
- **Compiled transformers cover the same ground with code.** [`tracr.md`](tracr.md) and [`alta.md`](alta.md) compile programs into transformer weights, and both give code. A state-machine plant would start from those, not from this post.
- **No follow-up issue.** The post gives nothing that runs. A stateful plant is a possible extension of #3201, which is closed, and the maintainer can reopen that line if wanted.
