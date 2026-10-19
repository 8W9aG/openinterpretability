---
name: neuron-steering-tests
description: Test a neuron hypothesis generatively by clamping the pole high (or ablating it) and reading what the model writes. Use when a hypothesis makes a claim about what the neuron makes the model say, when causal patching metrics are uninformative, or as an extra check before submitting.
---

# Neuron steering tests

## Papers

- Turner et al., 2023, *Steering Language Models With Activation Engineering* (ActAdd), arXiv:2308.10248. Adding activation directions at inference changes generated text predictably.
- Goodfire, *Neural Networks Are Already Clean* (2026), https://already-clean.goodfire.com/. "What the model writes when the neuron is turned up" is one of the essay's standard checks, alongside top activations and ablation. Its ablation examples measure next-word probabilities directly (e.g. P(return) 0.83 → 0.98 after removing one holding-back neuron).

## Procedure

```python
import torch


def steer(runner, prompt, layer, idx, pole, value, max_new_tokens=40):
    sign = 1.0 if pole == "positive" else -1.0

    def clamp(act):
        act = act.clone()
        act[..., idx] = sign * value  # value in units of the neuron's own max_activation
        return act

    enc = runner.encode(prompt)
    with torch.no_grad(), runner.hooks({layer: clamp}):
        out = runner.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return runner.tokenizer.decode(out[0, enc["input_ids"].shape[1] :])
```

1. Pick 5 neutral prompts unrelated to the hypothesis (e.g. "The weather today is").
2. Generate at `value = 0` (ablated) and at `value ∈ {1, 2, 4} × max_activation`, all greedy.
3. A good hypothesis predicts the steered continuations **before you read them**. Write the prediction down first, then compare.
4. Also measure next-token probabilities on the hypothesis' own base prompts with the pole ablated. This is cheaper and quantitative.

## Interpreting the result

- Coherent, predicted drift at 1–2× max supports the claim, and is strong evidence for lookahead neurons.
- Drift that appears only at ≥ 4× is off-distribution and weak evidence.
- No effect is common for single neurons in large models; redundancy and self-repair absorb it. That doesn't refute a detection claim. It refutes only an "it causes X" claim.

## Fit with the protocol

Steering results are not yet gated. Put predictions and outcomes in the explanation text and keep any claim proportional. If steering should become a gate, it needs a pre-registered prediction format; propose that as a protocol change.
