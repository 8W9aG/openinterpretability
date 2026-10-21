---
name: native-neuron-audit
description: Methodology for auditing a native MLP neuron pole without misreading it, following Goodfire's "Neural Networks Are Already Clean" (Cammarata, 2026). Use before writing any hypothesis for `openinterp audit`, when a neuron looks polysemantic, noisy or dead, or when choosing scan corpus, context length and activation range.
---

# Native neuron audit

The source is Nick Cammarata / Goodfire, *Neural Networks Are Already Clean* (15 Oct 2026, https://already-clean.goodfire.com/). It is a working-hypothesis essay, and its technical companion paper has not been published. In a preliminary Qwen3-32B audit, about 90% of neuron poles had one main job, about 7% had two related jobs, and 2–4% were not yet understood. The essay's central claim is that most *apparent* polysemanticity came from fixable tooling problems rather than superposition. Treat a messy-looking neuron as a possible tool failure before concluding that it is polysemantic.

## Checklist before writing a hypothesis

1. **Long context and broad data.** Short inputs make long-context neurons look polysemantic, and narrow datasets hide what a neuron actually tracks: one "noise" neuron matched health SEO spam once the data was broad enough. Scan windows of ≥ 512 tokens over a mixed corpus (web, code, math, dialogue, non-English). The 64-token windows used in the first `L11/N4255` run are too short to rule this out.
2. **Look forward as well as back.** One tool missed a neuron predicting that "boat" comes next because it only showed preceding context. Always print tokens *after* the peak and run the `vocab-projection-lookahead` skill.
3. **Check the punctuation "summary motif".** Language neurons often fire on punctuation that summarises the preceding clause. A peak on `,` `.` `)` or `\n` is a reason to read the whole preceding span, not a reason to dismiss the neuron.
4. **Look for offshoots, not unrelated meanings.** Neurons grow related offshoots over training (e.g. "iy" → "diy" → "do it yourself"). Before claiming two jobs, ask whether the second is a generalisation of the first. If the model publishes intermediate checkpoints (e.g. Pythia), compare them.
5. **Describe the shared pattern, not a list of nouns.** Ask the essay's question: *if I couldn't use groups of nouns at all, how would I describe this neuron?*
6. **Audit both poles.** The negative pole gets gradient whenever the neuron is active, and often encodes the distinction the positive pole draws (the essay's V-neuron's negative pole is an O). Scan `--pole negative` too, and mention the contrast in the explanation.
7. **Don't standardise activation scales across neurons.** Use each neuron's own range. `openinterp` already does this: the "active" threshold is a fraction of the neuron's own `max_activation`.
8. **Dial or detector?** Check the shape of the histogram. A detector has a near-zero bulk and a separate tail; a dial varies smoothly. Percentiles hide the difference, so read the raw histogram. Results also depend on how far into weak activations you look, so report examples from several strength bands (see the `explanation-simulation-scoring` skill).
9. **Leftovers point at tool problems.** If a consistent subset of examples doesn't fit, suspect the tool first. For SwiGLU models (Qwen, Llama, Mistral), run the regime decomposition below.
10. **Say so when you don't know.** A narrow, true explanation with acknowledged leftovers beats a broad one that the reviewer's holdouts will break.

## SwiGLU sign-regime decomposition

Gated MLPs compute `a = silu(g) * u`. Because `silu(g)` is slightly negative for `g < 0` (minimum ≈ −0.28), a positive `a` can come from two regimes: the usual `g > 0, u > 0`, or `g < 0, u < 0` (both factors negative). The essay reports finding a "SwiGLU double-negative regime" by chasing unexplained leftovers. The decomposition below is our own check for it; the essay does not give its exact definition.

```python
import torch
from openinterp.harness.hooks import NeuronHookRunner


def swiglu_regimes(runner: NeuronHookRunner, text: str, layer: int, idx: int):
    mlp = runner.layers[layer].mlp  # Qwen2/Llama: gate_proj, up_proj, down_proj
    cache = {}
    hooks = [
        mlp.gate_proj.register_forward_hook(lambda m, i, o: cache.__setitem__("g", o[0, :, idx])),
        mlp.up_proj.register_forward_hook(lambda m, i, o: cache.__setitem__("u", o[0, :, idx])),
    ]
    try:
        with torch.no_grad():
            runner.forward(runner.encode(text))
    finally:
        for h in hooks:
            h.remove()
    g, u = cache["g"], cache["u"]
    a = torch.nn.functional.silu(g) * u
    regime = [("g+" if gi > 0 else "g-") + ("u+" if ui > 0 else "u-") for gi, ui in zip(g, u)]
    return a, regime  # positive-pole peaks in "g-u-" belong to the double-negative regime
```

If the top positive-pole examples split across `g+u+` and `g-u-`, they are probably different mechanisms. Audit them as separate hypotheses, or state the split in the explanation.

## Fit with the protocol

- Scan long documents with `openinterp audit --window-tokens 512` (stride defaults to half a window). Provenance records the window length and stride (`snippet_tokens`, `snippet_stride`). Reviewers will discount short-context scans.
- Use the "Too little room" and "offshoot" points when arguing that a two-job neuron is really one job. The reviewer's positive holdouts should then include the offshoot.
- Steering ("what the model writes when the neuron is turned up") is one of the essay's main checks. See the `neuron-steering-tests` skill.
