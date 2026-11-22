# Adding a model

This page gives the process that adds a model to the registry (#3370). A model joins when it is in `[tool.openinterpretability] models` in `pyproject.toml`. Then `open-interpretability next` draws its units, and CI reproduces its records. Pythia-70m is the worked example: [#3838](https://github.com/8W9aG/openinterpretability/pull/3838) added it, and [#4580](https://github.com/8W9aG/openinterpretability/pull/4580) gave it a reproduction floor.

## What a model doesn't need

**A learned dictionary is not needed.** The registry explains a model's own MLP neurons, one pole at a time. A dictionary is a baseline to compare with, not a step before an audit. The evidence that the neuron basis is a fair unit comes from these results:

- On Qwen2.5-0.5B, 18 of 30 native poles were VERIFIED, against 1 of 30 units of a random rotation of the same layers ([`null-calibration.md`](analyses/null-calibration.md)). So the native basis is privileged under this protocol.
- On GPT-2 small, 37 of 40 transcoder latents that fire were VERIFIED, against 20 of 40 positive poles ([`papers/native-vs-learned`](../papers/native-vs-learned/README.md)). Counting the latents that never fire, the gap is no longer significant.

A dictionary can still give more per layer. That paper estimates that auditing every live latent of a layer takes about five times the audits, for ten times the VERIFIED units. So a dictionary is worth adding where one exists or is cheap. GPT-2 small has public transcoders. For Qwen2.5-0.5B, the repo trained its own ([`dictionaries.md`](dictionaries.md)). A new model joins without one.

**An SPD model is not needed.** A parameter decomposition tests whether the weights prefer the neurons. It is not a step in an audit. On GPT-2 small, a decomposition that starts from the neurons ends with a KL of 0.093, against 0.158 from a random start. But a random start doesn't turn towards the neurons ([`papers/spd-input-projections`](../papers/spd-input-projections/README.md)). So SPD can't find a new model's units for it.

## The process

### 1. Wait for the gate

No model joins until every configured model reaches `coverage_target` VERIFIED (default 80%, `openinterpretability/planner_models.py`). The maintainer approves each new model. The `area: models` issues stay at low priority until then.

### 2. Choose the model

- **Its licence allows publishing.** Read its terms on its Hub page. Records quote its outputs, and the site shows them. Llama and Gemma terms add conditions.
- **CI can run it.** CI reproduces each record on CPU runners in float32. Today the largest configured model is Qwen2.5-1.5B. A model too large for CI can only go in `signed_models`. Then each record must be signed, and the planner never draws it.
- **The hooks can find it.** Step 3 checks this.

### 3. Run the check

```bash
uv run open-interpretability model-check <hub name>
```

It loads the model in float32 and prints one line per check: `PASS`, `WARN` or `FAIL`. It exits 1 if any check fails. The checks are in `openinterpretability/model_check.py`:

| Check | What passes | If it fails |
| --- | --- | --- |
| layers, MLP | The layers and the MLP down projection resolve. The neurons are the down projection's input. | Add the paths to `LAYER_PATHS` or `DOWN_PROJ_PATHS` in `harness/hooks.py`. |
| MLP family | Gated (SwiGLU, GeGLU) or plain, with the activation. | Nothing fails here. Read it: a gated negative pole can be large, and a plain GELU negative pole is at most about 0.17. |
| attention | The attention output projection resolves. | Add its path to `ATTN_OUT_PATHS`. Head audits and the wiring need it. |
| norm before the MLP, final norm | Both norms resolve. | Add the paths to `PRE_MLP_NORM_PATHS` or `FINAL_NORM_PATHS`. Dictionary sites read the first. The Jacobian lens and the answer lens read the second. |
| block | Sequential or parallel. | Nothing fails here. A parallel block (GPT-NeoX, GPT-J, Phi) needs `parallel_residual` in `harness/wiring.py` to say so. |
| ablation | Ablating the last layer's strongest pole moves the logits. | The hooks attach to a module that the forward pass doesn't use. |
| final read | The final norm reads one residual stream of the model's width. | The final norm path names the wrong module. |
| models, licence, CI | The config entries of step 4. | Before step 4 these are warnings. After it, a gap fails. |

Run it on the revision that the registry will pin (`--revision`).

### 4. Configure it

Make these changes in one PR:

1. Add the model to `models` in `[tool.openinterpretability]`, in `pyproject.toml`.
2. Add its entry to `[tool.openinterpretability.licenses]`: `license`, `url`, `gated`, `publishing_allowed`, and `conditions` where its terms add any.
3. Add it to `ALLOWED_MODELS` in `.github/workflows/ci.yml`.
4. Optional: a `window_tokens` entry, if the model reads fewer positions than the corpus's documents.
5. Add its row to the *Models* table in [`experiments.md`](experiments.md#models): its MLP, what its negative pole means, and its scan window.
6. Optional: a `SIBLINGS` entry in `openinterpretability/sisters.py`, for a model of the same family, such as the Qwen2.5 sizes.

`tests/test_planner.py` checks that `models` and `ALLOWED_MODELS` agree, and that each model has a licence entry. Then run `model-check` again. Every line must pass.

### 5. Try the ANE

Every new model must try to scan on Apple's Neural Engine (ANE). A failure is an acceptable result. A model that fails still joins, and its scans run on torch. But the attempt and its result must be recorded. Do this step on an Apple-silicon Mac.

1. **Convert it.** The ANE pass is a forward pass written by hand. It runs only the architectures in `ARCHITECTURES` in `openinterpretability/harness/ane/forward.py`. If the model's `config.model_type` is not there, add its forward pass to that file, and its layout to `harness/ane/chunks.py`. If the architecture can't be converted, write down the reason, and go to item 4.
2. **Qualify it.** Run `scripts/ane_qualify.py` on a middle layer and on the last layer:

   ```bash
   uv run python scripts/ane_qualify.py <hub name> --layers <middle>,<last> --corpus corpus.jsonl
   ```

   A layer passes when it passes all five checks of [`ane-qualification.md`](analyses/ane-qualification.md) (top 20, correlation, worst batch, placement and speed).
3. **List it.** If both layers pass, add the model to `QUALIFIED` in `harness/ane/qualified.py`, with the layers and the date of the run. If only the middle layer passes, also add the deepest layer that passes to `DEEPEST`. If no layer passes, don't list the model.
4. **Record it.** Add the result to [`ane-qualification.md`](analyses/ane-qualification.md) and its `.jsonl`, also if it failed. For a model that couldn't be converted, add one line that gives the reason.

### 6. Set the reproduction floor

After the first records exist, measure how far float32 moves the model's activations between machines:

```bash
uv run python scripts/neuron_float32_floor.py --out <scratch>/floor.jsonl wiki/experiments/<org>--<name>/L*/N*/*.json
```

Pass the new model's record files. With no files, the script measures its own fixed selection (`SELECTION`): Pythia-70m's poles and GPT-2's negative poles.

Reproduction compares activations with a tolerance relative to each pole's scale. A small-scale pole can fail it from float32 noise alone. Pythia-70m's activations moved by up to 0.0049, and a GELU negative pole's scale is at most 0.17. In that case, set twice the largest move in `[tool.openinterpretability.reproduce_floor]` ([`neuron-float32-floor.md`](analyses/neuron-float32-floor.md)).

### 7. Let the release refresh it

The next release runs the slow analyses for every configured model (`.claude/skills/release/SKILL.md`). These are the sister search, the registry coverage, the circuit ids and the shared concept map. Nothing else is needed for the model to appear on the site.
