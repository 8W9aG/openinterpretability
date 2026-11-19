# Fix patches

A fix to a model is not a new model. It is a **patch**: a record that says how to change a base model, as a git commit says how to change a tree (#5229). The patched model is built only to run the fix's own checks, and it never enters the registry (#5220 §9.2). The code is in `openinterpretability/patches/`.

## The record

A patch lives at `wiki/patches/<model>/<id>.json`, and it is signed when written.

| Field | What it holds |
| --- | --- |
| `id` | The SHA-256 of the record without `id`, like a commit hash. |
| `base` | The base model and its revision, `<model>@<revision>`. |
| `parents` | The patches it builds on. A root has none, and a merge has two or more. |
| `registry_version` | The operation registry it needs. A reader refuses a newer one. |
| `ops` | Its declarative operations, applied in order after all its ancestors. |
| `message` | The failure it fixes, the explanation it rests on, and the method. |
| `evidence` | The fix's checks, and its `verdict`: `fixed` or `not fixed`. |
| `provenance` | Who or what wrote it, and when. |

Each operation has an `op_id`, a `type` from the registry (`patches/schema.py`), its `targets`, its `params` and an optional `gate`. Targets are names in the canonical model graph (`patches/graph.py`), such as `blocks.3.mlp.W_out` or `blocks.4.hook_resid_post`, so one patch format covers every architecture. Large tensors are kept by hash in the store beside the record evidence. A patch never holds code: a new operation type comes through a reviewed PR, and the registry's version goes up.

## Applying a patch

A patch applies through forward hooks, so the model's weights are never written and removing the patch is exact. Every op can name a gate, a part of the patch that decides at each position whether the op acts. The zero-start check runs the patch with every output scale at zero, and it fails if any logit moves beyond float32 noise.

## The workflow

1. Write a **draft**: the record's JSON with no `id`, whose parents are already in the store.
2. Check that it is the identity at zero strength:

   ```bash
   uv run open-interpretability patch check draft.json --model EleutherAI/pythia-70m
   ```

3. Run the fix's own checks with the patch applied. Each one writes its result to a file, never to the registry:

   ```bash
   uv run open-interpretability answers review --claim claim.json --probes probes.json \
       --patch draft.json --evidence-out answers.json
   uv run open-interpretability explain "A week has seven days and a year has twelve" \
       --model EleutherAI/pythia-70m --patch draft.json --evidence-out explain.json
   ```

4. Seal the draft with its evidence and its verdict. This fixes its `id`, writes the record and signs it:

   ```bash
   uv run open-interpretability patch seal draft.json --evidence answers=answers.json \
       --evidence explain=explain.json --verdict fixed
   ```

## History and branches

- `patch log`, `patch show` and `patch diff` read the history.
- `patch branch NAME REF` moves a branch, and `--create` makes one. Each base model's `main` branch only moves to a patch whose verdict is `fixed`, so `main` is the accepted set of fixes.
- `patch merge` makes a merge. It is refused while two sides edit the same parameters, unless `--resolution additive` is given. It is also refused while ops that are not additive share a hook point, unless `--order` sets their order. Units in both sides' footprints don't block it, but the merge's message lists them. A merge always needs its own evidence.
- `patch revert` makes a patch that drops chosen ops. `patch rebase` copies patches onto a new parent or base revision. Both need their evidence run again.

Some operation types are in the registry but not built yet, and `apply` refuses them by name: #5298 lists them.
