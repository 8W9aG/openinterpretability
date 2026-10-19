---
name: choose-next-experiment
description: Decide which experiment to run next, using `openinterp next` (randomised, balancing exploration of new neurons against follow-ups on studied ones, and resistant to many agents piling onto one neuron). Use whenever an agent is starting new work in this repo without a specific neuron assigned, asks "what should I do next?", or finishes an experiment and continues.
---

# Choosing the next experiment

The registry improves through two kinds of work:
- **exploration:** auditing neurons nobody has looked at;
- **exploitation:** following up on neurons already studied, i.e. revising, independently re-reviewing, replicating, challenging, or auditing the other pole.

Both matter, and so do null results. A REJECTED or NEEDS_REVISION verdict is data, and studying a neuron again is legitimate. What this skill prevents is two failure modes: every agent converging on the same "best" experiment (a thundering herd), and agents only ever polishing a few favourite neurons.

## Procedure

1. **Start from a fresh `main`.** Run `git pull`. The planner reads `audits/` and `wiki/`, so stale state means stale choices.
2. **Ask the planner, with fresh randomness:**
   ```bash
   uv run openinterp next --top 5
   ```
   Don't pass `--seed` unless you are reproducing a previous choice, and never reuse another agent's seed. Don't lower `--epsilon` or `--temperature` unless the user asks.
3. **Accept the draw.** Don't re-run until you get an experiment you like better. That is cherry-picking, and it brings the herd back. You may re-draw only if the chosen experiment is infeasible. For example, an independent review needs a reviewer model different from the proposer, and none is available. Say why in the PR description.
4. **Claim it at once.** Create a branch and a draft PR before doing any work, so other agents' planners down-weight the neuron (they read open PR titles and branches via `gh`):
   ```bash
   KEY=Qwen--Qwen2.5-0.5B-L11-N4255-positive        # slug with "/" → "-" (planner.claim_key)
   git switch -c "experiment/$KEY"
   git commit --allow-empty -m "Claim $KEY"
   git push -u origin HEAD
   gh pr create --draft --title "experiment: $KEY (<kind>)" --body "openinterp next seed: <seed>"
   ```
   For `explore`, run `openinterp audit --random` first; the neuron it picks becomes the key.
5. **Read the history.** Before writing anything, read `wiki/experiments/<slug>/README.md` and every experiment page it lists, including Discussion sections. Revisions must address the failed gates and failed probes recorded there.
6. **Run the experiment** using the table below and the skills the planner lists.
7. **Post the result, whatever it is.** Run `openinterp wiki`, then `make check lint test`, then mark the PR ready. A failed hypothesis gets committed exactly like a verified one. Never abandon a claimed experiment silently: if you stop, close the PR with a note saying why.

## What each kind means

| Kind | When the planner offers it | What to do |
| --- | --- | --- |
| `explore` | With probability ε (default 0.35), or when the registry is empty | `openinterp audit --model M --random --seed S`, then write a hypothesis following `native-neuron-audit`. `selection = random` keeps registry statistics unbiased. |
| `reevaluate` | The verdict predates the current protocol | `openinterp reevaluate --record <path>`. Cheap, with no LLM calls. |
| `independent_review` | The proposer reviewed its own hypothesis | `openinterp verify --audit-file <record> --agent-model <different model>`. Write probes per `interpretability-illusions-review`. |
| `revise` | NEEDS_REVISION | New submission that fixes the failed gates (narrow the claim, longer context, recalibrated Δ), then verify. |
| `new_hypothesis` | REJECTED | The rejection stands. Propose a *different* idea, e.g. a different feature, the other mechanism, or a long-context rescan. |
| `replicate` | VERIFIED by only one reviewer model | Independent review with a new probe set from a different model. |
| `challenge` | VERIFIED and replicated | Look for counterexamples: cross-dataset holdouts, long-context probes, steering (`neuron-steering-tests`). |
| `opposite_pole` | The other pole of a studied neuron is unstudied | `openinterp audit … --pole <other>`. The negative pole often encodes the distinction the positive one draws. |

## How the planner decides

- **Base weights** reflect expected information per unit of effort: `reevaluate` 3.0 > `independent_review` 2.5 > `revise` 2.0 > `replicate` 1.2 > `opposite_pole` 1.0 > `new_hypothesis` 0.8 > `challenge` 0.6.
- **Repeat decay:** a weight is multiplied by `1/√(experiments on this neuron)`. Revisiting a neuron gets less likely, never impossible.
- **Cooldown:** ×0.2 if the neuron was studied in the last 6 h, ×0.5 within 24 h, so a wave of agents spreads out.
- **Claims:** ×0.05 if an open PR's title or branch contains the neuron's key. This is not zero, so abandoned claims can't block a neuron forever.
- **Exploration:** with probability ε the planner ignores follow-ups and picks a uniformly random new neuron (epsilon-greedy).
- **Exploitation:** otherwise it samples a follow-up with probability ∝ `weight^(1/T)` (Boltzmann sampling). It never takes the argmax, which is what keeps agents from converging.
- **Reproducibility:** every choice prints its seed. Put it in the PR body so the choice can be audited with `openinterp next --seed <seed>`.

## When to override

If the user assigns a specific neuron or experiment, do that instead. Record `selection = "manual"` (the CLI default without `--random`) and skip the planner. Still claim, read the history and post the result.
