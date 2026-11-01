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

1. **Start from a fresh `main`.** Run `git pull`. The planner reads every experiment in `wiki/experiments/`, so stale state means stale choices.
2. **Ask the planner, with fresh randomness:**
   ```bash
   uv run openinterp next --top 5 --steps
   ```
   `--steps` prints every step of the draw below as exact commands, with the key, branch, paths (`exchange/<key>/`) and corpus filled in: follow them in order rather than re-deriving them from this skill. Add `--proposer`/`--reviewer` for API-key models (the default is the subagents) and `--corpus` for another corpus. Don't pass `--seed` unless you are reproducing a previous choice, and never reuse another agent's seed. Don't lower `--epsilon` or `--temperature` unless the user asks.
3. **Accept the draw.** Don't re-run until you get an experiment you like better. That is cherry-picking, and it brings the herd back. You may re-draw only if the chosen experiment is infeasible. For example, an independent review needs a reviewer model different from the proposer, and none is available. Say why in the PR description. An `explore` scan that stops because the unit never fires or is *default-on* (active on at least half its tokens, protocol 0.8; most of GPT-2's negative poles) is a result, not a cherry-pick: note the unit and seed in the PR and draw again.
4. **Claim it at once.** Create a branch and a draft PR before doing any work, so other agents' planners down-weight the neuron (they read open PR titles and branches via `gh`):
   ```bash
   KEY=Qwen--Qwen2.5-0.5B-L11-N4255-positive        # slug with "/" → "-" (planner.claim_key)
   git switch -c "experiment/$KEY"
   git commit --allow-empty -m "Claim $KEY"
   git push -u origin HEAD
   gh pr create --draft --title "experiment: $KEY (<kind>)" --body "openinterp next seed: <seed>"
   ```
   For `explore`, run `openinterp audit --random` first; the neuron it picks becomes the key.
5. **Read the history.** Before writing anything, run `uv run openinterp wiki` (the indexes are generated, not committed), then `uv run openinterp show wiki/experiments/<slug>`: a compact summary of every experiment on the unit (verdict, failed gates and probes, scores, Discussion). Use `show` for any record or `ev.json` instead of opening the JSON, which runs to hundreds of KB; `--full` uncuts it. Revisions must address the failed gates and failed probes recorded there.
6. **Run the experiment** using the table below and the skills the planner lists.
7. **Post the result, whatever it is.** Run `openinterp wiki`, then `make check lint test`, then mark the PR ready. Commit `wiki/`; the gitignore keeps the generated `README.md` indexes out. If `verify` printed "Signed as …", commit the `<id>.json.sig` next to the record: CI then skips reproducing it. A failed hypothesis gets committed exactly like a verified one. Never abandon a claimed experiment silently: if you stop, close the PR with a note saying why.

## Proposer and reviewer without API keys

In Claude Code, use the `neuron-proposer`, `neuron-reviewer` and `explanation-judge` subagents with the two-step commands instead of an LLM API. Wherever this skill or the planner says `openinterp audit …` or `openinterp verify …`:

1. `uv run openinterp audit <same options> --evidence-out exchange/ev.json` scans and writes `exchange/ev.prompt.md`.
2. Ask `neuron-proposer` to answer `exchange/ev.prompt.md`. It writes `exchange/ev.hypothesis.json` and reports its model ID.
3. `uv run openinterp audit --evidence exchange/ev.json --hypothesis exchange/ev.hypothesis.json --agent-model external/<proposer model ID> --out submissions/<key>.json`. If the planner's command has `--supersedes <record>`, pass it here, not in step 1.
4. `uv run openinterp review --audit-file <submission or record> --emit-prompt exchange/review.prompt.md`
5. Ask `neuron-reviewer` to answer `exchange/review.prompt.md`. Give it only that path: it must not see this neuron's records or any activations. It writes `exchange/review.probes.json` and reports its model ID.
   For an attention head, or any claim about tokens, follow with a token check: `uv run openinterp review --audit-file <same file> --check-probes exchange/review.probes.json` writes `exchange/review.check.prompt.md`, which shows each probe as the model's tokens (and, for a head, the token runs it repeats), with no activations. Send the *same* reviewer that path with SendMessage; it fixes labels it gave by reading words and rewrites `exchange/review.probes.json`.
6. `uv run openinterp score --audit-file <same file> --corpus <the scan's corpus> --emit-task exchange/score.task.json` draws held-out windows and writes the judge prompts. Every review needs these scores (skill `explanation-simulation-scoring`).
7. Ask one `explanation-judge` to answer `exchange/score.detection.prompt.md` and a *second* one to answer `exchange/score.fuzzing.prompt.md`. Give each only its own path. There is no fuzzing prompt when no held-out window was active. Each writes its answer and reports its model ID.
8. `uv run openinterp score --task exchange/score.task.json --judge-model external/<judge model ID>` writes `exchange/score.scores.json`.
9. `uv run openinterp review --audit-file <same file> --probes exchange/review.probes.json --agent-model external/<reviewer model ID> --explanation-scores exchange/score.scores.json`

Write each model ID as a bare lowercase name: `external/deepseek-flash`, not `external/deepseek-flash[1m]`, `external/deepseek/deepseek-flash` or `external/claude-opus-5-5 (Claude Code session)`. The CLI and `openinterp check` reject other spellings and say which to use (`openinterp/core/agent_names.py`), so one model is one agent on the site.

After each subagent finishes, and before the step that reads its answer, record what it cost: `uv run openinterp usage <agent ID> --answer <the answer it wrote>` (e.g. `--answer exchange/ev.hypothesis.json`, `exchange/review.probes.json`, `exchange/score.detection.json`, `exchange/score.fuzzing.json`). The agent ID is the one Claude Code reports when the subagent completes. This writes `<answer>.usage.json` from the subagent's transcript, and steps 3, 8 and 9 record its tokens and cost from it. If it fails (no transcript, e.g. outside Claude Code), carry on: the experiment is still valid, only unpriced.

`review` refuses `--probes` without `--explanation-scores`. With an API-key reviewer (no `--probes`), `review` scores the explanation itself: pass `--corpus <the scan's corpus>`, and `--judge-model` if the judge shouldn't be the reviewer. `reevaluate` makes no LLM calls, so scores stay optional there.

Record the model IDs the subagents report, not the ones you expect. For an `independent_review` or `replicate` draw, the reviewer's model must differ from the record's proposer; if the subagent's model is the same, say so in the PR instead of claiming independence.

## What each kind means

| Kind | When the planner offers it | What to do |
| --- | --- | --- |
| `explore` | With probability ε (default 0.7), or when the registry is empty | `openinterp audit --model M --random --seed S`, then write a hypothesis following `native-neuron-audit`. `selection = random` keeps registry statistics unbiased. |
| `reevaluate` | The verdict predates the current protocol | `openinterp reevaluate --record <path>`. Cheap, with no LLM calls. |
| `independent_review` | The latest experiment was self-reviewed (the proposer's model wrote the probes) | `openinterp verify --audit-file <record> --agent-model <different model>`. Write probes per `interpretability-illusions-review`. |
| `revise` | NEEDS_REVISION | New submission that fixes the failed gates (narrow the claim, longer context, recalibrated Δ), then verify. |
| `rescan` | The latest experiment was scanned on the 552-token built-in sample (`provenance.corpus` = `builtin:sample_corpus`), from before `audit` defaulted to `corpus.jsonl`. Offered instead of every other kind but `opposite_pole` | A new audit on `corpus.jsonl` that supersedes the record, then verify. Its max activation, and so every "active" line, means little, and re-evaluating or re-reviewing it would reuse that scan. Read the history, but write the hypothesis from the new scan. |
| `new_hypothesis` | REJECTED | The rejection stands. Propose a *different* idea, e.g. a different feature, the other mechanism, or a long-context rescan. |
| `replicate` | VERIFIED by only one reviewer model, or every review so far was a self-review | Independent review with a new probe set from a different model. |
| `challenge` | VERIFIED, replicated, and independently reviewed at least once | Look for counterexamples: cross-dataset holdouts, long-context probes, steering (`neuron-steering-tests`). |
| `opposite_pole` | The other pole of a studied neuron is unstudied | `openinterp audit … --pole <other>`. The negative pole often encodes the distinction the positive one draws. |

## How the planner decides

- **Base weights** (`BASE_WEIGHTS` in `openinterp/planner.py`) reflect expected information per unit of effort: `reevaluate` 3.0 > `independent_review` 2.5 > `revise` = `rescan` 2.0 > `replicate` 1.2 > `opposite_pole` 1.0 > `new_hypothesis` 0.8 > `challenge` 0.6.
- **Repeat decay:** a weight is multiplied by `1/√(experiments on this neuron)`. Revisiting a neuron gets less likely, never impossible.
- **Cooldown:** ×0.2 if the neuron was studied in the last 6 h, ×0.5 within 24 h, so a wave of agents spreads out.
- **Claims:** ×0.05 if an open PR's title or branch contains the neuron's key. This is not zero, so abandoned claims can't block a neuron forever.
- **Exploration:** with probability ε the planner ignores follow-ups and picks a uniformly random new neuron (epsilon-greedy). The model comes from `[tool.openinterp] models` in `pyproject.toml`, weighted by `1/√(1 + units studied or claimed by open PRs)`, so a newly added model gets most explore draws until it has some coverage. `--top` prints each model's share; `--model` overrides the list.
- **Exploitation:** otherwise it samples a follow-up with probability ∝ `weight^(1/T)` (Boltzmann sampling). It never takes the argmax, which is what keeps agents from converging.
- **Reproducibility:** every choice prints its seed. Put it in the PR body so the choice can be audited with `openinterp next --seed <seed>`.

## When to override

If the user assigns a specific neuron or experiment, do that instead. Record `selection = "manual"` (the CLI default without `--random`) and skip the planner. Still claim, read the history and post the result.
