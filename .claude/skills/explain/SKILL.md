---
name: explain
description: Explain a model's answer to one prompt from start to finish in Claude Code, with no API key. It runs `open-interpretability explain` for the analysis and the plan, and asks before it spends tokens. It runs the planned experiments by coverage gain through the neuron-proposer, neuron-reviewer and explanation-judge subagents. Then it runs the answer experiment, narrates with explanation-narrator and records the explanation. Use when the user asks to explain a prompt or an answer, or to "run explain", and gives no LiteLLM model or provider.
---

# Explain an answer in Claude Code

`open-interpretability explain` explains a model's answer to one prompt (`docs/experiments.md`, *Explaining one answer*). Its LLM steps are the experiments that the plan lists, the answer experiment and the narration. Outside a session, `explain --run` and `--complete` call LiteLLM models, or start a headless Claude Code session for each experiment (#5831). This skill does all of it in this session, with the repository's subagents answering every prompt. It needs no API key.

Ask the user for the prompt and the model if they didn't give them. The default model is `Qwen/Qwen2.5-0.5B`, and `EleutherAI/pythia-70m` is the fastest. Read `--target` and a limit on experiments from the request. The defaults are a target of 1 and no limit.

## Procedure

1. **Analyse and plan.** This calls no LLM:
   ```bash
   uv run open-interpretability explain --model <model> --target <target> --json -- '<prompt>' > logs/explain-plan.json
   ```
   Read `coverage.coverage`, `coverage_plan` (`now`, `reach`, `experiments`) and `experiments`. Each planned experiment has its unit (`target`), its kind (`draw`), its coverage gain and its `commands`. The plan is in order of gain.
2. **Ask before spending tokens.** Tell the user what will run:
   - the number of experiments that reach the target (`coverage_plan.experiments`), or all of the plan when the target is out of reach
   - at most 5 attempts for each
   - the answer experiment
   - the narration

   Each experiment takes three to five subagent runs. Ask "Continue?" and wait for the answer, unless the user said yes in the request. If the answer is no, stop and give the plan.
3. **Run the planned experiments, by gain.** For each experiment, in the plan's order:
   1. Run it as `land-experiment` does, with its `commands` as the draw. Claim it, read the history, run it, post it, and mark the PR ready. Use the two-step flow of `choose-next-experiment`, *Proposer and reviewer without API keys*: `neuron-proposer` writes the hypothesis, `neuron-reviewer` writes the probes, and two `explanation-judge` subagents score.
   2. Don't wait for CI, and don't merge. `auto-merge.yml` merges the PR once CI passes (#5826).
   3. Run step 1 again, and read the coverage. Say it, as `experiment k of N: <unit>, <verdict>, coverage X%`. An open PR's records count before it merges.
   4. Stop when the coverage reaches the target, when the limit is reached, or when the plan has no experiment left. A unit that isn't VERIFIED after 5 attempts leaves the plan.
4. **Run the answer experiment.** `answer_experiment` in the JSON gives its commands. `answer-proposer` answers the proposer prompt and `answer-reviewer` the reviewer prompt, as their agent files say.
5. **Narrate and record.** Start the record in the background, because the narrator waits for its answer:
   ```bash
   uv run open-interpretability explain --model <model> --target <target> --level long --record --yes -- '<prompt>'
   ```
   In this session the narrator is `external/` with no option (#5831). Ask `explanation-narrator` to answer `exchange/Narrator.prompt.md`. Then the command writes the record and opens its PR. The PR doesn't wait for CI either. To keep everything off GitHub, run step 3 with `--local` records instead (#5827), and share them later with `open-interpretability explain-share <record>`.
6. **Report.** Give the coverage before and after, each experiment's verdict and PR, the answer experiment's verdict and the explanation's PR. Then give the narration's paragraph.

## Notes

- **Record the model IDs that the subagents report**, as `choose-next-experiment` says. Spell each one as a bare lowercase name.
- **One experiment at a time.** Each experiment changes the registry that the next draw and the coverage read.
- **Pause.** If `open-interpretability next` or `pause --status` says new experiments are paused, stop at the next experiment.
- **Without this session.** In a plain terminal with `claude` on `PATH`, `explain --run` runs the same experiments in headless sessions, one each (`openinterpretability/loop_sessions.py`).
