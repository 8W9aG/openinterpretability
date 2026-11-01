# Smaller driver models: pilot

Run on 2026-10-31 (TODO "Performance"). Question: once `openinterp next --steps` and `openinterp show` (PR #1401) do the work the driving session used to do from the skills and the JSON, can a smaller model drive an experiment end to end as reliably as Opus, and what does it save?

## Method

- **Arms.** Two `scripts/experiment-loop.sh` runs of 4 experiments each, one with `--driver claude-sonnet-5-5` and one with `--driver claude-haiku-5-5`. The two runs went in parallel, each in its own worktree, from `origin/main` at b41d6a18.
- **Everything else fixed.** Both arms used `--proposer external/claude-opus-5-5 --reviewer external/claude-sonnet-5-5`, so the `neuron-proposer` (opus), `neuron-reviewer` (sonnet) and `explanation-judge` (haiku) subagents answered. The subagents pin their own models, so the driver never wrote a hypothesis, a probe or a judgement. Every draw was a real `openinterp next` draw and landed in the registry. Nothing was synthesised for the pilot.
- **Baseline.** The 38 Opus single-experiment sessions in this project's Claude Code transcripts from 2026-10-19. They ran `choose-next-experiment` up to an open PR, without `--steps` or `show` and without land-experiment's build, CI wait and merge. So they are a reference point, not a matched arm.
- **Measures.** Read from each session's transcript, counting the driver's own turns only, not its subagents':
  - *Completion:* a `RESULT` line and a merged PR with no help.
  - *Correctness:* model IDs recorded as reported, explanation scores present, record signed.
  - *Size:* turns, tokens and wall time.
  - *Driver cost:* the session's `modelUsage[driver].costUSD` (list price), minus the share used by same-model subagents (the reviewer in the Sonnet arm, the judges in the Haiku arm). That share is split in proportion to tokens.
  - *Opus baseline cost:* estimated from the Opus proposer subagents' cost per token in these sessions. Treat it as rough.

## Results

| Driver | Draw | Unit | Verdict | PR | Turns | Input tokens | Output | Driver cost | Minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sonnet | independent_review | Qwen L17/N3962− | REJECTED | #1403 | 32 | 1.70M | 6.6k | $0.43 | 9.7 |
| Sonnet | explore | GPT-2 L2/N2682+ | NEEDS_REVISION | #1406 | 25 | 1.35M | 6.6k | $0.38 | 8.9 |
| Sonnet | explore | GPT-2 L1/N2717+ | VERIFIED | #1409 | 27 | 1.50M | 6.0k | $0.41 | 9.3 |
| Sonnet | explore | GPT-2 L11/N438+ | VERIFIED | #1413 | 29 | 1.71M | 7.2k | $0.42 | 9.7 |
| Haiku | reevaluate | Qwen L20/N2248− | REJECTED | #1402 | 19 | 1.02M | 6.2k | $0.02 | 6.0 |
| Haiku | explore | GPT-2 L6/N94+ | VERIFIED | #1404 | 30 | 1.74M | 10.8k | $0.03 | 10.5 |
| Haiku | explore | GPT-2 L1/N1316+ | NEEDS_REVISION | #1408 | 44 | 2.71M | 14.0k | $0.05 | 8.6 |
| Haiku | independent_review | Qwen L8/N3818+ | NEEDS_REVISION | #1412 | 38 | 2.39M | 12.2k | $0.04 | 7.5 |

Input tokens include cache reads, which make up about 97% of them.

| Driver | Completed and merged | Median turns | Median input tokens | Median output | Driver cost per experiment |
| --- | --- | --- | --- | --- | --- |
| Opus, baseline (n = 38, to an open PR, no `--steps`) | not recorded | 24 | 1.33M | 8.0k | ~$1.44 (rough; $0.84–33) |
| Sonnet (n = 4, landed) | 4/4 | 28 | 1.60M | 6.6k | $0.41 |
| Haiku (n = 4, landed) | 4/4 | 34 | 2.06M | 11.5k | $0.04 |

- **Reliability.** Both small drivers completed and merged 8 of 8 experiments with no intervention, across explore, independent_review and reevaluate draws. Every record carries the subagents' reported model IDs (`external/claude-opus-5-5` proposer, `external/claude-sonnet-5-5` reviewer). Every review has detection and fuzzing scores (none are expected for a reevaluate), and every record is signed. Each explore was claimed before its scan, with the key `--steps` worked out from the seed.
- **Cost.** Haiku took about 30% more turns and tokens than Sonnet, and still cost about a tenth as much to drive: $0.04 against $0.41 per experiment. Against the rough Opus baseline, that is about 35 times cheaper, and the pilots also did the build, CI wait and merge that the baseline sessions didn't.
- **Reading.** The skill files were barely read: at most one `cat` of a skill per session. In the baseline the median was 2 reads of skills, docs or JSON.

## Caveats

- n = 4 per arm, and the draws were easy ones: three GPT-2 explores (a 128-token-window scan takes a minute or two) and reviews of existing records. No arm ran a Qwen explore (a ~15-minute scan to wait on in the background), a `revise` or `new_hypothesis` (`--supersedes`), a scan that stops on a dead or default-on unit, a failing CI check or a merge conflict. Those are where a weaker driver is most likely to go wrong.
- The arms ran in parallel on one machine. Sonnet waited for its scans with `pgrep -f evidence-out`, which also matches the other arm's scan, so its wall times may be inflated slightly. Token counts are unaffected.
- Every session tried land-experiment's `SHA=$(gh pr view …) && gh pr merge …` and was refused by Claude Code's permission check ("quoted text in this command can't be checked"). Each driver recovered by pasting the SHA in. The skill now uses two separate commands.
- The three Qwen draws followed up records whose scans used the 552-token built-in sample, not `corpus.jsonl`. The follow-ups inherit those scans' max activation, so their verdicts are as weak as the originals'. This isn't a driver problem: 568 records on 483 units are affected, and it is in TODO under Pipeline.

## Second pilot: Haiku on harder draws

Run the same day, after the first pilot: one `--driver claude-haiku-5-5` loop of 8 experiments, set up as before. I didn't pick the draws. Re-drawing until the planner offers a wanted kind would bias the registry's random-selection statistics. All 8 draws were explores, 5 of them on Qwen.

| # | Unit | Verdict | PR | Turns | Input tokens | Output | Driver cost | Minutes | What happened |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GPT-2 L2/N2914− | REJECTED | #1419 | 41 | 2.64M | 16.9k | $0.054 | 9.0 | Ran `next` twice and claimed one draw's key with the other's seed; renamed the claim when the scan named the unit. Started a new reviewer for the token check instead of messaging the first. |
| 2 | Qwen L9/N416+ | NEEDS_REVISION | #1420 | 39 | 2.40M | 14.1k | $0.045 | 34.7 | Clean; waited out a ~25-minute scan |
| 3 | GPT-2 L1/N2187− | REJECTED | #1428 | 59 | 4.36M | 20.1k | $0.069 | 13.4 | First draw, L10/N965−, stopped as default-on (76% active). Committed its draw record and drew again, but left that PR (#1427) as a draft, so I merged it afterwards. |
| 4 | Qwen L23/N2950− | VERIFIED | #1430 | 45 | 2.87M | 14.1k | $0.050 | 11.4 | Tried `timeout` (not installed) in a watch, then recovered |
| 5 | Qwen L23/N2284+ | NEEDS_REVISION | #1433 | 70 | 4.69M | 21.2k | $0.084 | 31.9 | Clean; waited for CI checks to register |
| 6 | GPT-2 L7/N570+ | VERIFIED | #1438 | 42 | 2.63M | 14.4k | $0.045 | 8.0 | Clean (PR title not in the `experiment: <key> (<kind>)` form) |
| 7 | Qwen L6/N4564+ | VERIFIED | #1440 | 52 | 3.59M | 15.2k | $0.058 | 17.0 | Clean |
| 8 | Qwen L17/N2717+ | VERIFIED | #1443 | 43 | 2.58M | 14.8k | $0.045 | 9.9 | Landed, but the final message has no `RESULT` line, so the loop logged "no RESULT line" |

- **Completion.** 8 of 8 experiments landed. Every record has random selection with its seed, `corpus.jsonl` (~200k tokens), the subagents' reported IDs, both judge scores and a signature. One by-product needed a human: the default-on draw's PR, which held a draw record and should have merged.
- **Cost.** About $0.056 per experiment (median 44 turns, 2.76M input tokens). That is more than the first pilot's $0.04, because Qwen scans meant longer waits and more checks. It is still about 7 times cheaper than Sonnet's first-pilot $0.41, which came from easier draws.
- **Errors.** Haiku makes more slips than Sonnet did, but it noticed and recovered from each one itself except the draft PR. Fixes from this pilot:
  - `--steps` now says a stopped scan's PR must be landed, not left as a draft. It said "post it", which Haiku read as "comment on it".
  - `experiment-loop.sh` runs `uv sync` before each session. A fresh worktree's venv build had buried the first `openinterp next` output, and that is why Haiku ran it twice.
- **Still untested:** a `revise` or `new_hypothesis` draw (`--supersedes`), a failing CI check and a merge conflict. None came up in 16 draws.

## Conclusion

Over both pilots, Haiku drove 12 experiments and landed 12 of them, at $0.02–0.08 each. These covered Qwen explores with long scan waits and a scan stopped on a default-on unit. It needed one rescue, and that came from an unclear instruction that is now fixed. Its slips are mostly ones it catches itself. A Haiku driver is good enough to be the loop's default. The cases that remain (`--supersedes`, CI failures) are worth watching in the first unattended runs, but they don't need another pilot first: both are spelled out step by step in `--steps` and land-experiment.
