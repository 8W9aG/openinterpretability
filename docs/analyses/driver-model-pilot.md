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

## Conclusion

A Haiku driver lands easy draws as reliably as Sonnet, at about a tenth of Sonnet's driver cost. Before making it the loop's default, run a second pilot that covers the cases above.
