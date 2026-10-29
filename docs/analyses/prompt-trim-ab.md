# Trimmed agent prompts: A/B against the full prompts

Run on 2026-10-29, by hand, before making the windowed prompts the default (TODO "Performance", first item). The arms' files stayed in `exchange/ab/`, which is gitignored, so they aren't in the repository.

## Method

- **Units.** Eight neuron poles were drawn with `random.Random(20261029)` from the scan evidence already in `exchange/`. Five are Qwen2.5-0.5B poles (L7/N4292−, L22/N1436−, L14/N4395+, L18/N1049+, L22/N2686+) and three are GPT-2 small poles (L3/N1517+, L11/N830+, L9/N330+). Every unit was reviewed on the same `ev.json` in both arms.
- **Arms.**
  - *Full* is the prompt as it was before trimming, which `--full-snippets` reproduces byte for byte (checked on 232 prompts).
  - *Trim* is the new default. It shows a 48-before/16-after token window around each peak and drops near-duplicate examples, and the reviewer sees the peak marked.
  - Each arm changes both the proposer's and the reviewer's prompt.
- **Blinding.** Each unit's arms were labelled `a`/`b` at random, so a subagent couldn't tell which arm it was in. Every subagent was given only its own prompt path.
- **Agents.** Proposers were `neuron-proposer` subagents (claude-opus-5-5) and reviewers were `neuron-reviewer` subagents (claude-sonnet-5-5), so every review is independent of its proposer. No token check was run, in either arm.
- **Verdict.** `openinterp audit --evidence … --hypothesis …`, then `ReviewerAgent.review` on the probes, under the default policy. Explanation scores were skipped in both arms. They are reported, not gated, so they can't change a verdict.
- **Caveats.**
  - The first proposers shared one scratch directory and overwrote each other's scripts. Every affected agent reported that it re-ran its measurements from a private directory before answering.
  - Nine proposers and one reviewer were stopped by an API session limit and then resumed. Their token counts therefore cover two transcript segments.

## Verdicts

| Unit | Full | Trim | Balanced accuracy (full → trim) |
| --- | --- | --- | --- |
| Qwen L7/N4292− | NEEDS_REVISION (sensitivity, confound baseline) | VERIFIED | 0.70 → 0.71 |
| Qwen L22/N1436− | REJECTED | NEEDS_REVISION (sensitivity, confound baseline) | 0.51 → 0.61 |
| Qwen L14/N4395+ | VERIFIED | VERIFIED | 0.95 → 0.95 |
| Qwen L18/N1049+ | NEEDS_REVISION (lookahead) | REJECTED | 0.81 → 0.48 |
| Qwen L22/N2686+ | VERIFIED | NEEDS_REVISION (specificity) | 0.85 → 0.86 |
| GPT-2 L3/N1517+ | NEEDS_REVISION (specificity) | NEEDS_REVISION (specificity) | 0.77 → 0.73 |
| GPT-2 L11/N830+ | VERIFIED | VERIFIED | 0.91 → 0.95 |
| GPT-2 L9/N330+ | NEEDS_REVISION (specificity, confound baseline) | REJECTED (specificity) | 0.60 → 0.59 |
| **Totals** | 3 VERIFIED, 4 NEEDS_REVISION, 1 REJECTED | 3 VERIFIED, 3 NEEDS_REVISION, 2 REJECTED | median change +0.005 |

- **Verdict rates.** Both arms verified 3 of 8 units. Only three units got the same verdict in both arms. Four of the five that differ sit within 0.04 balanced accuracy of each other on either side of a gate, and they go both ways (two up, two down). That pattern looks like run-to-run noise, not a shift, but there was no same-arm replicate to measure that noise directly.
- **The one large difference is L18/N1049+.** Both proposers wrote the same hypothesis: ` =` after a numeric-sounding variable name. The trim reviewer, which saw the peaks marked, turned the top examples the hypothesis leaves out into hard negatives: the `return` line at 3.84 and the traceback indent at 3.06. The hypothesis does fail on those, and the trim arm's own proposer had listed them as gaps. One of that reviewer's probes was also probably mislabelled, a case its reviewer flagged itself, which the token check would normally catch.
- **What that means.** Marking the peak for the reviewer can make reviews stricter. That is the intended direction for an adversarial reviewer, but it means verdict rates may drift down somewhat.

## Cost

| | Full (median) | Trim (median) |
| --- | --- | --- |
| Proposer task text (characters) | 49,590 | 12,215 |
| Reviewer prompt (characters) | 22,746 | 6,494 |
| Proposer input tokens over the whole run (including cache reads) | 420,747 | 224,895 |
| Reviewer input tokens | 46,147 | 33,082 |

- **Proposer tokens.** Proposer input tokens fall by about 47% on the Qwen units. They don't change on GPT-2 L3 and L11, whose scanned documents are short anyway.
- **Requests.** The number of proposer requests is about the same in both arms (8–15). Proposers measure their own counterfactuals, so the conversation length is set by that work, not by the prompt.

## Run-to-run variance

To tell a prompt effect from noise, the trimmed arm was run a second time on the same eight units. Each unit got a fresh proposer and a fresh reviewer, with the same models and the same `ev.json`. This time every agent had a private scratch directory from the start, and none hit the API limit.

| Unit | Full | Trim, run 1 | Trim, run 2 |
| --- | --- | --- | --- |
| Qwen L7/N4292− | NEEDS_REVISION 0.70 | VERIFIED 0.71 | VERIFIED 0.90 |
| Qwen L22/N1436− | REJECTED 0.51 | NEEDS_REVISION 0.61 | VERIFIED 0.80 |
| Qwen L14/N4395+ | VERIFIED 0.95 | VERIFIED 0.95 | VERIFIED 0.90 |
| Qwen L18/N1049+ | NEEDS_REVISION 0.81 | REJECTED 0.48 | NEEDS_REVISION 0.82 |
| Qwen L22/N2686+ | VERIFIED 0.85 | NEEDS_REVISION 0.86 | VERIFIED 0.91 |
| GPT-2 L3/N1517+ | NEEDS_REVISION 0.77 | NEEDS_REVISION 0.73 | VERIFIED 0.91 |
| GPT-2 L11/N830+ | VERIFIED 0.91 | VERIFIED 0.95 | NEEDS_REVISION 0.92 (lookahead) |
| GPT-2 L9/N330+ | NEEDS_REVISION 0.60 | REJECTED 0.59 | REJECTED 0.59 |

| Comparison | Same verdict | Mean \|Δ balanced accuracy\| | SD of Δ |
| --- | --- | --- | --- |
| Trim run 1 vs trim run 2 (same prompt) | 3/8 | 0.130 | 0.139 |
| Full vs trim run 1 | 3/8 | 0.069 | 0.130 |
| Full vs trim run 2 | 3/8 | 0.095 | 0.118 |

- **The A/B difference is within noise.** Running the same prompt twice disagreed as often (5 of 8 units) and by as much as switching prompts did. The L18/N1049+ gap from the A/B closed on the rerun (0.82, NEEDS_REVISION like the full arm). The full arm verified 3 units, trim run 1 verified 3 and trim run 2 verified 5.
- **Where the noise comes from.** It comes mostly from what each proposer chooses to claim, then from how hard each reviewer pushes. For example, the second L11/N830+ proposer called the unit lookahead, where both earlier ones called it retrospective, and that alone failed the lookahead gate.
- **Size of a future prompt A/B.** With a per-unit SD of about 0.14 in paired Δ balanced accuracy, 80% power at α = 0.05 needs about 16 units to detect a 0.10 shift and about 61 units to detect 0.05. Eight units can only rule out very large effects.
- **For the registry.** A single experiment's verdict reproduced on only 3 of 8 units when the whole pipeline was rerun. Independent reviews and replications therefore matter more than any one verdict.
