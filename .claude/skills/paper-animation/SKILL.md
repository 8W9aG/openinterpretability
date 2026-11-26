---
name: paper-animation
description: Make a paper figure move, from measured data, in the repo's SVG pipeline. Choose the form (change over training, reading along tokens, before and after an intervention, two states, build-up, steering). Build the keyframes with scripts/figure_motion.py, and check the result. Use when writing or revising a paper (write-paper calls it), or when a figure's argument is about change, order or cause.
---

# Paper animation

A figure moves when its argument is about **change, order or cause**. The motion shows the reader what a static chart makes them work out. Every frame comes from the paper's `results.jsonl`, so the motion never shows a value that was not measured.

## Choose the form

[`project.md`](project.md) gives a paper of this project for each form (#6631).

| Pattern | When | Example |
| --- | --- | --- |
| **Change through training** | Points or curves change across checkpoints or steps | Each point steps through the checkpoints beside the loss. |
| **Reading along a sequence** | A unit's activation token by token | A caret moves across the tokens and lights each one at its activation. |
| **Before and after an intervention** | An ablation, a clamp or a patch moves a distribution or the logits | Morph from the clean state to the intervened one, and label both. |
| **Two states compared** | Two conditions of one measure | Each curve morphs between the two conditions. |
| **Build-up** | The argument assembles from parts | The arms appear one at a time, or each group's points appear, then its median. |
| **Steering** | Generated text changes as a clamp rises | Show the text token by token while a counter names the clamp value. |

Don't animate a chart that has no order, such as a bar chart of categories. Don't spin or pulse a figure, and don't add motion for decoration.

## Make it

Build every animation with `scripts/figure_motion.py`, so all figures share one form:

- `animate(name, stops, cycle, timing=…, hold=…, rest=…)` writes a `@keyframes` rule and the class that runs it. Give `rest` the final frame.
- `translate(x, y)` moves a mark that you draw at the origin, and `even_stops(frames, until)` spreads frames evenly up to `until` percent.
- `reveal(name, at, cycle)` shows a mark from `at` percent of the cycle to the end.
- `counter(name, starts, cycle)` gives one class per label, such as "step 1,000", each shown during its frame.

The rules:

- **CSS keyframes inside the SVG only.** Use no JavaScript and no SMIL, so the figure moves on GitHub, on the site and in a plain `<img>`.
- **A short loop.** Make the cycle 4 to 10 seconds. A figure that reads along many tokens may take up to 15. Hold the final frame for the last 15 to 20% of the cycle.
- **Show where the loop is.** Add a step counter or a progress bar when the frames are steps.
- **Measured frames only.** For measured checkpoints, use `step-end` timing, so a mark jumps between measured values and never stops between them. Use `linear` timing only where the curve between points is itself drawn. Use `ease-in-out` for a morph between two measured states. If only the two ends are measured, say so in the subtitle.
- **Reduced motion.** The base CSS of `figure_svg.py` stops every animation under `prefers-reduced-motion`. Each mark then shows its resting state, so set `rest` to the final frame. The still figure must make the point alone.
- **No colour that is new to the paper.** A moving "now" marker can use the ink colour (`ink ring`). Motion must not carry the point by colour alone.

## Check it

- `uv run python scripts/paper_figures.py`, then `uv run pytest tests/test_paper_figures.py`. The tests check each animated SVG for the reduced-motion rule, a cycle of 4 to 15 seconds, a resting state and at most 200 KB.
- Render a mid-cycle frame and the final frame, in light mode and in dark mode, and look at them. Headless Chrome can do it. Put the SVG in an HTML page with `*{animation-play-state:paused!important;animation-delay:-<seconds>s!important}`, and add `--force-dark-mode` for dark mode. Attach the frames to the PR.
- The maintainer is colour blind, so no point may rest on colour alone. They trust the author's judgement on graphics: check the frames yourself, then merge on green CI as for any PR.
