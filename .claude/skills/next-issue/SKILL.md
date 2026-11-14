---
name: next-issue
description: Pick the next issue to work and start it. It takes the next release's highest-priority unstarted issue. When the release has none left, it pulls one in from Linear. Then it hands over to work-an-issue. Use when the user says "do the next issue" or "what's next", or asks you to keep working through the issues. Also use it when you finish an issue and should continue.
---

# The next issue

`scripts/next_issue.py` makes the choice, so that every agent chooses the same way. Run it, check its pick, and then work the issue with `work-an-issue`.

## Procedure

1. **Ask the script.** It needs `gh` and `LINEAR_API_KEY`:
   ```bash
   python3 scripts/next_issue.py
   ```
   It prints the pick and up to four runners-up. It chooses in this order:
   - **The release first.** The next release is the oldest open milestone. Its open issues are candidates when their Linear state is not In Progress or In Review, they have no `proposed` label, and `issue_state.py --check` says they are eligible.
   - **Then the backlog.** If the release has no candidate, every open issue with no milestone is a candidate, by the same rules. `area: models` is held back: no new model until every configured model reaches `coverage_target`.
   - **Ranking.** Candidates rank by priority, which comes from Linear, or from the `priority:` label when Linear has none. Ties go to the area that the release has served least, then to the oldest issue. Each place in the ranking counts against its area, so the runners-up spread across workstreams.
2. **Check the pick before you take it.** Skip it and take the next runner-up if any of these holds. Say why in one line.
   - **It needs the maintainer's decision.** One example is a protocol change that offers options with no choice made. Comment on the issue with the question, and leave it.
   - **It depends on an open issue or PR.** Name the blocker in a comment.
   - **It can't run here.** For example, it needs a GPU this machine doesn't have.
3. **Pull it in if it came from the backlog.** The script says "pulled into". Then run it again with `--pull`. That adds the milestone and the `release: <version>` label:
   ```bash
   python3 scripts/next_issue.py --pull
   ```
   Pull one issue at a time, and only when the release has nothing left. The release cadence (`release` skill, *When to release*) decides when the release goes out, not how full it is.
4. **Work it with `work-an-issue`.** That skill marks the issue In Progress, posts the plan, posts deviations, opens the PR and posts what shipped. Its eligibility check runs again there.
5. **Continue.** When the issue is In Review, go back to step 1, unless the user asked for one issue. Don't wait for a PR to merge before you start the next issue. Do resync a PR if `main` moves under it and it conflicts.

## Notes

- **An empty pick means nothing is eligible.** The script then exits with "No eligible issue". Say so, and propose new work as the `release` skill's step 3 does. Don't work on an ineligible issue.
- **Linear's states are the record of who is working on what.** An issue In Progress belongs to whoever started it. Never take it, even if it looks stalled. Ask instead.
- **The script only reads, except with `--pull`.** `--pull` sets one milestone and one label. `work-an-issue` sets the states.
