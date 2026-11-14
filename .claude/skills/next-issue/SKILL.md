---
name: next-issue
description: Give unprioritised issues a priority, then pick the next issue to work and start it. It takes the next release's highest-priority unstarted issue. When the release has none left, it pulls one in from Linear. Then it hands over to work-an-issue. Use when the user says "do the next issue" or "what's next". When the last release is over 24 hours old, it cuts a release in place of an issue. It works one issue, waits for CI, merges the PR, and stops. To work through several, run it under `/loop`.
---

# The next issue

`scripts/next_issue.py` makes the choice, so that every agent chooses the same way. Run it, check its pick, and then work the issue with `work-an-issue`.

## Procedure

0. **Release first, if one is due.** Once 24 hours have passed since the last release was published, a release takes the place of the next issue. The maintainer decided this on 2026-11-14:
   ```bash
   gh release view --json tagName,publishedAt --jq '"\(.tagName) \(.publishedAt)"'
   date -u +%Y-%m-%dT%H:%M:%SZ
   ```
   If the last release is more than 24 hours old, run the `release` skill as this iteration's work, in place of an issue. It has its own checks and steps. Its release PR lands as in step 6, and then the iteration stops at step 7. Otherwise go on to step 1.
1. **Give every open issue a priority.** The ranking is only as good as the priorities, so first list the issues that have none:
   ```bash
   python3 scripts/next_issue.py --unprioritised
   ```
   Read each one, then set its priority. This sets the `priority:` label and Linear's priority:
   ```bash
   python3 scripts/next_issue.py --prioritise <number> <urgent|high|medium|low>
   ```
   Judge by the north star, total interpretability of open models (`docs/research-questions.md`, question 0):
   - **urgent:** it breaks the registry, CI, a release, or the correctness of stored records.
   - **high:** it changes whether verdicts can be trusted (protocol validity, false positives), or it removes a bottleneck to more VERIFIED units.
   - **medium:** a useful analysis, feature or paper review that doesn't block anything. This is the default when you are unsure.
   - **low:** nice to have, and every `area: models` issue until each configured model reaches `coverage_target`.

   Keep the workstreams balanced: don't raise one area's issues just because you know it best. Give the reason for each priority in one line when you report back. Setting a priority isn't working on an issue, so do it for ineligible issues too.
2. **Ask the script.** It needs `gh` and `LINEAR_API_KEY`:
   ```bash
   python3 scripts/next_issue.py
   ```
   It prints the pick and up to four runners-up. It chooses in this order:
   - **The release first.** The next release is the oldest open milestone. Its open issues are candidates when their Linear state is not In Progress or In Review, they have no `proposed` label, and `issue_state.py --check` says they are eligible.
   - **Then the backlog.** If the release has no candidate, every open issue with no milestone is a candidate, by the same rules. `area: models` is held back: no new model until every configured model reaches `coverage_target`.
   - **Ranking.** Candidates rank by priority, which comes from Linear, or from the `priority:` label when Linear has none. Ties go to the area that the release has served least, then to the oldest issue. Each place in the ranking counts against its area, so the runners-up spread across workstreams.
3. **Check the pick before you take it.** Skip it and take the next runner-up if any of these holds. Say why in one line.
   - **It needs the maintainer's decision.** One example is a protocol change that offers options with no choice made. Comment on the issue with the question, and leave it.
   - **It depends on an open issue or PR.** Name the blocker in a comment.
   - **It can't run here.** For example, it needs a GPU this machine doesn't have.
4. **Pull it in if it came from the backlog.** The script says "pulled into". Then run it again with `--pull`. That adds the milestone and the `release: <version>` label:
   ```bash
   python3 scripts/next_issue.py --pull
   ```
   Pull one issue at a time, and only when the release has nothing left. The release cadence (`release` skill, *When to release*) decides when the release goes out, not how full it is.
5. **Work it with `work-an-issue`.** That skill marks the issue In Progress, posts the plan, posts deviations, opens the PR and posts what shipped. Its eligibility check runs again there.
6. **Land the PR.** Don't move on while the issue's PR is open. The maintainer decided on 2026-11-14 that this skill merges its own PR once CI passes.
   1. Bring the branch up to date with `main`: `git fetch origin && git merge origin/main`. If that conflicts, don't resolve someone else's change by hand. Run `git merge --abort`, comment on the PR, and stop at step 7 without merging. Otherwise run `make lint` and `make test`, then push.
   2. Wait for every check: `gh pr checks <number> --watch --fail-fast --interval 30`. Checks can take a minute to appear after a push. No checks is not green, so wait and retry. If they are still pending after 45 minutes, comment that they didn't finish, and stop at step 7.
   3. If a check fails, read its log (`gh run view <run-id> --log-failed`). Fix it if this PR caused it, push, and wait again. Stop after two failed attempts, or if the failure isn't this PR's (a flaky runner, a broken `main`). Then comment what you found and stop at step 7 without merging.
   4. Merge only a PR that this session opened for this issue. It must not be draft, every check must have passed on its head commit, and `gh pr view <number> --json mergeable -q .mergeable` must print `MERGEABLE`:
      ```bash
      gh pr view <number> --json headRefOid -q .headRefOid     # the head commit
      gh pr merge <number> --squash --match-head-commit <that SHA>
      git push origin :<branch>                                 # the remote branch
      ```
      `--match-head-commit` refuses the merge if someone pushed after the checks ran. Never pass `--admin`, and never get round a failing or pending check another way. Drop the remote branch with the push above, not with `--delete-branch`, which tries to switch the local checkout.
   5. A merged PR whose body says `Closes #<number>` closes the issue. Post one line on the issue that it merged, with the merge commit.
   6. Return the checkout to `main` and pull it, so the next issue starts from the merged state.
7. **Stop.** Report the issue, its PR, what shipped and whether it merged, then stop. One run works one issue. To work through several, run this skill under `/loop`. Each iteration then picks afresh, so it sees what other agents started in the meantime. A PR left unmerged in step 6 stays open for the maintainer. Say why in the report.

## Notes

- **An empty pick means nothing is eligible.** The script then exits with "No eligible issue". Say so, and propose new work as the `release` skill's step 3 does. Don't work on an ineligible issue.
- **Linear's states are the record of who is working on what.** An issue In Progress belongs to whoever started it. Never take it, even if it looks stalled. Ask instead.
- **The script only reads, except with `--pull` and `--prioritise`.** `--pull` sets one milestone and one label. `--prioritise` sets one label and Linear's priority. `work-an-issue` sets the states.
