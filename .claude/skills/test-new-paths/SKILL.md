---
name: test-new-paths
description: Check that a code change tests every path it adds, before the PR is marked ready. List each new branch, early return and error in the diff. Find the test that reaches it, and check that the test asserts on the result. Then run `make coverage`, which also runs diff-cover. Use for every PR that changes Python under openinterpretability/ or scripts/.
---

# Test the new paths

Coverage says which lines ran. It doesn't say whether a test checked what they did. CI runs the deterministic half: branch coverage of the whole package, and diff-cover on the lines that a PR changes (#3166, #5855). This skill is the half that needs judgement.

## Procedure

1. **List the new paths.** Read the diff against `origin/main` (`git diff origin/main...HEAD -- '*.py'`). For each changed function, list the paths that the change adds:
   - each new `if`, `elif` and `else`.
   - each `except` and each early `return` or `raise`.
   - each loop that can run zero times.
   - each new default or optional argument.
2. **Find the test for each path.** Name the test that reaches the path. If no test reaches it, write one.
3. **Check the assertion.** The test must assert on what the path returns, writes or raises. A test that only runs the path doesn't count. For an error path, assert on the error's type and on its message.
4. **Check the repo's edge cases.** Each one that the change can meet needs a test:
   - an empty input, and an input with one item.
   - a dead pole and a default-on pole.
   - both poles of a neuron, positive and negative.
   - every kind of unit that the code accepts: a neuron pole, an attention head, a dictionary feature.
   - an old record that has no value for a new field. Records are write-once, so old records stay in the registry.
   - a bad input from the user, with the message that the user sees.
5. **Run the checks.** Run `make coverage`. It runs the suite with branch coverage, and fails below `fail_under` in `[tool.coverage.report]`. Then it runs diff-cover against `origin/main` at `fail_under` in `[tool.diff_cover]`, which is 100% (#5855). Every changed line must be tested, or marked as the next step says.
6. **Mark only what no test can reach.** A line that CI can't run gets `# pragma: no cover` and a comment with the reason. Examples are a guard for a device that CI lacks (CUDA, MPS, the Neural Engine) and a real network download. Never mark code that a test can reach, however hard. Mock the boundary, not the code under test.
7. **Mutation testing is planned.** #5763 would add `make mutation` for the verdict code. Until then, read the tests of any verdict code that the change touches, and ask what change to the code they would miss.

## Notes

- **100%, honestly.** The target is 100% of the package (#5855). A test that only runs a line, with no assertion, doesn't count. A line that no test can reach is marked, with its reason, so a reader sees why.
- **Raise the floor.** When the total rises, raise `[tool.coverage.report] fail_under` to just below it. Never lower it.
- **A test reaches a path through behaviour.** Prefer a test that calls the public function with real inputs over one that mocks the code under test.
