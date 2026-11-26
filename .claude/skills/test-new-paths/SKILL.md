---
name: test-new-paths
description: Check that a code change tests every path it adds, before the PR is marked ready. List each new branch, early return and error in the diff. Find the test that reaches it, and check that the test asserts on the result. Then run `make coverage`, which also runs diff-cover, and `make mutation BASE=origin/main` on the changed package files. Use for every PR that changes Python in a package or in scripts/.
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
4. **Check the edge cases.** Each one that the change can meet needs a test:
   - an empty input, and an input with one item.
   - an old record that has no value for a new field, when records are write-once.
   - a bad input from the user, with the message that the user sees.
   - the project's own cases, which [`project.md`](project.md) lists (#6631).
5. **Run the checks.** Run `make coverage`. It runs the suite with branch coverage, and fails below `fail_under` in `[tool.coverage.report]`. Then it runs diff-cover against `origin/main` at `fail_under` in `[tool.diff_cover]`, which is 100% (#5855). Every changed line must be tested, or marked as the next step says.
6. **Mark only what no test can reach.** A line that CI can't run gets `# pragma: no cover` and a comment with the reason. Examples are a guard for a device that CI lacks (CUDA, MPS, the Neural Engine) and a real network download. Never mark code that a test can reach, however hard. Mock the boundary, not the code under test.
7. **Run mutation testing on the changed code.** After `make coverage`, run `make mutation BASE=origin/main` (#5856). It mutates each package file (`[tool.mutmut] source_paths`) that the branch changes, and runs the tests that import those files. It runs only the mutants in the functions that the branch changes, at most 100 of them, a seeded sample (#5857). It fails when a mutant survives or no test reaches it. A run takes from about 30 seconds to a few minutes. For each mutant that it lists:
   - **Kill it.** Add a test that fails on the mutant and passes on the code. See the diff with `uv run python scripts/mutation.py --base origin/main show <name>`.
   - **Or record it as equivalent.** If the mutant changes nothing that a caller can see, put mutmut's pragma on its line, with the reason: `# pragma: no mutate, equivalent: <why>`. mutmut then makes no mutants on that line, so later runs don't report it. "Hard to test" is not a reason.

   Run it again until it passes. A surviving mutant in a changed verdict file (`only_mutate` in `[tool.mutmut]`) blocks the PR: kill it or record it before you mark the PR ready (#6359). `make mutation` with no `BASE` mutates that fixed verdict code (#5763), in about 2 minutes, and its tests kill every mutant there. `make mutation RUNS=5` runs it 5 times, each in its own test order. It fails when a mutant's result changes between runs, which means that a test depends on order or state. Treat that as a flaky test (`next-issue`, *Flaky tests*).

## Notes

- **100%, honestly.** The target is 100% of the package (#5855). A test that only runs a line, with no assertion, doesn't count. A line that no test can reach is marked, with its reason, so a reader sees why.
- **Raise the floor.** When the total rises, raise `[tool.coverage.report] fail_under` to just below it. Never lower it.
- **A test reaches a path through behaviour.** Prefer a test that calls the public function with real inputs over one that mocks the code under test.
