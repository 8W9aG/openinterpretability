# Code review instructions

Guidance for anyone reviewing a change to this repo, human or agent (Claude Code's code review reads this file).

## Duplicated definitions

Load the `single-source-of-truth` skill (`.claude/skills/single-source-of-truth/SKILL.md`) and apply its "Reviewing a diff" checklist. In short, flag:

- a new literal (number, status, model name, path, class name) that equals an existing constant, or a new list that repeats an existing one;
- a template map keyed by values that already exist in Python;
- a number in docs, a skill or a workflow that quotes code with no test pinning it and no comment naming its source;
- a changed source whose copies were not updated.

Name the constant, global or test to use instead.
