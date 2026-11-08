# Code review instructions

Guidance for anyone reviewing a change to this repo, human or agent (Claude Code's code review reads this file).

## Duplicated definitions

Load the `single-source-of-truth` skill (`.claude/skills/single-source-of-truth/SKILL.md`) and apply its "Reviewing a diff" checklist. In short, flag:

- a new literal (number, status, model name, path, class name) that equals an existing constant, or a new list that repeats an existing one.
- a template map keyed by values that already exist in Python.
- a number in docs, a skill or a workflow that quotes code with no test pinning it and no comment naming its source.
- a changed source whose copies were not updated.

Name the constant, global or test to use instead.

## Comments and docstrings

Load the `commenting` skill (`.claude/skills/commenting/SKILL.md`) and apply its "Reviewing a diff" checklist. In short, flag:

- a new public definition with no docstring, or a docstring that only restates the name.
- a comment that restates the code, or that a rename or an extracted function could replace.
- a value, threshold or rule with no source (a `docs/PROTOCOL.md` section, a paper, a page).
- a sentence of more than 25 words, a word from `.vale/styles/STE/Words.yml`, or two meanings for one word.
- a comment that the change made wrong.
