# Literature review queue

Papers and posts waiting for a page are [GitHub issues with the label `area: literature`](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20literature%22). Each is in the Linear project *Literature review*, with a priority. Until 2026-11-13 this file listed them.

To add a paper, open an issue: `gh issue create --title "Review <title>" --label "area: literature" --label "priority: low"`. Give the citation and link in the body. Raise the priority when the paper bears on a gate or a score. The PR that adds the paper's page in `docs/literature/` closes the issue.

Each issue's body has a tag line. `**Bears on:**` names the code, skill, analysis or protocol section that the work would change. `**Off-topic:**` marks work that gets a short *not applicable* page. To write a page, use the `literature-review` skill (`.claude/skills/literature-review/SKILL.md`).
