# Prompt compression with LLMLingua-2

Run on 2026-11-01 (TODO "Performance", first item), with `scripts/prompt_compression.py`. LLMLingua 0.2.2, model `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, on CPU. Per-paragraph keep scores and per-text compression results are in `prompt-compression.jsonl`.

**Conclusion: not adopted.** Compressing the agents' prose would save about 1k tokens per proposer request, mostly cached tokens. At that saving it already drops the paths, file names and commands the agents are told to use. No A/B was run, since it is only needed before adopting.

## What could be compressed

The evidence is never a candidate: the token windows and activations are what the proposer explains and the reviewer tests. Since the snippet trim (`prompt-trim-ab.md`), the evidence part of a proposer prompt is about 8–10 KB. What each agent reads besides it is fixed prose:

| Agent | Prose read | Characters |
| --- | --- | --- |
| Proposer | system prompt, `neuron-proposer` definition, skills `native-neuron-audit`, `vocab-projection-lookahead`, `neuron-family-triage` | 19,408 |
| Reviewer | system prompt, `neuron-reviewer` definition, skill `interpretability-illusions-review` | 8,091 |
| Judge | system prompt, `explanation-judge` definition | 1,642 |

The proposer's prose is therefore about two-thirds of what it reads before its first tool call. The JSON answer schema at the end of each prompt is also fixed, but it must stay valid JSON, so it isn't a candidate either.

## Results

LLMLingua-2's `rate` is the share of tokens kept. Characters and *literals* are totals over each agent's texts. A literal is inline code, a number, or a path (the regex `LITERAL` in the script), something an agent acts on verbatim.

| Agent | Rate | Characters | Literals kept |
| --- | --- | --- | --- |
| Proposer | 0.7 | 19,408 → 14,889 (−23%) | 164 / 177 |
| Proposer | 0.5 | → 11,172 (−42%) | 113 / 177 |
| Proposer | 0.33 | → 7,506 (−61%) | 91 / 177 |
| Reviewer | 0.7 | 8,091 → 6,278 (−22%) | 52 / 55 |
| Reviewer | 0.5 | → 4,725 (−42%) | 37 / 55 |
| Judge | 0.7 | 1,642 → 1,255 (−24%) | 7 / 8 |
| Judge | 0.5 | → 940 (−43%) | 2 / 8 |

Even the mildest rate (0.7) dropped literals that change what an agent does:

- **Proposer definition:** two of the three skill paths it is told to read (`.claude/skills/vocab-projection-lookahead/SKILL.md`, `.claude/skills/neuron-family-triage/SKILL.md`), and the answer file name `<name>.hypothesis.json`.
- **Skills:** the `openinterp activations … --corpus corpus.jsonl --top 0 --format jsonl` command line, `measure_lookahead(runner, prompts, layer, idx, pole)`, `W_U · w_out` (twice), and `provenance.selection = "manual"`.
- **Judge definition:** `*.task.json`, the file it must not read.

It also removes the markdown structure (`#` headings, list markers, `---` frontmatter fences). A skill file without its frontmatter isn't a skill Claude Code can load. So the compressed text could only be a second copy next to the real skill, kept in sync by hand, against the `single-source-of-truth` rule.

## Saving

The proposer's prose is about 4.9k tokens (at about 4 characters per token). Rate 0.7 removes about 1.1k of them, and rate 0.5 about 2k. That prose sits in the context of every later request, so over a proposer's 8–15 requests rate 0.7 saves roughly 14k of its median 225k input tokens (about 6%). Nearly all of those are cache reads, billed at a tenth of the input price, so the cost saving is around 1%. The run's length is set by the proposer's own measurements, not by the prose (`prompt-trim-ab.md`, Cost).

## What the keep scores say

The script's second use, as the TODO suggested, is a guide for trimming by hand. Weighted by length, mean keep probability is 0.86 on code blocks and 0.61 on prose. By text, it ranges from 0.49 (judge system prompt) to 0.70 (`neuron-family-triage`). The lowest-scoring paragraphs (0.48–0.50) are the agent definitions' role statements and reply instructions ("Reply with the answer's path…", "Read only the prompt file you were given…", "The protocol forbids you from seeing probe activations…").

In each of them, the eight lowest words are articles and auxiliaries (`the`, `an`, `that`, `were`). So low scores mark ordinary English grammar, not content that can go. The model was trained on meeting transcripts, where such words are filler. No paragraph scored low because its content was redundant, so the scores don't point at anything to trim.

## Per-token relevance in the evidence

LLMLingua's keep scores are not added to the evidence. They measure how much a small, different model would miss a token, not anything about the neuron under study.
