# Part of speech in the scan

Run on 2026-11-14 for issue #3223. Every scan now records a `part of speech` label set: each word's universal tag (UPOS), from the same spaCy pass that finds names (`openinterp/names.py`). Reported, not gated.

The question: how much does part of speech explain of a pole's activity? And does it pick out poles whose explanations are about syntax?

## Method

- **Units.** Six VERIFIED Qwen2.5-0.5B poles, all judged under protocol 0.8. Three have explanations about syntax, and three about particular words:

  | Pole | Kind | Explanation, in short |
  | --- | --- | --- |
  | L7/N4292− | syntax | a preposition of means or manner after a verb phrase |
  | L17/N1422+ | syntax | a transitive verb whose object came before it |
  | L6/N3151+ | syntax | the verb that puts something forward for others to accept |
  | L0/N788+ | word | the word-initial piece "mc" |
  | L4/N1889+ | word | "c" as the name of the C language |
  | L3/N3738− | word | "official", as an adjective or adverb |

- **Scan.** `openinterp audit --evidence-out` on `corpus.jsonl`, 200,000 tokens, with no scan cache, so each scan tagged its tokens. The six scans and the tagging of the corpus took 8 minutes in all.
- **Measures.** The label set's R² and mutual information, as for every label set. Then, for the three tags with the most active tokens, the tag's share of active tokens over its share of all tokens. A ratio above 1 means that the pole is active on that tag more often than the tag's frequency predicts.

## Result

| Pole | Token R² | Part-of-speech R² | Bits | Active tokens | Most active tags (active, ratio) |
| --- | --- | --- | --- | --- | --- |
| L7/N4292− | 0.23 | 0.014 | 0.028 | 133 | ADP 36 (3.6×), NOUN 30 (1.1×), DET 17 (2.1×) |
| L17/N1422+ | 0.25 | 0.028 | 0.074 | 370 | VERB 247 (9.0×), AUX 45 (3.5×), X 17 (0.8×) |
| L6/N3151+ | 0.42 | 0.012 | 0.023 | 116 | VERB 85 (9.9×), ADP 18 (2.1×), NOUN 5 (0.2×) |
| L0/N788+ | 0.96 | 0.044 | 0.148 | 419 | PROPN 218 (5.0×), NOUN 133 (1.5×), ADJ 28 (1.1×) |
| L4/N1889+ | 0.34 | 0.007 | 0.031 | 20 | NOUN 9 (2.1×), PROPN 4 (1.9×), VERB 3 (2.0×) |
| L3/N3738− | 0.65 | 0.009 | 0.041 | 25 | ADJ 14 (9.2×), ADV 4 (6.8×), X 3 (2.0×) |

- **R² and mutual information don't tell the kinds apart.** Part of speech explains at most 4% of any pole's variance, far below token identity (23–96%). A pole is inactive on almost every token, so most of its variance is in tokens that carry every tag. By R², part of speech is not a confound for any of the six.
- **The active tokens do tell them apart.** The two verb poles fire on verbs at 9–10 times the verbs' share of the corpus: 67% and 73% of their active tokens are verbs. The preposition pole fires on prepositions (ADP) at 3.6 times their share. So the tags show the syntactic claim directly.
- **They also check word-level claims.** "Official" as an adjective or adverb shows as ADJ at 9.2 times and ADV at 6.8 times, which is the explanation's own split. The "mc" pole fires mostly on proper nouns (5.0 times), as a name piece such as *McDonald* would be.

## What this means

- **The proposer gets the useful part already.** Its prompt lists each label set's active tokens per tag, as active / count. So the verb poles' proposers would have seen *VERB 247/…* next to R² 3%. They don't need a new number.
- **Don't gate on part-of-speech R².** It is near zero for poles whose explanations name a part of speech. A gate would need the enrichment of active tokens, or the precision of "fires on every verb", which is a different measure.
- **A rule "active on every verb" would score badly.** The verb poles are active on 247 and 85 of about 15,000 verbs. So part of speech names their class but not their rule. A hypothesis still has to say which verbs.

## Limits

- **Six poles.** The study shows what the tags look like for clear cases. It is not a rate.
- **English tags.** The tagger is an English model. On code and on other languages its tags are less reliable, and the active-token shares carry that error.
- **Words, not tokens.** A tag covers its whole word, so every piece of a split word gets the word's tag. For a pole on a word-initial piece, such as "mc", the tag is the whole word's.

## Reproduce

```bash
uv run openinterp audit --model Qwen/Qwen2.5-0.5B --layer 7 --neuron 4292 --pole negative \
    --corpus corpus.jsonl --max-tokens 200000 --no-scan-cache --wiki-dir '' --evidence-out ev.json
# the label set is in ev.json under confounds.labels["part of speech"]
```
