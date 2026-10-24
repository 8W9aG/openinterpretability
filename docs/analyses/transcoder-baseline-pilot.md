# Transcoder baseline: pilot on GPT-2 small

The first run of the learned-basis baseline (`TODO.md`, *Scope expansion*): 10 random latents of Dunefsky et al.'s GPT-2 small transcoders (`gpt2-small-transcoders`) against 10 random GPT-2 small neuron poles, audited under protocol 0.7. The home page's *Learned-basis baseline* table is computed from the same records (`openinterp/builder/learned_baseline.py`). Every draw, live or dead, is in `transcoder-baseline-pilot.jsonl`.

## Setup

- **Draws.** `openinterp audit --random --seed s` with seeds 1, 2, 3, … in order for each arm (`--dictionary gpt2-small-transcoders` for latents, `--model openai-community/gpt2` for neurons), until each arm had 10 live units. No draw was skipped or repeated.
- **Scan.** `corpus.jsonl` (sha256 `ded48cfc…`), the default 200,000-token sample, 128-token windows with stride 64. The transcoders fit no longer context (`harness/dictionaries.py`), and both arms were scanned the same way.
- **Agents.** Every unit had the same roles: proposer `claude-opus-5-5`, reviewer `claude-sonnet-5-5`, and judges `claude-haiku-5-5`. All reviews are cross-model.
- **Probe labels.** Before each review, the probe labels were checked against the submitted hypothesis and the 0.25 × max activity line. When a label contradicted the hypothesis, it went back to the same reviewer, which rewrote it (see *Label corrections*).

## Result

| Random units | Audited | Verified | Share (95% Wilson) | Mean balanced accuracy |
| --- | --- | --- | --- | --- |
| Transcoder latents | 10 | 9 | 90% (60–98%) | 0.92 |
| Neuron poles | 10 | 2 | 20% (6–51%) | 0.81 |
| ↳ positive poles | 4 | 2 | 50% (15–85%) | 0.95 |
| ↳ negative poles | 6 | 0 | 0% (0–39%) | 0.72 |

The two-sided Fisher exact test gives p = 0.0055 for latents against all neuron poles, and p = 0.18 for latents against positive poles only.

**Dead draws.** 4 of the first 14 latent draws never fired on the scan and could not be audited (seeds 1, 3, 6 and 13). No neuron draw was dead. Over the scanned layers, the drawn latents' scans put the dictionary's dead share at 29% on average. The dictionary explained 68% of the MLP output's variance, with 44 active latents per token.

**Sampling frame.** If a dead draw is counted as not verified, latents are 9 of 14 (64%).

## What drives the gap

**Negative poles in a GELU MLP.** In a GELU MLP the negative pole cannot pass this protocol as it stands. GPT-2's negative pole is GELU's dip, at most 0.170. All six drawn negative poles sat in the dip on most tokens, so each hypothesis described a default-on dial. A probe is active when its peak over all tokens reaches 0.25 × max, and almost any text has some token in the dip, so a hard negative can't pass.
- Two of the six hypotheses predict every prompt active, so their probe sets have no hard negatives, and they fail `enough_hard_negatives`. L9/N1046− says the pole is active on essentially any text. L3/N1517− states a floor at the first token that lies above the line.
- The other four got hard negatives built from the few tokens the hypothesis calls silent. Those negatives fired anyway: 0 of 11 passed for L5/N617−, L9/N330− and L9/N133−, which were REJECTED. For L7/N2511−, 7 of 11 passed, short of the specificity gate.
- This failure follows from how the protocol measures activity, not from a judgement that these neurons are uninterpretable. The comparison that tests the premise is latents against positive poles, and four positive poles are too few to settle it.

**Positive poles.** Of the four positive poles:
- L3/N2427+ (nearness, affection and attempt words) and L3/N1242+ (nouns of gradable quantity) were VERIFIED.
- L0/N375+ ("help" and its near-synonyms) failed specificity: 4 of 11 hard negatives fired.
- L2/N2331+ (lowercase subword fragments) stated the first-token floor, so it had no hard negatives.

**Latents.** Nine of the ten latents got narrow lexical or syntactic hypotheses that passed every gate. Examples: " reform", " inbox", curly-apostrophe "n’t", restrictive "that" after a wanted or needed noun, and "good" in generic statements. The exception is L0/F3001, which fires on 2 of 200k tokens, needing a control character and late position. It failed sensitivity, with 2 of 6 holdouts firing.

## Label corrections

Each fix went to the reviewer that wrote the probe:
- **L3/F12137.** " reformed" was relabelled active, since the hypothesis puts it at 0.3 of max. Two straddling probes were replaced.
- **L9/F1067.** Two holdouts used single-token words (" detector", " inspector"), which have no stem token. One negative contained `select`+`or` inside an identifier, which the hypothesis predicts active. One probe was ambiguous. All four were replaced.
- **L3/N2427+.** The courtroom "tried" negative sat above the line by the hypothesis's own number, and verb "close" sat on it. Both were replaced.
- **L3/N1242+.** "A degree of reassurance" was replaced.
- **L5/N617−.** A negative containing "across" was replaced, and five more were rewritten to avoid split words.
- **L3/N1517−.** Negatives with verbs right after a subject were replaced. After that, all its probes were relabelled active, since its explanation states a position-0 floor of 0.09 against a line of 0.0425.
- **L2/N2331+.** All probes were relabelled active, since its explanation states the first-token floor.
- **L9/N133−.** I first asked for the same relabel from a floor the proposer reported to me but that the submitted explanation does not state. The reviewer pointed this out, and its original labels were restored before review.

The judges also inverted two negative-pole explanations (L9/N133−), marking the texts the explanation says turn the pole off. Explanation scores are reported, not gated.

## Next

- Decide how GELU negative poles should be audited before running the full arm. One option is to sample positive poles only for GELU models in the baseline. Another is to measure a probe's activity away from position 0, which would be a protocol change.
- Then run N ≥ 40 per arm, which is what it takes to tell latents from positive poles.
