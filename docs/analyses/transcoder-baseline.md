# Transcoder baseline: GPT-2 small, 40 per arm

The full run of the learned-basis baseline (`TODO.md`, *Scope expansion*). It compares 40 random latents of Dunefsky et al.'s GPT-2 small transcoders (`gpt2-small-transcoders`) with 40 random GPT-2 small positive neuron poles. Both arms were audited under protocol 0.7. The home page's *Learned-basis baseline* table is computed from the same records (`openinterp/builder/learned_baseline.py`). Every draw, live or dead, is in `transcoder-baseline.jsonl` with its verdict and record. The 10-per-arm pilot it extends is in `transcoder-baseline-pilot.md`.

## Setup

- **Draws.** Latents come from `openinterp audit --random --seed s --dictionary gpt2-small-transcoders`. The pilot drew seeds 1–14 and the full run seeds 15–61, in order, until 40 latents were live. Neuron poles come from `openinterp audit --random --pole positive --seed s --model openai-community/gpt2`, seeds 1–40.
- **Why positive poles only.** The pilot found that GPT-2's negative poles cannot pass the protocol (see the pilot's *What drives the gap*), so the native arm samples positive poles only.
- **Pilot units carry over.** `--pole positive` draws the layer and neuron the way uniform-pole sampling does. Seeds 1–4 therefore gave the pilot's four positive poles, and their pilot records stand. The six negative poles the pilot drew stay in the registry. The site reports them as their own row and leaves them out of the main comparison.
- **No draw skipped or repeated.** No seed was skipped and none was audited twice.
- **Scan.** All scans read `corpus.jsonl` (sha256 `ded48cfc…`): the default 200,000-token sample, in 128-token windows with stride 64. The transcoders fit no longer context (`harness/dictionaries.py`), and both arms were scanned the same way.
- **Agents.** Every unit had the same roles: proposer `claude-opus-5-5`, reviewer `claude-sonnet-5-5`, and judges `claude-haiku-5-5`. All reviews are cross-model. Each proposer and reviewer worked in its own scratch directory.
- **Probe labels.** Before each review, the labels were checked against the submitted explanation, the 0.25 × max line and GPT-2's token splits. A label that contradicted the explanation, or hinged on a borderline number, went back to the same reviewer (see *Label corrections*).

## Result

| Random units | Audited | Verified | Share (95% Wilson) | Mean balanced accuracy |
| --- | --- | --- | --- | --- |
| Transcoder latents | 40 | 37 | 92% (80–97%) | 0.95 |
| Positive neuron poles | 40 | 20 | 50% (35–65%) | 0.90 |
| *Negative poles (pilot, not in the comparison)* | 6 | 0 | 0% (0–39%) | 0.72 |

The two-sided Fisher exact test of latents against positive poles gives p = 4.3 × 10⁻⁵. Against all 46 neuron poles, including the pilot's negative ones, it gives p = 9.7 × 10⁻⁷.

**Dead draws.** 21 of the 61 latent draws never fired on the scan, so they could not be audited: 4 of 14 in the pilot and 17 of 47 in the full run. No neuron draw was dead. Over the scanned layers, the drawn latents' scans put the dictionary's dead share at 29% on average. The dictionary explained 70% of the MLP output's variance, with a mean L0 of 54 active latents per token.

**Sampling frame.** Counting a dead draw as not verified leaves latents at 37 of 61 (61%, Wilson 48–72%). That is still above the positive poles' 50%, but the gap is no longer significant (Fisher p = 0.31). The comparison above is among units that fire. The dictionary's coverage is a separate question: about a third of its latents never fire on 200k tokens.

## What drives the gap

**Latents.** 37 of 40 latents got narrow lexical or syntactic hypotheses that passed every gate. Many are single-token detectors: ' 39', ' favorite', ' passionate', 'iOS', ' per' after a number, and 'range' with no leading space. Others are one word in one sense: 'result' as a causal verb, ' take' as a noun, and 'let' opening a clause. A few are syntactic slots:
- the determiner of a prepositional or verbal object;
- the hyphen after ' time' in a compound;
- the first word after 'including' or 'such as';
- a conjunction after a legal term.

Three latents failed:
- **L0/F3001 (pilot): sensitivity.** It fires on 2 of 200k tokens.
- **L0/F15387: counterfactual reproduction.** It is a GPU detector, and 3 of its 8 pairs missed their expected delta.
- **L9/F6460: lookahead consistency.** Its hypothesis is a lookahead one, firing on the token before a likely money noun.

None failed specificity.

**Positive poles.** 20 of 40 failed, in three ways:
- **Specificity: 13.** Hard negatives fired: 3 to 6 of 10–11 for each. These are broad or polysemantic poles, such as orientation words, ALL-CAPS runs, locative words, negators, size words, Germanic-language text and restrictive 'only'. The proposer described a family, and the reviewer found members outside it.
- **No hard negatives: 6.** The explanation states a floor that every prompt clears, so every probe is predicted active and `enough_hard_negatives` fails. All six are in layers 1–2: L1/N2522+, L1/N3040+, L1/N874+, L2/N1712+, L2/N993+ and L2/N2331+. These early-layer poles sit above the line at the first token of any text. That is the positive-pole form of the position-0 effect that sank the negative poles in the pilot.
- **Confound baseline: 1.** L3/N47+ is a code-declaration keyword detector, and it did not beat the token-identity baseline.

Of the 18 that failed in the full run, every one passed all its counterfactuals, and all but three passed every positive holdout. So each hypothesis described something real about its pole. They failed on being too broad, not on being wrong. Leaving out the six floor cases gives 20 of 34 (59%).

**Balanced accuracy.** Mean balanced accuracy is 0.95 for latents and 0.90 for positive poles. A verified pole is about as sharp as a verified latent. The poles' lower mean comes from the failures, most of which still score 0.75–0.86.

## Label corrections

Each fix went to the reviewer that wrote the probe, before the review ran. In the full run, corrections fell into four kinds:
- **Trigger inside an unlisted context.** A negative contained a listed trigger in a context the explanation names. For example, L0/F9374's "3900" tokenizes as ' 39' + '00'. L9/F6460's "annual festival" and "extra waiters" contain modifiers that its explanation says fire on their own.
- **Borderline label.** The label hinged on a word the explanation doesn't place. Examples: "at the" and "from several" for L8/F2494; "ten-minute-long" for L4/F1786, after a time-quantity noun; and causal "Since 500" for L7/F5954.
- **Stated floor.** An explanation stated a floor that every prompt clears, so all its probes were relabelled active. This happened for the positive poles listed above.
- **Tokenizer splits.** Probes were rewritten where a label depended on how GPT-2 splits a word.

The checks used only the submitted explanation, never the proposer's private notes, as the pilot's L9/N133− case had taught.

## Caveats

- **Same agents, both arms.** One agent of each role did both arms, so a model that writes better hypotheses for sparse units favours latents. That bias is the thing being measured, since an interpretable unit is one that has a short, testable description. But the result is specific to these agents and this protocol.
- **Judge prompts.** In explanation-scoring judge prompts, curly apostrophes in corpus text render as "��". Explanation scores are reported, not gated, so verdicts are unaffected.
- **Layers.** Layers were drawn uniformly. The early-layer floor affects the positive poles of layers 1–2 most.

## Next

- Audit GELU poles away from position 0 (the *(protocol)* item in `TODO.md`). That would let the six floor cases and the negative poles be tested on their tokens.
- Run the same comparison on another model and dictionary (Gemma Scope transcoders), to see whether the gap holds beyond GPT-2 small.
