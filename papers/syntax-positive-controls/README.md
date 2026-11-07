# A bracket neuron in GPT-2, and the label that hid it

Syntax labels from a parser should find the code neurons that the literature leads us to expect. So we swept every neuron pole of GPT-2 small over labelled source code. Our first bracket-depth label found nothing convincing, because it counted all brackets together. Counted for round brackets alone, one pole stands out: L2 N676+ is the best pole in its layer and across all 12 layers. On written text the same token reads 4.19 to 11.02 under one to five open brackets, and near zero once they close. At one open bracket, ablating the pole lowers the predicted chance of a closing bracket from 0.46 to 0.20. It is a short-range, imperfect counter of bracket characters, and it has not been through the registry's review.

![A function with its tokens lit by the pole's strength, read token by token](media/trace.svg)

**Results.**

1. **The labels carry signal.** Every label set far exceeds a null that gives each window another document's labels. The highest null R² in any layer and set is 0.028, against a best of 0.603 for `syntax node`. This does not show that a label's best pole encodes that label: token identity explains more of that pole (0.78).
2. **The first bracket label's winners were false leads.** Its best pole overall, L4 N3038−, has a flat strength from depth 1 up. L6 N1416− peaks in licence comments. On written text neither pole responds to depth. Of the two poles that won narrower depth tests, one doesn't rise with depth and the other falls, at a strength of at most 0.30.
3. **Round depth finds L2 N676+.** Its R² on held-out documents is 0.277, first of the layer's 6,144 poles, and 0.310 when we count `(` characters with no parser. On all brackets the same pole has at most 0.095. The first label hid it.
4. **Written text supports a depth signal.** With the target token and the token before it fixed, strength rises with open brackets and falls to about zero when they close. In one snippet it climbs to 9.15 at depth 4, and drops to 0.65 at the token that closes two brackets.
5. **The pole matters for a closing bracket.** Ablating it lowers the mean probability of a closing bracket at depths 1 to 4. Twenty random poles of the layer barely change it. At depth 5 the effect has a small opposite sign.
6. **The signal has clear limits.** It fades over about 20 list items, it weakens over line breaks, and it does not fully count down. It also fires inside strings and comments, so it follows characters and not the grammar.
7. **Cut documents broke the labels until we fixed them.** Corpus documents end mid-class. Before the fix, 45.9% of the characters inside brackets in those documents read as `top level`. Adding the missing closing brackets before a second parse lowers this to 12.5%.

These results do not show that L2 N676+ is the only bracket signal in GPT-2. They do not show how the model computes depth, or that the pole would earn VERIFIED under the protocol. The probes use narrow templates.

## Method

The scripts are `scripts/syntax_controls.py`, `scripts/syntax_probes.py` and `scripts/figures_syntax.py`. The tagger is `openinterp/syntax.py`. The output is in `results.jsonl`, which keeps no text from the corpus, because each file keeps its repository's licence.

- **Model.** GPT-2 small (Radford et al., 2019), on CPU, at 128-token windows, as the registry scans it. Each neuron has 2 poles, so each layer has 6,144.
- **Data.** 500 documents of [`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean) at commit `c48d40f9`, seed 0, each cut to 3,000 characters. The card gives the licence as Apache-2.0 and names no citation. Its `license` column shows that each file keeps its repository's licence. Of these, 396 documents have a grammar, and the seeded 200,000-token sample holds 266 of them.
- **Labels.** [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) parses each document, with a grammar for each of 15 languages. Its page describes it as useful even in the presence of syntax errors. We sort each grammar's node names into shared labels by pattern, so the labels are approximate.
- **Held-out R².** For each pole and label set, per-label means from half of the documents predict strengths in the other half. The two ways to split are averaged. So a pole can't score by fitting noise, and a useless label scores near or below zero. A planted signal scored 0.985 in a sanity check, and noise scored about 0.
- **Null.** Each window takes the labels of a window from the document before it. That keeps the labels' positions and run lengths, and removes their link to the text.
- **Probes.** Eight target tokens each follow `first,`, in templates that change only what comes earlier. All of the probe text is written for this paper. Each cell has 8 items, and the tables give the mean strength. We designed the probes after seeing the corpus results, so they check the sweep's winners on new text. They were not predictions registered in advance.
- **Ablation.** `ablate_pole` zeroes the positive part of a neuron's activation at every token of L2. The measure is the summed probability of next tokens that start with `)`. The 20 control poles are the positive poles of random L2 neurons, seed 0.

## What came before

The expectation came from recurrent models. Karpathy et al. (2015) trained LSTM cells on the Linux kernel source. They report cells that "turn on inside quotes, the parenthesis after if statements, inside strings or comments". Another cell turns on "with increasing strength as the indentation of a block of code increases". That is a character-level LSTM, which carries state from step to step. A transformer's MLP neuron has no such state, so the step from those cells to GPT-2 is an assumption. This repo's TODO made that step when it said that these neurons are known to exist in code models.

Evidence closer to GPT-2 is mixed. Friedman et al. (2024) study bracket depth in small transformers trained on parenthesis languages, and in character-level models trained on code, not in GPT-2. Gurnee et al. (2024) show a suppression neuron for tokens with an open parenthesis in a GPT-2 medium model. That is about output tokens, not depth. Clark et al. (2023) searched GPT-2 small for a circuit that tells a position it is inside brackets, in a report from a research sprint. They "did not find significant evidence of a circuit responsible for this behaviour", and found some evidence that the information is highly distributed. They also warn about the limits of the available tools. Our finding does not contradict theirs: one pole that matters is not a whole circuit, and the partial effect here fits a distributed signal.

## From the first label to round depth

`bracket depth` counts round, square and curly brackets, and ignores those in comments and strings. Its best poles looked like a result until we read them.

| Label set, best pole overall | Held-out R² | Token R² | What the pole does |
| --- | --- | --- | --- |
| `syntax node`, L1 N2566+ | 0.603 | 0.781 | `.` and `;` in Java imports. A token detector |
| `bracket depth`, L4 N3038− | 0.232 | 0.509 | Depth 0 against the rest. Flat from depth 1 (0.122, 0.133, 0.136, 0.137) |
| `indent depth`, L6 N1416− | 0.184 | 0.322 | Peaks in licence comments and imports. Also wins `syntax construct` at L6 |
| `syntax construct`, L2 N676+ | 0.276 | 0.258 | Call arguments 1.87 and parameters 1.83, against 0.46 or less elsewhere |

Token R² is in-sample and has many more levels than a label set, so it is a generous comparison. The first three poles fail it, and the fourth passes narrowly.

To remove the licence header, we repeated the depth sets without comments and strings. To ask for depth itself, not inside against outside, we also dropped depth 0. The best R² across layers falls, and the winners change.

| Label set | All tokens | No comments or strings | Also no depth 0 |
| --- | --- | --- | --- |
| `bracket depth` | 0.232, L4 N3038− | 0.195, L9 N2645+ | 0.080, L5 N1743+ |
| `indent depth` | 0.184, L6 N1416− | 0.259, L9 N2645+ | 0.118, L4 N1330+ |

Written probes (see *On written text*) then tested these winners. None rises with bracket depth.

| Pole | Open, depths 1 to 5 | Closed, depths 1 to 5 |
| --- | --- | --- |
| L4 N3038− | 0.15, 0.15, 0.16, 0.15, 0.16 | 0.15, 0.15, 0.16, 0.15, 0.16 |
| L6 N1416− | 0.00 at each depth | 0.00 at each depth |
| L5 N1743+ | 0.19, 0.12, 0.06, 0.03, 0.01 | 0.30, 0.28, 0.29, 0.29, 0.25 |
| L4 N1330+ | 0.00, 0.01, 0.02, 0.03, 0.03 | 0.04, 0.07, 0.07, 0.09, 0.15 |

L5 N1743+ does respond to open brackets. It falls from 0.19 to 0.01, and it stays near 0.29 when they close. So it may carry the opposite signal, at a strength of at most 0.30, where L2 N676+ reaches 11. These probes test only bracket depth, so they say nothing about L4 N1330+ and indentation.

L2 N676+ scored 0.276 on `syntax construct`, and much of that is call arguments and parameter lists. Its mean strength also rose with depth. This suggested two reasons for the weak depth score: curly brackets, which mostly hold blocks, and strings and comments, which the label leaves out. We then split the count.

![R² per layer for round and all brackets, and the three counts for L2 N676+](media/labels.svg)

| Count of depth | Best pole, L2 | Best R², L2 | L2 N676+ | Its rank in L2 |
| --- | --- | --- | --- | --- |
| All brackets | L2 N1513− | 0.142 | at most 0.095 (in-sample) | not in the top 5 |
| Round | L2 N676+ | 0.277 | 0.277 | 1 |
| Round characters, no parser | L2 N676+ | 0.310 | 0.310 | 1 |
| Square | L2 N1441+ | 0.051 | 0.017 | 4 |
| Curly | L2 N1513− | 0.106 | 0.000 | 3,998 |

Its round R² is the highest of any layer's best pole. The nearest are L3 N695+ (0.232) and L8 N689+ (0.230). L2 N1195+ (0.236) and the other pole of the same neuron, L2 N676− (0.225), are next in its layer. We did not test those on written text.

## On written text

![Strength at one token, by depth and condition](media/probes.svg)

| Condition | Depth 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Open: `out = load(parse(first, value` | 4.19 | 6.75 | 8.63 | 9.99 | 11.02 |
| Closed: `out = load(parse(x))`, then a new line | 0.00 | 0.00 | 0.12 | 0.54 | 0.00 |
| One closed: one more opened, then closed | 1.65 | 4.11 | 5.82 | 7.08 | 8.00 |
| In a string | 4.63 | 7.06 | 8.85 | 10.17 | 11.17 |
| In a comment, then a new line | 2.30 | 4.72 | 6.48 | 7.77 | 8.76 |
| Square brackets | 3.51 | 5.62 | 7.14 | 8.22 | 9.02 |
| Curly brackets, as `load{first, value` | 1.49 | 2.82 | 3.76 | 4.45 | 4.98 |
| Each call on its own line | 2.82 | 3.36 | 2.52 | 1.38 | 0.49 |

The standard errors over items are at most 0.21. Four things limit the reading of "counter".

- **It doesn't fully count down.** A pole that counts would read the same under `One closed` as under `Open`. It reads lower, by 2.5 to 3.0. The tokens just before the target differ too, so this comparison is rough.
- **Strings and comments do not stop it.** A parser says these brackets are not brackets. The pole reads them almost as it reads code, and its R² is higher for characters than for parsed brackets.
- **Lines matter.** With each call on its own line, strength peaks at depth 2 and then falls. The comment row still reads high after one line break, so one newline is not a reset.
- **Curly brackets differ between tests.** The pole responds to `load{`, but its corpus R² for curly depth is zero. Curly brackets in real code mostly hold blocks. We did not write probes with blocks.

Strength also fades with distance. These rows have one open bracket and a list of filler items between it and the target.

| Filler items | 0 | 5 | 10 | 20 | 40 | 80 | 160 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Open | 4.23 | 2.20 | 0.56 | 0.00 | 0.00 | 0.00 | 0.00 |
| Closed | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

![Strength with distance from one bracket, and across line breaks](media/reach.svg)

This is far shorter than the LSTM's range: Karpathy et al. (2015) show a quote cell that holds for about 230 characters. Their cell and this pole measure different things on different data, so the comparison is only a rough one.

## Ablation

![Probability of a closing bracket with and without the pole](media/ablation.svg)

| Open brackets | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Model | 0.457 | 0.464 | 0.453 | 0.387 | 0.309 |
| L2 N676+ ablated | 0.199 | 0.255 | 0.263 | 0.297 | 0.324 |
| Mean of 20 random poles ablated | 0.458 | 0.464 | 0.453 | 0.387 | 0.309 |
| Lowest of those 20 | 0.455 | 0.462 | 0.450 | 0.384 | 0.306 |

So the pole carries part of what the model uses at low depth. The probability doesn't reach zero, which fits other routes to the same output. The effect shrinks as strength grows, which we did not expect and do not explain. Other routes may take over at depth, or the measure may fit less well there. We did not test either idea.

## Cut documents

The corpus cuts documents at 3,000 characters, usually inside a class or a function. The parser then has no node for that class, and its contents read as top level. We found this when a Java example at depth 1 and indent 4 read `top level`. So the tagger now adds the missing closers and a `;` before each `}`, parses again, and labels only the original text.

To measure this, we parse the first 1,500 characters of the 157 documents at the 3,000-character limit. Then we compare their labels with those of the longer cut. A stable label agrees. Inside brackets, a label of `top level` is wrong almost always.

| Language | `syntax construct` agreement, before | After | Inside brackets but `top level`, before | After |
| --- | --- | --- | --- | --- |
| All | 0.948 | 0.966 | 0.459 | 0.125 |
| C | 0.975 | 0.970 | 0.500 | 0.204 |
| C# | 0.928 | 0.982 | 0.679 | 0.050 |
| C++ | 0.945 | 0.986 | 0.433 | 0.143 |
| Go | 0.999 | 0.999 | 0.158 | 0.157 |
| Java | 0.915 | 0.970 | 0.589 | 0.040 |
| JavaScript | 0.916 | 0.959 | 0.397 | 0.059 |
| PHP | 0.943 | 0.831 | 0.626 | 0.311 |
| Python | 0.983 | 0.991 | 0.001 | 0.001 |
| Ruby | 0.999 | 0.999 | 0.257 | 0.257 |
| Scala | 0.859 | 1.000 | 0.000 | 0.000 |
| TypeScript | 0.940 | 1.000 | 0.142 | 0.000 |

`syntax node` agreement stays at 0.975 overall. The fix helps most languages. It lowers PHP's agreement, and it can't help Ruby, whose blocks close with `end`. Few documents stand behind some rows: Scala has 1, TypeScript 3 and Go 4. Haskell, Lua, Rust and Shell had no document at the limit in this sample, so this table doesn't test them.

## Caveats

- **One model, one corpus sample, one seed.** We did not test Qwen, or another sample of documents.
- **Selection.** We chose this pole from 73,728 poles, and we changed the label after seeing results. Held-out R² guards against noise in the fit, not against our choices. The written probes and the ablation are separate tests of the pole, but they came after we chose it.
- **Few probes.** Each cell has 8 items from one template family, in Python-like text. Standard errors cover the items, not the choice of template.
- **No confidence interval on ablation.** The 20 controls show the size of a random pole's effect, and no more.
- **R² is a weak search.** A pole that responds to a label only sometimes scores low, and the depth-tracking poles here score under 0.32. A sweep can miss signals, including any spread over several neurons.
- **The label is approximate.** Node names are sorted by pattern, and angle brackets don't count.

## Effect on the protocol

None. The labels are reported, not gated. A scan of a corpus that records programming languages now has a `round depth` label set, because of this result. The registry's scan corpus has no such field, so its scans don't change. The registry has no record of L2 N676 today.

## Reproduce

The sweeps take about an hour each on CPU. The other commands take a few minutes. `results.jsonl` stores the outputs, in the order of these commands. Parser and grammar versions are in `uv.lock`.

```bash
uv run openinterp corpus --dataset github-code --docs 500 --seed 0 --out code.jsonl
uv run python scripts/syntax_controls.py sweep --corpus code.jsonl --out papers/syntax-positive-controls/results.jsonl
uv run python scripts/syntax_probes.py --results papers/syntax-positive-controls/results.jsonl
uv run python scripts/syntax_controls.py kinds --corpus code.jsonl --out papers/syntax-positive-controls/results.jsonl
uv run python scripts/paper_figures.py
```

The corpus's SHA-256 is `549de3ddf5ee7d13c54dd32241e213745f6a21c8beab3d00e14e4b1279111a31`.

## Next

Audit L2 N676+ through the protocol, on a corpus with code. Test the other high round-depth poles in the same way, and ablate them together. That shows whether they account for what is left of the closing bracket. Repeat the sweep on Qwen2.5-0.5B and 1.5B. Write probes with real blocks for curly brackets. Improve the labels of cut documents in PHP and Ruby.

## References

- Clark, T., Roman, A. & Thurnherr, H. (2023). *Who cares about brackets?* Research report, Apart Research Alignment Jam #10 (Interpretability 3.0). [PDF](https://webflow.com/files/634e78122252d2e2fc3a9ab9/formUploads/4679d69c-0a9d-4e19-a09a-2638b1b8b133.pdf).
- Friedman, D., Lampinen, A., Dixon, L., Chen, D. & Ghandeharioun, A. (2024). *Interpretability Illusions in the Generalization of Simplified Models*. ICML 2024. [arXiv:2312.03656](https://arxiv.org/abs/2312.03656).
- Gurnee, W., Horsley, T., Guo, Z. C., Kheirkhah, T. R., Sun, Q., Hathaway, W., Nanda, N. & Bertsimas, D. (2024). *Universal Neurons in GPT2 Language Models*. [arXiv:2401.12181](https://arxiv.org/abs/2401.12181).
- Karpathy, A., Johnson, J. & Fei-Fei, L. (2015). *Visualizing and Understanding Recurrent Networks*. [arXiv:1506.02078](https://arxiv.org/abs/1506.02078).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
