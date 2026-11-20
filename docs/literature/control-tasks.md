# Designing and Interpreting Probes with Control Tasks

**Citation.** John Hewitt and Percy Liang, of Stanford University. "Designing and Interpreting Probes with Control Tasks". *Proceedings of EMNLP-IJCNLP 2019*, pages 2733–2743, Hong Kong, November 2019. [ACL Anthology D19-1275](https://aclanthology.org/D19-1275/), [doi:10.18653/v1/D19-1275](https://doi.org/10.18653/v1/D19-1275). The anthology page marks it Best Paper Runner-Up. The preprint is [arXiv:1909.03368](https://arxiv.org/abs/1909.03368), version 1 of 8 September 2019, the only version. Code, data and experiments are on a [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0xb0c351d6f1ac4c51b54f1023786bf6b2). Funding: a PECASE Award.

**Mismatch.** Section 4.2 says ELMo's second layer is more selective "consistent across all probes". But Table 2 gives the bilinear dependency probe 6.7 on the first layer and 6.2 on the second. Section 3.5 gives the chosen MLP tagging probe rank 45, and the Table 1 caption gives rank 10. Section 3.2 says "6 methods" of complexity control, and the text and Figure 4 describe 5.

**Status.** Used. `papers/default-on-poles` cites it for control tasks and selectivity, and compares its token-lookup baseline with them. No code that the registry runs depends on it.

**Read.** On 2026-11-20 a `web-fetch` subagent read the anthology page and the HTML of arXiv version 1 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in two fetches: the abstract, Sections 1 to 6, the acknowledgements and the references. Figures 2 and 4 are images, and the subagent read only their captions. The PDF came back as bytes that it couldn't read.

## Summary

**Claim.** A probe's accuracy alone doesn't show what a representation encodes, because an expressive probe can learn the task itself. A control task, which only the probe can learn, shows how much of the accuracy is the probe's. A good probe is *selective*: accurate on the real task, and poor on the control.

**Method.**

- **Control tasks.** Each word type has an output chosen at random, and every token of that type keeps it. For part-of-speech tagging, the 45 outputs are drawn from the empirical tag distribution. For dependency edges, each type attaches to itself, to the first word or to the last, with equal chance. The output is "a deterministic function of the word type", so the probe must memorise it.
- **Selectivity.** "The difference between linguistic task accuracy and control task accuracy" (Figure 2 caption).
- **Representations.** ELMo's first and second BiLSTM layers, and Proj0, an untrained BiLSTM over ELMo's character embeddings.
- **Probes.** Linear (bilinear for dependencies), and MLPs with one or two hidden layers of 1,000 units by default.
- **Complexity control.** Rank or hidden size, dropout, fewer training examples, weight decay and early stopping, each swept on ELMo's first layer. Penn Treebank, with results on the development set.

**Findings, as the source reports them.**

| Setting | Accuracy | Control | Selectivity |
| --- | --- | --- | --- |
| Tagging, linear probe, defaults (Table 1) | 97.2 | 71.2 | 26.0 |
| Tagging, MLP-1, defaults (Table 1) | 97.3 | 92.8 | 4.5 |
| Tagging, MLP-1, chosen with control tasks (Table 1) | 97.2 | 80.6 | 16.6 |
| Dependencies, bilinear, defaults (Table 1) | 89.0 | 82.4 | 6.6 |
| Dependencies, MLP-1, defaults (Table 1) | 92.3 | 93.0 | −0.7 |
| Tagging, linear, ELMo layer 1 against layer 2 (Table 2) | 97.2 against 96.6 | | 26.0 against 31.4 |
| Tagging, linear, untrained Proj0 (Table 2) | 96.3 | | 20.6 |

- **Linear probes are selective, and default MLPs aren't.** On tagging, an MLP gains 0.1 points of accuracy and loses 21.5 points of selectivity.
- **Dropout doesn't fix it.** "Dropout ... does not consistently lead to selective MLP probes." A small hidden size works best, at little cost to accuracy.
- **A layer can lose accuracy and gain selectivity.** ELMo's second layer tags 0.6 points less accurately than its first but is 5.4 points more selective. The authors suggest that the first layer gives easier access to word identity.

## Where the claims go beyond the evidence

- **One number per setting.** The paper gives no seeds, no runs and no intervals. Many selectivity gaps are a few points.
- **One model, two tasks.** All results are on ELMo and the Penn Treebank. The complexity study uses only ELMo's first layer.
- **Settings chosen on the development set.** The probes "designed with control tasks" are picked by selectivity on the same data on which Table 1 reports them.
- **The text and tables disagree in places.** The mismatches above are small, but they touch the paper's chosen settings and one of its general claims.
- **Selectivity mixes two things.** A control task measures what a probe can memorise per word type. A representation that encodes word identity well also raises the control accuracy. So low selectivity can come from the probe or from the representation. The authors say as much in their explanation of ELMo's first layer.

## Use in this repo

- **Our citations check out.** `papers/default-on-poles` cites the paper twice.
  - Control tasks pair word types with random outputs that "can only be learned by the probe itself". The quote is from the abstract, and the description matches Section 2.
  - A good probe is selective, accurate on the task and poor on the control. That matches the definition in the Figure 2 caption.
  - The reference's authors, venue and pages match the anthology page. No change is needed.
- **The lookup baseline is an analogy, not a control task.** Our lookup reads the real target from the current token alone. A control task gives each type a random output and measures what the probe memorises. Both isolate word identity, so "like the control tasks" is fair. But the lookup doesn't measure a probe's capacity, and the paper doesn't say that it does.
- **The probe comparison has no control task.** The [probe comparison](../analyses/probe-comparison.md) reports probe accuracy only. A control task with random labels per token type would show how much of each accuracy is memorisation. The [probing-classifiers page](probing-classifiers.md) makes the same point from Belinkov's review.
- **Proj0 is the paper's random-weights null.** An untrained network tags at 96.3, close to ELMo's 97.2. That supports the repo's untrained null ([`docs/analyses/untrained-null.md`](../analyses/untrained-null.md)): accuracy near an untrained model's says little.
- **Question for the maintainer, recorded on this page only.** Should the probe comparison add a control task per token type and report selectivity?
- **Related pages.** [`probing-classifiers.md`](probing-classifiers.md) reviews control tasks among other controls. [`linear-probe-evidence.md`](linear-probe-evidence.md) and [`pretty-linear-liars.md`](pretty-linear-liars.md) review later probing work.
