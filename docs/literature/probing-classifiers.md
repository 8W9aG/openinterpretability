# Probing Classifiers: Promises, Shortcomings, and Advances

**Citation.** Yonatan Belinkov, of the Technion – Israel Institute of Technology. "Probing Classifiers: Promises, Shortcomings, and Advances". *Computational Linguistics*, 48(1), 207–219, 2022. [doi:10.1162/coli_a_00422](https://doi.org/10.1162/coli_a_00422). The volume, issue, pages, year and licence (CC BY-NC-ND 4.0) come from Crossref's record. The preprint is [arXiv:2102.12452](https://arxiv.org/abs/2102.12452), with versions of 24 February, 4 March, 5 August and 22 September 2021. The journal published it as a squib, a short critical article. Funding: Israel Science Foundation grant 448/20, an Azrieli Foundation Early Career Faculty Fellowship, and the Viterbi Fellowship at the Technion.

**Mismatch.** None found. The section numbers below are those of arXiv version 4, which uses the journal's template. We couldn't open the publisher's page, so we didn't check that the printed numbering is the same.

**Status.** Used. `papers/default-on-poles/` cites it for its random baselines and for testing use by intervention, and its decoding tests follow that advice. It is not code that the registry runs.

**Read.** On 2026-11-20 a `web-fetch` subagent read the arXiv abstract page and the HTML of version 4 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 5, the footnotes, the acknowledgements and the first part of the references in the source's own text. The rest of the reference list came back only as a summary. The publisher's page refused the request (HTTP 403), so the metadata above is from Crossref. We didn't read versions 1 to 3.

## Summary

**Claim.** A probing classifier, trained to predict a property from a model's representations, is a useful but easily misread tool. Its accuracy means little without baselines and controls. And it shows what can be decoded, not what the model uses.

**Method.** A critical review. Section 2 sets out the framework: the original model f, and a probe g trained on f's representations. A careful study adds baselines, skylines, control tasks, control functions and control datasets. It can also add selectivity, information gain, minimum description length (MDL) and representations after an intervention. Section 3 covers the method's promises, and Section 4 its shortcomings and the advances that answer them.

**Findings, as the source reports them.**

| Section | Point |
| --- | --- |
| 4.1 Comparisons and controls | Random baselines show that "even random features capture significant information that can be decoded by the probing classifier". Control tasks with random labels define selectivity, and linear probes tend to be more selective than non-linear ones |
| 4.2 Which classifier to use? | Simple probes are often advised. Selectivity can reverse a ranking that accuracy gives. MDL measures a probe's complexity with its accuracy |
| 4.3 Correlation vs. causation | Probing may show a correlation, but "it does not tell us whether this property is involved in predictions" of the model. Interventions that remove or change the property, such as iterative projection (Elazar et al.), test use instead |
| 4.4 Datasets vs. tasks | A probing dataset is an imperfect stand-in for a task, and its size and make-up are little studied |
| 4.5 Properties must be pre-defined | Probing finds only properties chosen and labelled in advance |
| 5 Summary | Better controls and metrics allow relative claims. Causal methods may show which properties are used |

## Where the claims go beyond the evidence

- **A review, with no new experiments.** Each point rests on the work it cites. That is the article's purpose, and the author says so.
- **No single prescription.** The article doesn't require every study to run every control, so it leaves the choice of control to the reader.
- **Little on single units.** Neuron-level probing and neuron ablation are out of its scope. Footnote 11 sets aside intervention studies on neurons, such as Bau et al. (2019).

## Use in this repo

- **Our citations check out.** `papers/default-on-poles/README.md` cites it three times:
  - Section 4.1 and the random-features quote. The words and the section match.
  - Section 4.3 and "does not tell us whether this property is involved in predictions". The words match up to "predictions", where the source continues "of f". The section matches.
  - "A readout shows what is present, not what later layers read." That is a fair paraphrase of Section 4.3.

  No change to the paper is needed.
- **The registry already follows its two main lessons.** A verdict tests a unit with a random null, as Section 4.1 asks, and with interventions, as Section 4.3 asks. Probing alone never decides a verdict. The [probe comparison](../analyses/probe-comparison.md) reports probes but gates nothing.
- **Selectivity bears on the probe comparison.** Section 4.1 reports that linear probes are more selective than non-linear ones. The probe comparison uses linear probes, but it has no control task with random labels. Our reading: a selectivity score would show how much of each probe's accuracy is the probe memorising the labels. The [Pretty Linear Liars page](pretty-linear-liars.md) raises the related risk of weak regularisation.
- **Section 4.5 is the case for explanations written from activations.** Probing needs the property chosen first. The registry's proposers write a hypothesis from a unit's top activations instead, and the reviewer then tests it. That is the opposite direction, from unit to property.
- **No question for the maintainer.**
- **Related pages.** [`linear-probe-evidence.md`](linear-probe-evidence.md) and [`activation-probes-few-positives.md`](activation-probes-few-positives.md) review later probing work. [`neurons-in-a-haystack.md`](neurons-in-a-haystack.md) is probing at the neuron level.
