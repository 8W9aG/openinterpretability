# Interpretable Weight-Space Programming Builds Transparent Transformers and Brain Encoding Models

**Citation.** Yash Mehta (Microsoft Research and Johns Hopkins University), Chandan Singh (Microsoft Research), Weijia Xu (Microsoft Research), Richard J. Antonello (Columbia University), Adeen Flinker (NYU), Nima Mesgarani (Columbia University), Michael F. Bonner (Johns Hopkins University), Nebojsa Jojic (Microsoft Research) and Jianfeng Gao (Microsoft Research). "Interpretable weight-space programming builds transparent transformers and brain encoding models". arXiv:2611.18114 [cs.CE], v1, 13 November 2026. [Paper](https://arxiv.org/abs/2611.18114), [code, prompts and logs](https://github.com/csinva/weight-space-programming-autoresearch) (MIT). The agents are Claude Code with Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5. The story and ECoG searches used Claude Opus 4.7, and the visual search Gemini 4 Carbon. We found no funding statement. **Mismatch:** our issue listed the paper under NeurIPS 2026. arXiv gives no venue, and the repository's citation is `@misc`, so we treat it as a preprint.

**Status.** Planned. #4782 would use it in its strand on weight-space programming. Its rebuild test is evidence for #3249, which would explain heads as executable programs. No code, skill or protocol here depends on it.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the arXiv abstract page and the whole HTML of v1 in six parts. It has all appendices, tables and figure captions in the source's own words. It also read the repository's README and the alphaXiv page. We did not see the figure images or the PDF, and we did not use alphaXiv's AI overview. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** An LLM coding agent can write a model's weights or features directly in Python, with no backpropagation, by revising its code against a fixed evaluator. The models it writes are readable: a fresh agent can rebuild one from a description of at most 200 words. On brain encoding, the written models match or beat classical interpretable models, and come near pretrained networks.

**Method.**

- **Algorithmic and NLP tasks.** A small GPT-style transformer (3 layers, width 64) whose weights the agent writes in `write_weights`. "The model is NEVER trained." There are 13 algorithmic tasks (for example addition, sort, GCD, edit distance) and four NLP tasks (TREC, SST-2, MRPC, SNLI). The visible set has 200 examples, seen each iteration, and the hidden set 2,000. There were about 39 iterations a run, in a restricted session with six tools and no web.
- **Brain encoding.** The agent writes features. A fixed evaluator fits ridge regression and returns Pearson r on a visible set. The data are visual fMRI (LAION-fMRI, tested zero-shot on NSD), story-listening fMRI (one subject) and speech ECoG (one patient, five more for transfer).
- **Guards.** Hidden sets, and an auditing agent that reads the saved models and logs afterwards. The paper says that the harness "cannot stop the agent's own Python from training", so those rules are "instructions that we audit afterward".
- **The rebuild test.** Opus 5.5 writes a description of the best model, at most 200 words. A fresh session gets only that and the task, and rebuilds the model. The control gets the task with no description.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Algorithmic, two runs per task per model | 100% hidden accuracy: Opus 26 of 26, Sonnet 25 of 26, Haiku 25 of 26 |
| Smallest model at 100% visible accuracy (median) | Opus 166 parameters, Sonnet 41,600, Haiku 100,000 |
| TREC, hidden accuracy | Opus 87.3%, Sonnet 86.6%, Haiku 71.1% |
| Rebuild, 13 algorithmic models | All at 100% hidden accuracy, with 1.74 times the original's parameters (controls 2,043 times) |
| Rebuild, NLP | About 1.4 points of hidden accuracy lost. Error correlation with the original 0.74, against 0.51 for controls |
| Visual fMRI on NSD, four subjects | Early cortex 0.443 against 0.400 for the best classical model. Ventral 0.317 against 0.256. AlexNet ventral 0.391 |
| Story fMRI, visible set | 0.067 → 0.104 in one step, from features of a word's position in the story. Position features alone about 0.090. GPT-2 XL 0.079 |
| Speech ECoG, all 166 electrodes, hidden | 0.065 against Whisper's 0.070. Combined 0.075, a gain of 0.0052 (95% CI 0.0039–0.0064) |
| Cost | About $17,500 in agent calls at list prices |

## Where the claims go beyond the evidence

- **The scan-position confound came from the visible set.** The story search optimised a score that the agent saw every iteration, and it found a drift in the scanner, not language. The authors caught it by reading the code. On the hidden set, the models after the jump stay below GPT-2 XL's 0.046 until the final one. With the scan edges removed, the hidden scores are 0.023 and 0.026, against 0.047 for GPT-2 XL.
- **The guards are instructions and a post-hoc audit.** The story and ECoG searches ran with no sandbox. An LLM audits the logs afterwards.
- **"Readable" means describable to another agent.** The rebuild test shows that 200 words carry enough to rebuild the behaviour. It does not show that a person can follow the model, and no unit is tested by its activations or by ablation. The GCD model of Figure 2e (three ReLU units, repeated) differs from the more compact one described in Table A3.
- **No trained baseline for the transformers.** The algorithmic and NLP results compare with majority answers only.
- **Few subjects and runs.** One story subject and one ECoG patient for the searches. Two runs per task per model, with no seeds.
- **The trim length varies.** The text recommends about 40 TRs, and the figures use 30.

The authors state several limits. The agents "do not invent genuinely new computational principles", and use shallow motifs. The method relies on the agent's written explanations staying faithful to its weights. It is slow and costly. The neuroscience is observational, with few subjects. A perfect score on the test does not prove right outputs on all inputs.

## Use in this repo

- **The rebuild test is a stricter cousin of simulation scoring.** `open-interpretability score` asks a judge to predict activations from an explanation (detection and fuzzing, the `explanation-simulation-scoring` skill). The rebuild test asks a fresh agent to write the mechanism from a description, and scores what it builds. For a head, that is #3249's executable program. We put this on #3249 as evidence.
- **A fixed visible evaluator invites confounds. The registry's design guards against that.** The scan-position confound is what an agent finds when it iterates on a score it can see. The registry's proposer never sees the reviewer's probes, and the reviewer never sees a measurement (`docs/PROTOCOL.md`). This case belongs in the `interpretability-illusions-review` skill as an example of a high score from a confound in the data.
- **Written transformers are ground truth by construction.** The TREC model's neurons are described as exact AND gates, and the sentiment model has one neuron per hand-written word pattern. Like the planted neurons of [`docs/analyses/ground-truth-calibration.md`](../analyses/ground-truth-calibration.md) (#3201), these units have a known role. The pipeline runs only on configured Hugging Face models, so auditing them would need a loader for the paper's small GPT.
- **Neighbours.** [`rikiyat-weight-space-programming.md`](rikiyat-weight-space-programming.md) has an LLM read and edit weights. [`weight-space-programming-without-transformers.md`](weight-space-programming-without-transformers.md) has an LLM write tables. This paper is the most careful of the three about hidden sets and audits.
- **A question for the maintainer.** Should the registry audit this paper's written transformers as a ground-truth set, now that the code and weights are public? It would need a loader for a model outside `[tool.openinterpretability] models`. That widens the pipeline's scope, so we open no issue.
