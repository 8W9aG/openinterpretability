# OLMo: Accelerating the Science of Language Models

**Citation.** Dirk Groeneveld, Iz Beltagy, Pete Walsh and 40 other authors, the last Hannaneh Hajishirzi, most of them at the Allen Institute for AI. "OLMo: Accelerating the Science of Language Models". *Proceedings of the 62nd Annual Meeting of the ACL (ACL 2024)*, Volume 1, pages 15789–15809. [ACL Anthology 2024.acl-long.841](https://aclanthology.org/2024.acl-long.841/), [doi:10.18653/v1/2024.acl-long.841](https://doi.org/10.18653/v1/2024.acl-long.841). The anthology page marks it Best Theme Paper. The preprint is [arXiv:2402.00838](https://arxiv.org/abs/2402.00838), version 1 of 1 February 2024, last revised as version 4 on 7 June 2024.

**Mismatch.** The anthology lists the third author as Evan Walsh, and arXiv as Pete Walsh. Our reference follows arXiv.

**Status.** Planned. `papers/neuron-development` names OLMo as a next step, and open issues plan the work. #4156 would repeat the neuron-development measures on OLMo. #3338 would add OLMo 2 1B, a later model, as a configured model. #3377 would link verified neurons to OLMo's training data.

**Read.** On 2026-11-20 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the anthology page, the arXiv abstract page and the HTML of version 4. It had Sections 1 to 6, the limitations and the ethics statement in the source's own words. Appendices A to E came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** A truly open language model comes with its training data, its code, its logs and its intermediate checkpoints. That lets researchers study how such models work and how they're made.

**Method.**

- **Models.** OLMo-1B (16 layers, width 2,048) and OLMo-7B (32 layers), trained on at least 2T tokens, plus a second 7B run on other hardware.
- **Architecture.** No biases, a layer norm with no learned parameters, SwiGLU and rotary embeddings. A vocabulary of 50,280 tokens, from a modified GPT-NeoX tokenizer.
- **Data.** Dolma, about 2.7T tokens, mostly from Common Crawl.
- **Release.** "500+ intermediate checkpoints at intervals of 1000 steps" for every model, with the training code and the full Weights & Biases metrics. Dolma comes with tools to recreate the data order and inspect the data seen at each step. Code and weights are under Apache 2.0.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Zero-shot average, 8 core tasks (Table 3) | OLMo-7B 69.3, against Llama 2 7B 70.5 and Pythia 6.9B 63.0 |
| Same, 1B scale (Table 3) | OLMo-1B 60.4, against Pythia 1B 54.5 |
| After instruction tuning and DPO, 7B (Table 4) | MMLU 28.3 to 46.2, toxic outputs 81.4% to 1.7% |

## Where the claims go beyond the evidence

- **Open, but English only.** The authors say the data is English, and "likely contains problematic content like toxic language, personal information, and copyrighted text".
- **No failed runs.** The authors didn't publish logs of runs that "diverged or failed to learn", for lack of space.
- **Benchmarks are noisy.** The authors call their evaluations "very noisy" and not representative of chat use.
- **One run per configuration.** Each model is one training run.

## Use in this repo

- **Our citation checks out.** `papers/neuron-development` says OLMo's "training run is also public". The paper releases the checkpoints, data, code and logs, so that holds. The reference's first and last authors, ACL 2024 and the link match.
- **Its checkpoints suit the neuron-development measures.** OLMo releases a checkpoint every 1,000 steps, as Pythia does. The source names no checkpoints before step 1,000. Pythia has steps 1 to 512 ([`pythia.md`](pythia.md)). Our reading: unless OLMo released earlier steps elsewhere, it can't show the very start of training, and #4156 should check.
- **Its data can be traced.** OLMo releases the data order and tools to find what the model saw at each step. That makes #3377's link from a neuron's triggers to its training text possible, which Pythia's data also allows.
- **OLMo 2 is a different paper.** #3338 adds OLMo 2 1B, from a later report. This page reviews the first OLMo only.
- **No question for the maintainer.**
- **Related pages.** [`pythia.md`](pythia.md) reviews the other suite with public checkpoints.
