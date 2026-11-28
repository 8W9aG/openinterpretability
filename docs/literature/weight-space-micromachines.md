# Weight-Space Micromachines: Brain Implants for LLMs

**Citation.** "Weight-Space Micromachines: Brain Implants for LLMs". [Post](https://papail.io/papers/wsm/) on papail.io, November 2026, with code at [anadim/weight-space-micromachines](https://github.com/anadim/weight-space-micromachines) under the MIT licence. **Authorship isn't stated.** The post has no byline and no date. The licence names "The Weight-Space Micromachines authors". The site and the GitHub account are Dimitris Papailiopoulos's, and the post's references list him as a co-author only of other works. So this page credits the post to its site, not to a named author. The repo's first commit, "initial release", is of 17 November 2026, and the latest we saw is of 23 November. The post says that Claude Code wrote the implants: "So I had Claude Code try this on a few small open models". The `claude` account co-authored the commits. It is a blog post with no peer review, and it states no funding.

**Status.** Planned. The fix campaign (#5220) plans small changes that give a model a skill or block a behaviour. An implant is such a change, but it sits beside the frozen model rather than inside its weights. #7356 would test a VERIFIED pole as an implant's trigger. No code, skill or protocol here depends on the post. The maintainer designs the fix campaign, so this page reports evidence only.

**Read.** On 2026-11-28 a `web-fetch` subagent read the post for this page's author, a model (Claude Opus 5.5). It also read the repo's README, licence and commit history, and two of its documents, `DETAILS.md` and `REPRODUCE.md`. It read the whole post and the README in the source's own words. It read about the first 40,000 characters of `DETAILS.md` and `REPRODUCE.md` in their own words, and the rest only as a summary. Quotes were capped at 125 characters, so the rest came back paraphrased. We didn't see the off-task benchmark numbers in `REPRODUCE.md` §24, the curves of §12, or the sample size of the latent write.

## Summary

**Claim.** A coding agent can give a small language model an exact skill. It writes a small network, an *implant*, straight into weight space beside the frozen model. An implant stays silent, adding exactly zero, until a trigger fires. It then writes its result back into the context or the hidden states. The post says these are not LoRAs.

**Method.**

- **The agent's loop.** Run the model, read its failures, pick a pattern, write a machine for it into weights, attach it, and keep it if the scores rise.
- **The implants.**
  - A calculator of about 2.4M weights, about 50 compiled ReLU layers, with read and write ends fitted by hand.
  - A mistake detector, a three-head probe on the hidden states.
  - A topic blocker: a linear probe compiled into one ReLU unit, on Qwen3-1.7B chat.
  - A latent write into the hidden states, which looks up to 6 tokens ahead.
  - 14 IFEval rules, each a reader and an enforcer.
  - 8 BBH tools behind a call gate.
- **RL.** GRPO for 300 updates on Qwen3-0.6B-Base, with and without implants attached.
- **Dead neurons.** The post suggests writing implants into dead neurons, but doesn't do it.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| BBH | Qwen3-0.6B-Base: base, against a fine-tuned host with 8 tools and the call gate | 40.98% → 54.58%. Without the gate, 48.36% |
| IFEval, prompt-level strict | Qwen3-0.6B-Base, implants with no RL | 12.94% → 34.75% |
| IFEval, prompt-level strict | Qwen3-0.6B chat, 541 prompts | 313 → 390 prompts (58% → 72%), 80 gained and 3 lost |
| GSM8K after RL | 6 seeds, the implant arm with a voting stack against plain RL | 80.4% against 68.3% |
| IFEval after RL, implants against plain RL | 6 seeds, paired | +2.96 points, 95% interval −0.28 to +6.35 |
| BBH after RL, implants against a matched control | Paired | +2.62 points, 95% interval +1.99 to +3.24 |
| 4- and 5-digit multiplication | Base against the calculator | 1% and 0% → 100% |
| Calculator on word problems | GSM8K, GSM-Hard | 52% → 53%, and 27% → 35% |
| Topic blocker | Qwen3-1.7B chat | Catches 98.7%, blocks 4.2% wrongly, fires on 23.0% of MMLU |
| Scrambled implants, a control | GSM-Hard, three runs | The real implants 56.1, 56.2 and 55.2, against 51.9, 41.6 and 51.3 |

The two IFEval rows explain a mismatch that the issue noted. The 13% figure is the base model, and the 58% figure is the chat model.

**Where the claims go beyond the evidence.**

- **The headline gains need more than implants.** The BBH result uses a fine-tuned host as well as the tools and the gate. The GSM8K result after RL includes a stack that votes over several tries.
- **Some RL gains are within noise.** The IFEval gain from implants after RL has an interval that includes 0.
- **The comparison is a tool, not a fix.** The author writes "most of the gains are where a tool would help anyway". An implant mostly does what a Python tool call would do, without the call.
- **One test set was used to fit.** The first BBH pass was fit to the test set, and the post retired it. The later results are on held-out data.
- **Small models, few seeds.** The hosts are 0.6B to 1.7B parameters, with 3 to 6 RL seeds. The intervals in `REPRODUCE.md` and the statement in `DETAILS.md` that there are none disagree.
- **Interference is claimed more than shown.** The post says that the off-task benchmarks (MMLU, ARC, HellaSwag, WinoGrande and NQ) stay within a point, but we didn't see the numbers.

## Use in this repo

- **An implant is not an edit of the model's own weights.** The fix campaign's arms change the model's weights (#5220), and ROME edits change one of its matrices (`papers/rome-keep/`). An implant adds new weights beside a frozen model and leaves every existing weight as it was. So the registry's explanations of the host's neurons keep, by construction, until the implant fires. That is a different kind of fix from those in `fixing-models.md`, and closer to the external memories that it reviews.
- **Silence is the interference control.** An implant adds exactly zero when its trigger doesn't fire, so interference can come only from wrong firing. The topic blocker fires on 23.0% of MMLU, so it is not silent off-task. Interference with the host's other skills thus depends on the trigger's precision, which the registry's review measures for its own poles.
- **A VERIFIED pole as a trigger.** The topic blocker's trigger is a linear probe on the hidden states. A VERIFIED pole is a trigger that has passed hard negatives, so its false-fire rate on held-out text is known. #7356 would gate a simple implant on a VERIFIED pole and on a probe of the same concept, and compare their false fires.
- **Writing into dead neurons.** The post suggests it but doesn't try it. By the end of Pythia-70m's training almost no pole is dormant (`papers/neuron-development/`). The `neuron-family-triage` skill classes some neurons as dead on a broad corpus, but most of GPT-2's neurons are default-on, not dead (`papers/default-on-poles/`). So a small model may have few free neurons to write into.
- **Related pages.** `interpretable-weight-space-programming.md` and `weight-space-programming-without-transformers.md` compile programs into weights. `rikiyat-weight-space-programming.md` has a frontier model edit weights directly. `tracr.md` compiles programs into a whole transformer. This post compiles small machines and attaches them to a trained model, so it sits between compiling and editing.
