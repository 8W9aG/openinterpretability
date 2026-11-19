# Fixing models

This page sums up 40 reviews of how to fix a model's behaviour (#4784). It reports what the papers found: which kinds of fix held up, where each failed, and how the papers tested a fix. It does not design the fix campaign. The maintainer designs that campaign.

The reviews are in four sections of the [literature index](README.md). The sections are editing, wrong answers, fixes guided by an explanation, and activation steering. Each page reads its source and says what it did not read. This page cites the pages, not the papers, so each number here has a source page to check.

## The case that the reviews keep in view

EleutherAI/pythia-70m answers " days" to "A week has seven days and a year has twelve" (p 0.7257). The right answer, " months", is at rank 4 (p 0.0295). A VERIFIED answer route explains the failure (#4781). Layer-0 count-word and time-unit neurons mark " seven", " days" and " twelve". Early heads line up the two clauses. Then the copy heads L3/H3, L4/H0 and L4/H7 carry " days" to the last position. No verified unit supplies " months". The other five prompts in `docs/analyses/hallucination-shortlist.jsonl` show the same copy failure. Three more shortlist prompts are right because the model copies the unit: " legs" (0.5494), " wheels" (0.4149) and " legs" again (0.5178).

## Why models give wrong answers like this one

The area 4 papers expect this failure in a small model, and each gives a way to read it.

- **Small models take the answer from the context.** In Pythia and GPT-2, some heads push the answer in the context and others push the memorised one. Smaller models take the context answer more often ([Characterizing Mechanisms for Factual Recall](factual-recall-mechanisms.md)).
- **The copy is a fuzzy induction match.** The count-word neurons make " seven" like " twelve", and the copy heads then copy " days". The induction-heads paper names this cost: a copy is wrong when the true continuation differs ([Induction Heads](induction-heads-in-context-learning.md)). Head scores on 32-token repeats are 0.33 for L4/H7, 0.13 for L3/H3 and 0.05 for L4/H0.
- **The one check against copying is too weak.** A quick check found one copy-suppression head in pythia-70m, L4/H1. On this prompt it lowers " days" by only 0.31 logits, against +1.43 from L3/H3. No head in layer 5 attends to " days" ([Copy Suppression](copy-suppression-head.md)). This is one prompt and not a verified claim.
- **The fact may be absent or outvoted.** Several papers separate a memory that is absent from one that is present but loses. A feed-forward layer rarely changes a prediction that is already strong ([Key-Value Memories](ffn-key-value-memories.md)). The open question is in #5015, with a cheap test from each paper.

## Which fixes held up, and where each failed

**Edits to weights.** Single-fact editors reach high success on the edited prompt. They fail on what the fact implies and on near neighbours:

- Editors that recall over 88% of edited facts answer only 0.3–16.0% of the multi-hop questions that follow ([MQuAKE](mquake.md)).
- MEND, ROME and MEMIT fail most ripple tests, and an in-context baseline does best ([Ripple Effects](ripple-effects-of-knowledge-editing.md)).
- Under free decoding, Llama-2 single-edit success is 38.5%, and sequential editing collapses by 1,000 edits ([The Mirage of Model Editing](mirage-of-model-editing.md)).

**The premise of an explained fix.** Hase et al. found that Causal Tracing does not predict the best layer to edit. The edit layer explains 94.7% of the variance in ROME's success, and tracing adds 0.1% ([Does Localization Inform Editing?](does-localization-inform-editing.md)). Mechanistic Unlearning tests the premise again. Edits at the fact-lookup MLPs hold better than edits placed by Causal Tracing, but editing all MLPs is still a strong baseline ([Mechanistic Unlearning](mechanistic-unlearning.md)). So where a localization comes from matters.

**Fixes that remove.** Ablating features judged irrelevant raised Pythia worst-group accuracy from 24.4% to 76.0% ([Sparse Feature Circuits](sparse-feature-circuits.md)). Circuit breakers stop a behaviour but give no right answer: the model writes meaningless text after the break ([Circuit Breakers](circuit-breakers.md)). A fix that removes a signal can take " days" away, but nothing in these papers shows that " months" then wins.

**Suppression comes back.** Several papers show that a fix can hide a behaviour without removing it:

- In-context examples, a small relearn and linear probes bring unlearned knowledge back ([Eight Methods to Evaluate Unlearning](robust-unlearning-evaluation.md)).
- 500 fine-tuning steps restore a behaviour after unlearning by training. Distilling into a fresh model removes it ([Distillation Robustifies Unlearning](distillation-robustifies-unlearning.md)).
- RMU's near-topic test shows damage that the overall MMLU score hides ([RMU](wmdp-rmu-unlearning.md)).

**Steering.** Steering moves behaviour, but the strongest studies find a simple prompt matches or beats it:

- On 500 concepts, prompting steers best (0.894) and SAE steering scores 0.165 ([AxBench](axbench.md)).
- None of 23 steering methods beats a prompt on effect against side effects ([Does Steering Break Your Model?](does-steering-break-your-model.md)).
- A steering vector moves almost half the inputs the wrong way on many datasets ([Steering-Vector Reliability](steering-vector-reliability.md)).
- A vector fitted to one sentence can make a model write it at almost any middle layer. So a steering success does not locate a cause ([Latent Steering Vectors](latent-steering-vectors.md)).
- The direction that correlates best with a concept changed behaviour least ([Representation Engineering](representation-engineering.md)).

## How the papers tested a fix

The reviews agree on what a fix report needs beyond its own prompt. These are the tests that the papers used. Which of them the campaign adopts is the maintainer's decision.

| Test | What it shows | Papers |
| --- | --- | --- |
| Paraphrases and implied facts | The fix reaches what the fact implies | [MQuAKE](mquake.md), [Ripple Effects](ripple-effects-of-knowledge-editing.md), [KnowledgeEditor](editing-factual-knowledge.md) |
| Hard negatives on the same route | The fix does not break right answers that use the same mechanism | [SERAC](memory-based-model-editing.md), [MEND](fast-model-editing.md), [Steering Specificity](steering-specificity.md) |
| Free decoding | The fix gives right text, not only a right top token | [The Mirage of Model Editing](mirage-of-model-editing.md) |
| Relearning and in-context attacks | The fix removed the behaviour and did not only hide it | [Eight Methods](robust-unlearning-evaluation.md), [Distillation](distillation-robustifies-unlearning.md) |
| Probes after the fix | The old answer is no longer readable from the activations | [RMU](wmdp-rmu-unlearning.md), [Eight Methods](robust-unlearning-evaluation.md) |
| A capability or fluency score | The fix costs little elsewhere | [Persona Vectors](persona-vectors.md), [AxBench](axbench.md), [Representation Surgery](affine-steering.md) |
| A matched random null | The effect is more than any change of that size | [ActAdd](activation-addition.md), [Concept Ablation Fine-Tuning](concept-ablation-fine-tuning.md) |
| A prompt or no-explanation baseline | An explanation adds something over a simple fix | [AxBench](axbench.md), [GRACE](aging-with-grace.md), [Ripple Effects](ripple-effects-of-knowledge-editing.md) |

Most of the papers report no seeds or intervals. Several have text that disagrees with their own tables. Each review lists these problems.

## What this repo already has

- **A matched random null.** The repo's answer route patches 0.75 against −0.004 for matched random sets, and coverage is 93.8% against 10.1%. Several papers lack a null like this one.
- **The hard negatives.** The shortlist holds the prompts where copying is right. Several reviews name them as the test that a #4781 fix must pass.
- **What it lacks.** The steering samples have no random control (#5069), no fluency score (#5070) and no head steering (#5071). Answer records score one token and store no free-decoded text. The repo has no fine-tuning or weight-edit code.

## Follow-ups

- **Evidence for #5015.** #5063 asks whether L4/H1 is a copy-suppression head. #5074 asks whether the route's units separate wrong-copy prompts from right-copy prompts. #5060 tests why the copy heads attend to " days". #5061 tracks the copy over training checkpoints.
- **Tools.** #5059 adds attention-edge knockout. #5062 adds an OV vocabulary lens for heads. #5065 adds a dictionary preset for Pythia-70M. #5066 adds a gradient-routed positive control. #5069–#5073 add steering controls and checks.
- **Questions for the maintainer.** The reviews record questions on their pages and open no issues for them. One example is which tests a fix must pass. Another is whether a fix counts when it stops a wrong answer but gives no right one. A third is whether a direction counts as a unit for a fix.
