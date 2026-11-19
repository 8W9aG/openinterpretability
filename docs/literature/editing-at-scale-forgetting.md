# Model Editing at Scale leads to Gradual and Catastrophic Forgetting

**Citation.** Akshat Gupta, Anurag Rao and Gopala Anumanchipalli, UC Berkeley. "Model Editing at Scale leads to Gradual and Catastrophic Forgetting". arXiv 2401.07453. Version 1 is from 15 January 2024, and we read version 4, from 10 June 2024. [Paper](https://arxiv.org/abs/2401.07453). The arXiv comment gives the venue as "ACL 2024 Findings", and the arXiv record has no journal reference. We did not check the ACL Anthology page. The PDF links to [code](https://github.com/scalable-model-editing/gradual-catastrophic-forgetting) and a [project page](https://scalable-model-editing.github.io/catastrophic). We did not open either link. The edits use the MEMIT authors' code (`kmeng01/memit`). The text we read names no funder and gives no tool disclosure.

**Status.** Planned. #5167 applies one ROME edit to pythia-70m and tests which VERIFIED poles keep. This paper's sequential protocol and its weight-distance measure are the natural next step for that test. #5230, the interference report, needs the matched-norm null that this paper does not have.

**Read.** On 2026-11-19, the reviewing model (Claude Opus 5.5) downloaded the v4 PDF from arXiv and converted it with `pdftotext`. It read all 31 pages of text itself, and no other model read the paper for it. We have the source's own text for the abstract, sections 1 to 7, and appendices A.1 to A.6, with Tables 1 to 8 as text. We looked at pages 4 and 5 as low-resolution images, which hold Figures 1 and 2. We did not see Figures 3 to 36 as images, only their captions. So this page gives no number that only a figure holds. We read the arXiv abstract page for the version history.

## Summary

**Claim.** Editors must make many edits to one model in sequence to be useful. Under sequential editing, ROME and MEMIT go through two phases. First the model slowly forgets earlier edits and loses downstream skill. Then one edit makes it forget almost everything at once. The authors trace both phases to the edited layer's drift from its original weights.

**Method.**

- **Editors.** ROME and MEMIT, with MEND and a constrained fine-tune (FT-C) as baselines. All use the default code and settings of the MEMIT authors.
- **Edit layers.** ROME and FT-C edit layer 18 of GPT2-XL and layer 6 of GPT-J. MEND edits layers 46 to 48 and 26 to 28. FT-C bounds the change of each parameter at each step to 5e-4 on GPT2-XL and 5e-5 on GPT-J.
- **Models.** GPT2-XL (1.5B) and GPT-J (6B), both base models.
- **Data.** CounterFact. The authors drop zsRE, because a fact edited in question form often does not appear in plain text completion. They draw four random subsets of 1,000 facts, with no subject edited twice. MEMIT on GPT-J uses a subset of 2,000 facts that continues sample 1.
- **Order.** One fact per edit, in sequence. MEMIT also edits one fact at a time, and the paper leaves batched edits for later work.
- **Editing proficiency.** Efficacy (ES), paraphrase (PS) and neighbourhood (NS) scores, as in the ROME paper. The curves show means over the last 5 and the last 50 edits.
- **Forgetting.** The share of earlier, successful edits that the model no longer recalls. The authors measure it after every 10 edits.
- **Downstream skill.** Few-shot SST-2, MRPC, NLI and CoLA. Each task uses a balanced subset of 200 test examples and the F1 score, measured about every 10 edits.
- **Source of forgetting.** The L2 distance between the edited layer and its original weights, divided by the number of neurons in the layer.

**Findings, as the source reports them.** All numbers are from the text and the tables.

| Setting | Result |
| --- | --- |
| zsRE edits in question form, GPT2-XL with ROME, then a completion prompt | the model does not give the edited answer about 70% of the time |
| ROME, efficacy under sequential edits | almost 100% until a point of decline. That point comes as early as 100 edits (sample 3) or as late as 1,000 (sample 2) |
| ROME, neighbourhood score before that point | falls steadily as edits accumulate |
| ROME, forgetting | a slow, gradual rise, then a sudden jump at the same point where efficacy starts to fall |
| ROME, downstream F1 | a gradual fall, then F1 drops to zero at the same point |
| ROME, mean normalised distance of one edit made first (Table 2) | 3.339 × 10⁻⁴ for a disabling edit, 8.156 × 10⁻⁷ for a normal edit |
| ROME, a disabling fact edited first | the model is still disabled |
| MEMIT, GPT-J, sample 1 | catastrophic forgetting starts after about 1,400 edits. Efficacy is below ROME's |
| MEMIT against ROME, forgetting (Figure 4 text) | MEMIT forgets "almost three times fewer facts" |
| MEMIT, all samples (appendix A.4) | the catastrophic point comes before 1,500 edits on GPT-J and near 4,000 on GPT2-XL |
| MEMIT, a disabling fact edited first | no catastrophic forgetting |
| MEMIT, downstream F1 | kept on GPT-J. On GPT2-XL it falls gradually, and MRPC is lost long before the catastrophic point |
| MEND | forgets all earlier edits, and F1 falls to zero, before 100 edits |
| FT-C | edits succeed, but the model keeps only a few earlier edits. Gradual forgetting only, with no catastrophic point |

Table 8 shows GPT-J text after the catastrophic point under ROME. One prompt gives "CBS" repeated. Another gives a run of commas and function words.

The authors state these limits. They test only two model sizes, 1.5B and 6B. They do not test batched edits or sequences of batches.

## Where the claims go beyond the evidence

**The drift explanation is a correlation, not a test.** The paper says the edited layer becomes "incompatible" with the rest of the model. It shows only that the weight distance rises with forgetting and jumps at the same edit. No experiment reads how later layers respond to the edited layer's output.

**No matched random change.** The paper never applies a random update of the same norm. So it cannot say whether the damage comes from the edit's direction or from its size alone.

**"Three orders of magnitude" is closer to two and a half.** Table 2 gives 3.339 × 10⁻⁴ against 8.156 × 10⁻⁷, a ratio of about 409. Table 2 also does not say how many disabling or normal edits it averages.

**One fact per MEMIT step understates MEMIT.** MEMIT is a batch method, and the paper runs it one fact at a time. Table 3 then ranks MEMIT above ROME on every property but efficacy. Neither ranking covers MEMIT's batch use.

**The downstream test is small and has no intervals.** Each task has 200 examples and one F1 score per checkpoint. No curve has an interval, and there are four samples of edit order. The gap between samples is large, as the ROME decline point ranges from 100 to 1,000 edits.

**The authors check forgetting every 10 edits.** So "a single update" that disables the model rests on Table 2's first-edit test, not on the curves.

**Small slips in the text.**

- Section 3.4 points to appendix A.3 for the inflection point in all samples, but A.3 describes the downstream tasks.
- Figure 2's caption says "GPT2-J".
- Appendix A.2 bounds "the norm of the gradient update", and A.6.4 bounds each parameter change with an L∞ limit.
- Section 3.2 says the model forgets facts "with one gradient update". ROME's weight update is in closed form.
- Section 2 still says a survey "will be added to the main paper upon acceptance".

The authors give themselves credit for one limit, and so do we: they say that larger models and batched edits may behave differently.

## Use in this repo

**The paper studies no units.** It measures recall of edited facts, neighbourhood facts and four tasks. It does not read neurons, heads or explanations. So it gives no direct evidence on whether a VERIFIED explanation keeps under editing. That question is #4782's, and #5167 is its first test.

**In pythia-70m, one-layer edits leave the lower poles unchanged, at any edit count.** The [ROME page](locating-editing-factual-associations.md) shows why. ROME changes only the edit layer's `W_out`, and pythia-70m has a parallel residual. So poles at or below the edit layer fire as before, on every input, after any number of edits there. That holds even past a disabling edit. The 86 VERIFIED poles are 16, 18, 9, 15, 15 and 13 in layers 0 to 5 (`papers/neuron-development/README.md`). So an edit at layer 2 leaves the activations of 43 poles as they were, and 43 can change. This is our inference from the method, and the paper does not discuss it.

**That makes the drift explanation testable here.** If the edited layer becomes unreadable to later layers, then poles above the edit layer should flip near the jump in distance. Poles at or below it cannot flip at all. The paper could not test this, because it reads no units. We propose a follow-up issue, below.

**Weight distance alone does not mean damage.** In `papers/neuron-development/README.md`, the 86 poles keep changing late in training. Adjacent checkpoints correlate at a median of only 0.78 to 0.82, and the model keeps its loss. Training moves all layers together. A one-layer edit moves one layer while the rest stays fixed, which is the paper's own account of the difference. So the paper's distance is a useful early warning for a one-layer edit, not a measure of harm.

**The repo's keep test reads more than the paper's scores.** `scripts/base_instruct.py` re-runs each VERIFIED pole's own probes on a changed model. In `docs/analyses/base-instruct.md`, 38 of 43 Qwen2.5-1.5B poles keep under instruction tuning (88%). On Qwen2.5-0.5B, 252 of 311 keep (81%), and 70–72% keep when the probes are inside the chat template. The paper finds that downstream skill falls before the edit metrics move. A unit-level test like this one could show early damage that the paper's scores miss. #5230 makes this keep test general.

**The paper supports #5230's matched-norm null.** #5230 counts a flip only above a matched random change of the same norm. The paper shows why the report needs that null. A single ROME edit can be about 409 times larger than a typical one, and it disables the model. Without the null, such a flip could come from size, not from the edit's direction. This is evidence for #5230. The report itself is part of the maintainer's campaign.

**Sequential editing is the non-interleaved case.** The [complementary learning systems page](complementary-learning-systems.md) records that focused learning of one item damages related knowledge, and interleaving avoids it. Here each fact goes in alone, one after another, into one layer. MEMIT spreads each edit over several layers and forgets less. The paper does not frame its result this way. The connection is ours.

**Evidence for the fix campaign, not a design.** #5220 arm F names ROME and MEMIT as a closed-form baseline. Three points from this paper bear on it:

- One ROME edit can disable a model, and the edit's weight distance flags it.
- The paper tests models of 1.5B and 6B, not a six-layer model like pythia-70m.
- The #4781 shortlist has 6 kept prompts, far fewer than the 100 to 1,000 edits where ROME broke. The paper gives no number for small counts that we read.

**What would change.** One follow-up issue and two comments. Both comments add evidence only:

- On #5167: record each edit's normalised weight distance. One outlier edit can disable a model, so a pole flip after such an edit says nothing about |u_j|.
- On #5230: the paper has no random change of the same norm. That supports the matched-norm noise floor in #5230.

**Proposed follow-up.** "How do VERIFIED pythia-70m poles fail as rank-one edits accumulate at one layer?" Make a sequence of ROME edits of unrelated facts at one layer of pythia-70m. After every k edits, run each VERIFIED pole's own probes again, and record the layer's normalised weight distance. Poles at or below the edit layer are a built-in check. Test whether poles above it flip gradually, at the jump in distance, or not at all. Use a random update of matched norm as the null. Area: research questions. Priority: medium. It extends #5167, is part of #4782, edits no fact that the fix campaign targets, and fixes nothing.

**Question for the maintainer.** Should the interference report record the edit layer's normalised weight distance next to its flips? The paper suggests that it gives an early warning, but this choice belongs to the campaign. So this page records it and opens no issue.

**Related reviews.** The methods are on the [ROME](locating-editing-factual-associations.md) and [MEMIT](memit.md) pages, and the [editing survey](editing-llms-survey.md) gives the field's metrics. Collapse after 1,000 sequential edits under free decoding is on the [Mirage](mirage-of-model-editing.md) page. Failures on implied facts are on the [Ripple Effects](ripple-effects-of-knowledge-editing.md) page. Editors that keep sequential edits out of the main weights are on the [GRACE](aging-with-grace.md) and [WISE](wise-lifelong-editing.md) pages. The synthesis for the #4781 " days" error is on the [fixing-models page](fixing-models.md).
