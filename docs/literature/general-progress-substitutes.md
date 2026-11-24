# General progress substitutes for specific training data in language models

**Citation.** Márk Marosi and Peter Antal. "General progress substitutes for specific training data in language models". arXiv 2611.24844, version 1, 18 November 2026. Preprint, no venue given. 49 pages, 16 figures and 6 tables. [Paper](https://arxiv.org/abs/2611.24844), [code](https://github.com/Mapika/progress-substitutes-data) (MIT licence). The HTML marks both authors with one affiliation, but we did not see its text. The data is public: Lichess games from 2024, ZINC-22 molecules, FineWeb-Edu and the DataDecide Dolma 1.7 recipe. Section 4.20 says that Claude, Gemini and Codex wrote training and analysis code, checked the labellers and edited the text. The authors say that they checked every reported result. We saw no funding statement.

**Status.** Planned. #6524 asks whether Pythia-70m's late checkpoints learn new data more slowly. This paper's acquisition cost is the measure that #6524 needs. It is the tokens to reach a level after a return, divided by the tokens from initialisation.

**Read.** On 2026-11-24, through WebFetch. A model read the abstract page and the v1 HTML for us, in four passes by offset. We have the source's own text for the abstract, introduction, Results 2.1 to 2.5, the discussion, Methods 4.1 to 4.22 and Appendices B to G. Two gaps of about 44 characters fall in a caption and in Methods. The tool limits each quote to 125 characters, so long passages came back as short fragments. We read the figure captions but not the 16 figures. We did not see the affiliation text, the rendered cells of Tables D2 and D3, or any funding or competing-interests statement.

## Summary

**Claim.** If a model's training withholds a skill's targets, the skill is only delayed. Training on the rest of the data lowers the background loss. The model then needs only a little of the withheld data to learn the skill. The paper's title gives the claim: general progress substitutes for specific training data.

**Method.**

- **Models.** Dense pre-norm decoder-only transformers from 0.755M to 162.8M non-embedding parameters, with CompleteP. Chess goes up to 431M, 1.28B and 3.0B. Web-text models follow the DataDecide Dolma 1.7 recipe at 20M, 60M, 300M, 1B and 7B. The 7B models stop at 3.4B and 3.9B tokens.
- **Skills.** A skill is a class of next-token targets that a rule picks. In chess it is the replies to a check, about 7.1% of tokens. Othello uses forced passes, SMILES uses ring-closing digits, and Dyck-4 uses closing brackets. Web text uses subject-verb agreement, brackets, repeated 4-grams, counting and two factual relations: country capitals and city countries.
- **Withholding.** *Target thinning* keeps a fraction f of a skill's targets in the loss, chosen by a hash of the position. The inputs never change. f = 0 is a never-supervised model, and f = 1 a fully supervised one. *Document filtering* instead removes whole games or documents, for example Rust, astronomy or Chinese in web text.
- **Metrics.** Competence q is a probability, such as the share of legal-move mass at in-check positions. Background loss is the held-out loss on every target outside the withheld set. *Onset* is the background loss at which q crosses a level. *Acquisition cost* is the tokens to reach a level after the withheld data returns, divided by the tokens that a same-size model needs from initialisation.
- **Controls.** Removing an equal share of random targets. Giving each in-check target a loss weight of 0.1 against keeping 10% at full weight. A permuted rule, which needs about as many tokens from initialisation as the true rule (within a factor of 1.06). Probes are scored against the same probes on a random network.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Chess, 9.7M, 10% of in-check targets kept | Onset at q = 0.70 moves by 0.40 nats of background loss |
| Chess, 9.7M to 3.0B, 10% kept | The delay is 0.29 to 0.49 nats, and changes by 0.00 to 0.03 nats for each tenfold in size |
| Chess, a filter of games with a check, against thinning to the same share | The filter gives 78 to 87% of the delay |
| Chess, 64M, never-supervised | 0.48 of the probability goes to moves that ignore the check, against 0.02 when fully supervised |
| Chess, source models at q = 0.32 to 0.50 | Acquisition needs 0.10 to 0.78% of the tokens from initialisation, from each run's lowest-loss checkpoint. The authors call these upper estimates |
| Cost against surplus, 312 runs from 27 source runs | Chess 1.12 ± 0.04, web text 0.91 ± 0.04 and SMILES 0.59 ± 0.07 decades of cost for each logit |
| At the acquisition loss | 30%, 24% and 38% of the tokens from initialisation, in chess, web text and SMILES |
| A permuted rule, from the same checkpoints | 2.4 to 16 times the true rule's tokens |
| Web text, 7B, country capitals, last three checkpoints, two seeds | 1.1% to 22.3% of the reference's tokens. Earlier checkpoints don't reach the level within 592M tokens |
| Web text, 7B, Rust filtered to 3%, one run | Half the gap to the unfiltered model closes after 2.1% of its tokens from initialisation, nine tenths after 13% |
| Chess, unlearning | Models whose skill was removed by gradient ascent reach q = 0.81 to 0.91 within 12 steps. Never-supervised models move by less than 0.03. Never-supervised models need 5 to 20 times more tokens |

## Where the claims go beyond the evidence

**Most skills are narrow and synthetic.** Chess, Othello, SMILES and Dyck have exact rules. Competence is a probability ratio, not a task score, except Rust's compile rate. The web-text skills are short probes of agreement, brackets and two factual relations.

**The 7B evidence is thin.** The capital test uses one retained fraction and two seeds, and the two seeds differ by 0.31 logits. The Rust filter is a single run. The authors say "The test covers one retained fraction" and "The 7B costs are not in the fit".

**The costs are upper estimates against a chosen denominator.** Each cost divides by a same-size run from initialisation, sometimes with another seed. Appendices D.3 and F.1 call several costs upper estimates. A different recipe for the run from initialisation would change every ratio.

**The derived delay uses a measured input.** The claim of "no fitted parameter" for 7B takes the corpus support from a never-supervised model of the same seed. It is a prediction from a second model, not from first principles.

**Seeds are few and no test adjusts for multiple comparisons.** Many results rest on one to three seeds, and several figures on one run. We saw no multiple-comparison adjustment across the many settings.

**Credit.** The controls are strong. The random-mask and permuted-rule controls show that the delay is specific to the skill's targets. Probes are scored against a random network. The code and data are public, and the tool use is disclosed.

## Use in this repo

**It gives #6524 its measure.** `papers/neuron-development/` finds that layers 2 to 5 of Pythia-70m lose effective rank after step 4,000. #6524 asks whether the late checkpoints learn new data more slowly. This paper's acquisition cost, divided by the tokens from initialisation, makes those checkpoints comparable. Its permuted-rule control separates learning a rule from fitting arbitrary labels.

**It warns against reading slower learning as lost plasticity.** In this paper a later checkpoint learns a withheld skill faster, because its background loss is lower. So a late Pythia checkpoint may learn new data faster although its rank is lower. #6524 must compare checkpoints at the same background loss, or report both effects.

**It fits the savings in the development paper.** `papers/neuron-development/`, Table 3, finds that 9 of 13 explanations that fail after step 16,000 pass again before the end. It calls the return of L4/N216+ closer to savings than to forgetting. This paper shows a skill returning fast once general progress has prepared for it. It studies skills and probes, not single neurons, so it can't say whether a pole's return works the same way.

**It strengthens the relearning warning for fixes.** In chess, a gradient-ascent unlearning comes back within 12 steps, but a model that never saw the skill doesn't. That agrees with `distillation-robustifies-unlearning.md`. Its question for the maintainer, whether a weight-space fix should report a relearning test, applies here too. The fix campaign (#5220) is the maintainer's, so this page opens no issue for it.

**No neuron result to use.** The paper's internal analyses use residual-stream probes, a difference-of-means direction and a rank-8 subspace. It has no single-unit analysis. Its random-network probe baseline matches the repo's untrained-model null (`docs/analyses/untrained-null.md`).
