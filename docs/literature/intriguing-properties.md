# Intriguing Properties of Neural Networks

**Citation.** Seven authors: Christian Szegedy (Google), Wojciech Zaremba (New York University), Ilya Sutskever (Google) and Joan Bruna (New York University). Then Dumitru Erhan (Google), Ian Goodfellow (University of Montreal) and Rob Fergus (New York University and Facebook). "Intriguing properties of neural networks". *ICLR 2014*. [arXiv:1312.6199](https://arxiv.org/abs/1312.6199), version 1 of 21 December 2013, last revised as version 4 on 19 February 2014. The arXiv page names no venue. ICLR 2014 comes from DBLP's record.

**Mismatch.** The abstract on the arXiv page has two typos that the HTML of version 4 corrects ("significant extend", "contains of the semantic information"). Quotes on this page are from the HTML.

**Status.** Used. It is the classic claim that a random direction reads as well as a single unit, which is the question the registry exists to test. `papers/default-on-poles` tests that claim on the low tail of GPT-2's neurons.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 4 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words: Sections 1 to 5 and the references. The PDF came back as bytes that it couldn't read. Figures 1 to 7 are images, so their content is known from their captions only.

## Summary

**Claim.** Two properties. First, "there is no distinction between individual high level units and random linear combinations of high level units". The meaning lives in the space, not in single units. Second, small, hard-to-see perturbations change a network's output, and they transfer between networks.

**Method.**

- **Units against directions (Section 3).** For a unit, take the held-out images x that maximise ⟨φ(x), eᵢ⟩, where eᵢ is the unit's basis vector. For a direction, take those that maximise ⟨φ(x), v⟩ for a random v. Compare the two sets of images by eye.
- **Networks.** A convolutional network on MNIST, with the MNIST test set, and AlexNet on ImageNet, with its validation set. Section 3 says only "some layer". The introduction refers to "the last feature layer".
- **Adversarial examples (Section 4).** Find the smallest perturbation r that changes the label, with box-constrained L-BFGS. Test whether the perturbed images fool other networks.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Units against random directions (Section 3, Figures 1–4) | "any random direction v∈ℝ^n gives rise to similarly interpretable semantic properties". Shown by 4 units and 4 directions per dataset |
| AlexNet adversarial examples (Figure 5) | Mean distortion 0.006508 over 64 examples, "strictly randomly chosen" |
| Transfer between MNIST models (Table 2) | Misclassification 2% to 48.1%, against 0% to 18.3% for Gaussian noise of standard deviation 0.1 |
| Training on adversarial examples (Section 4.2) | Test error below 1.2%, against 1.6% with weight decay. Called "preliminary" |

## Where the claims go beyond the evidence

- **Section 3 is judged by eye.** It rests on four units and four directions per dataset, with no measure, no sample of directions and no statistics. The text doesn't say how v was drawn or how many were tried.
- **One layer, two image networks.** The claim is general, but the evidence is the last feature layer of two vision models.
- **"No distinction" is stronger than "similarly interpretable".** The abstract says there is no distinction. Section 3 says random directions are "similarly interpretable". Neither is measured.
- **The adversarial bounds are upper bounds.** The authors say that large bounds "do not automatically translate into existence of adversarial examples".

## Use in this repo

- **Our citations check out.** `papers/default-on-poles` quotes the paper twice.
  - "no distinction between individual high level units and random linear combinations of high level units" is in the abstract. The paper cites it with no section, which fits.
  - "any random direction" and "similarly interpretable semantic properties" are in one sentence of Section 3, as the paper says.
  - The paper says the result is for image networks, which is right. The reference's seven authors and ICLR 2014 match.
- **The registry measures what Section 3 judged by eye.** The registry compares a unit with matched random directions, with a number on each side. Examples are the random null of each verdict and the [untrained null](../analyses/untrained-null.md). The default-on-poles paper finds that a random direction's low tail is as reliable as a neuron's (0.68 against 0.70). That supports Szegedy et al.'s claim for that tail. Its positive side is a different case.
- **The registry's premise is the opposite bet.** The registry audits native neurons because they may be cleaner than random directions in the residual stream or MLP. Section 3 is the main published reason to doubt that, and it was never measured. Each VERIFIED pole that beats its random null is evidence against Section 3, for that pole.
- **No question for the maintainer.**
- **Related pages.** [`neurons-as-an-interpretable-basis.md`](neurons-as-an-interpretable-basis.md) argues the other way for language models. [`toy-models-of-superposition.md`](toy-models-of-superposition.md) explains when a basis is privileged.
