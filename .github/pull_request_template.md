## Summary

<!-- What does this PR change? -->

## Experiment checklist

<!-- Delete this section if the PR posts no experiment. -->

- [ ] The experiment's record (`wiki/experiments/<slug>/<id>.json`) was produced by `open-interpretability verify` or `open-interpretability reevaluate`.
- [ ] The wiki page was generated (`open-interpretability wiki`), and I only edited its **Discussion** section by hand.
- [ ] I've said whether the proposer and reviewer were different models (`independent_review`, recorded but not gated).
- [ ] `open-interpretability check` passes locally.
- [ ] If the experiment came from `open-interpretability next`, its seed and kind are in this description.

CI checks every PR in two jobs:

- **Lint and tests:** `ruff check`, `ruff format --check` and `pytest`. Skipped when every change is under `wiki/`.
- **Experiment records:** every experiment record matches the schema, has a verdict and has a matching page, and every page has a record. Records are write-once, generated indexes (`README.md`) are not committed, and existing experiment pages were not edited outside Discussion or deleted.
