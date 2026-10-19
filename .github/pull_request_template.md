## Summary

<!-- What does this PR change? -->

## Experiment checklist

<!-- Delete this section if the PR posts no experiment. -->

- [ ] The audit record is in `audits/` and was produced by `openinterp verify` or `openinterp reevaluate`.
- [ ] The wiki page was generated (`openinterp wiki`), and I only edited its **Discussion** section by hand.
- [ ] The proposer and reviewer were different models (`independent_review` gate).
- [ ] `openinterp check` passes locally.
- [ ] If the experiment came from `openinterp next`, its seed and kind are in this description.

CI checks every PR in two jobs:
- **Lint and tests:** `ruff check`, `ruff format --check` and `pytest`.
- **Experiment records:** every audit record validates and has a verdict, is the only record for its neuron, and has a wiki page for its verdict. Every wiki page has an audit record, the wiki indexes are up to date, and existing experiment pages were not edited outside Discussion or deleted.
