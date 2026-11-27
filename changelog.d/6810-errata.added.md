- **`core/errata.py` joins the verdict code that `make mutation` mutates.** Its fast edge tests, `tests/test_errata_edges.py`, pin these:
  - the harness, protocol and date rules, and a record with no verdict.
  - edits that end at a list index, and edits applied in order.
  - set-aside records, and the reading of the errata file.

  All 2,062 mutants of the six files die, in about 3.5 minutes (#6810).
