- **`planner_states.py` joins the verdict code that `make mutation` mutates.** Its fast tests, `tests/test_planner_states.py`, pin these:
  - the skip of records whose evidence is missing.
  - the record that a unit keeps when a later one is set aside.
  - the count of independent reviews.

  All 2,113 mutants of the seven files die, in about 3 minutes (#6810).
