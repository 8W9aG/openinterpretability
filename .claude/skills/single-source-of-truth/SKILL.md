---
name: single-source-of-truth
description: Keep one definition for each constant, threshold, list of names or rule, and import or read it everywhere else. Use before adding a literal (a number, a status, a model name, a path, a class name) to code, templates, docs or workflows, when changing a value that may be copied elsewhere, and when reviewing a diff.
---

# Single source of truth

Every concept in this repo has one definition. Everything else imports it, reads it, or is pinned to it by a test. A value typed twice drifts: someone changes one copy, CI stays green, and the site, the docs or the decision rule quietly disagree.

## Before you add a literal

1. **Name the concept**, not the value. "The active line" is the concept; `0.25` is the value.
2. **Search for the value and the concept** across everything, not only Python:

   ```bash
   git grep -n -e '0\.25' -e 'activation_threshold' -e 'active line' -- \
     openinterp tests docs .claude .github scripts Makefile pyproject.toml README.md
   ```

   Search the spellings a copy could use: the number (`0.25`, `.25`, `25%`), the identifier (`activation_threshold`), and the words in prose and comments. For a list of names (statuses, gates, models), search for two of its members together.
3. **If a definition exists, use it.** If only copies exist, make one the source (see below) and point the others at it in the same change.
4. **If nothing exists**, define it once in the place below that fits and import it.

## Where definitions live

| Concept | Source |
| --- | --- |
| Protocol values: thresholds, gate defaults, verdict statuses, probe kinds, `PROTOCOL_VERSION` | `openinterp/core/schemas.py` (`ReviewPolicy`, `VerdictStatus`, `ACTIVATION_THRESHOLD`, …) |
| The wiki directory's default and layout (`experiments/`, `draws/`) | `openinterp/core/paths.py` (`DEFAULT_WIKI_DIR`, `experiments_dir`, `draws_dir`); CLI commands take `wiki_dir_option()` from `openinterp/cli.py` |
| Planner: base weights, ε, windows per model | `openinterp/planner.py` (`BASE_WEIGHTS`, `DEFAULT_EPSILON`, `configured_windows`) |
| Reproduction tolerances | `openinterp/reproduce.py` (`Tolerance`) |
| Models the planner and CI may use | `[tool.openinterp] models` in `pyproject.toml` |
| Site-wide display maps: unit statuses, their order and colours | `openinterp/builder/statuses.py` (`MAP_STATUSES`, `STATUS_FILL`, `STATUS_BADGE`) |
| Evidence strength bands (strong / moderate / weak) | `openinterp/builder/significance.py` (`STRONG_EVIDENCE`, `evidence_strength`) |
| Standing roles and `CONTESTED` | `openinterp/builder/standing.py` |
| Canonical agent model names | `openinterp/core/agent_names.py` |
| Corpus sampling defaults | `openinterp/corpus.py` |

A value used by one module only stays a module-level constant there, with a comment saying what it is. Move it to a shared place when a second module needs it, not before.

A policy default that a record also stores (`verdict.policy`) is read from the record when one is at hand, and from the default (`ReviewPolicy()` or the named constant) otherwise. Never retype it.

## Templates

Jinja templates can't import Python, so pass values in; never retype a Python list or map in a template.

- Site-wide values go in `env.globals` in `openinterp/builder/compiler.py` (`_environment`). Whole modules can be globals: `statuses` is, so a template reads `statuses.STATUS_FILL` with no per-constant wiring. Functions work too: `evidence_strength(c)` is a global, and `_macros.html` calls it.
- Page-specific values go in the view the page renders from.
- Macros imported with `{% from "_macros.html" import … %}` see globals, so a macro can read `statuses.STATUS_BADGE[status]` directly.
- Tailwind runs from the CDN and scans the rendered DOM, so class names may live in Python.
- SQL in `registry_stats.py` is a Python string: make it an f-string and interpolate the constant (`{STRONG_EVIDENCE}`), as the conclusive-null count does.

## When a copy has to exist

Some copies can't import their source:

- **Workflows.** `ALLOWED_MODELS` in `.github/workflows/ci.yml` copies the `pyproject.toml` models on purpose, so a fork PR can't widen what CI downloads.
- **Docs.** `docs/PROTOCOL.md`, the README and skills quote numbers so a reader doesn't need the code.
- **Rendered output and fixtures** that state an expected value.

For each one:

1. Add a test that the copy agrees with the source. Put it in `tests/test_single_source.py` unless an existing test module owns the concept (`tests/test_planner.py::test_configured_models_match_ci_allow_list` checks `ALLOWED_MODELS`). Format the value from the source in the test, and search for that text in the copy, so the test fails when either side moves.
2. Name the source at each copy: a comment (e.g. `<!-- Defaults quote ReviewPolicy in core/schemas.py; tests/test_single_source.py checks them. -->`), or in the prose itself when that reads naturally ("`BASE_WEIGHTS` in `openinterp/planner.py`"). Skip it when the text already points there (an env var or config key next to the number).
3. Check the test catches drift: change the copy, see it fail, change it back.

Docs, skills and agent files quoting a number go in the `_quotes()` list in `tests/test_single_source.py`. Numbers that can't drift need no test: maths ("0.5 is chance", the median percentile), and the protocol version a feature arrived in ("predates protocol 0.3").

Prose that explains a value in passing ("the rejection line, balanced accuracy 0.6") in a docstring is fine when the docstring also names the source. Don't add a test for every passing mention; add one where a reader would act on the number.

## Changing a value

1. Change the source.
2. Search for the old value and the concept's names (as above) to find copies the tests don't cover, and update them.
3. A protocol default is a decision-rule change: bump `PROTOCOL_VERSION` and add a `docs/PROTOCOL.md` §6 entry.

## Reviewing a diff

Flag:

- a new literal that equals an existing constant, or a new list whose members match an existing list;
- a template `{% set … = { … } %}` map keyed by values that exist in Python;
- a number in docs or a workflow with no test pinning it and no comment naming its source;
- a source changed without its copies.

Suggest the import or global to use, not only "this is duplicated".
