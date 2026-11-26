# This project's sources

The `single-source-of-truth` skill reads this file. It is this project's part of the skill (#6631).

## The search

```bash
git grep -n -e '0\.25' -e 'activation_threshold' -e 'active line' -- \
  openinterpretability tests docs .claude .github scripts Makefile pyproject.toml README.md
```

"The active line" is a concept, and `0.25` is its value.

## Where definitions live

| Concept | Source |
| --- | --- |
| Protocol values: thresholds, gate defaults, verdict statuses, probe kinds, `PROTOCOL_VERSION` | `openinterpretability/core/schemas.py` (`ReviewPolicy`, `VerdictStatus`, `ACTIVATION_THRESHOLD`, …) |
| The wiki directory's default and layout (`experiments/`, `draws/`, `stops/`) | `openinterpretability/core/paths.py` (`DEFAULT_WIKI_DIR`, `experiments_dir`, `draws_dir`, `stops_dir`). CLI commands take `wiki_dir_option()` from `openinterpretability/cli.py` |
| Planner: base weights, ε, windows per model | `openinterpretability/planner.py` (`BASE_WEIGHTS`, `DEFAULT_EPSILON`, `configured_windows`) |
| Reproduction tolerances | `openinterpretability/reproduce.py` (`Tolerance`) |
| Models the planner and CI may use | `[tool.openinterpretability] models` in `pyproject.toml` |
| Each model's licence and publishing terms | `[tool.openinterpretability.licenses]` in `pyproject.toml` (`configured_licenses`) |
| Site-wide display maps: unit statuses, their order and colours | `openinterpretability/builder/statuses.py` (`MAP_STATUSES`, `STATUS_FILL`, `STATUS_BADGE`) |
| Evidence strength bands (strong / moderate / weak) | `openinterpretability/builder/significance.py` (`STRONG_EVIDENCE`, `evidence_strength`) |
| Standing roles and `CONTESTED` | `openinterpretability/builder/standing.py` |
| Canonical agent model names | `openinterpretability/core/agent_names.py` |
| Corpus sampling defaults | `openinterpretability/corpus.py` |

A record stores some policy defaults (`verdict.policy`). Read such a default from the record when one is at hand, and from `ReviewPolicy()` or the named constant otherwise.

## Templates

- Site-wide values go in `env.globals` in `openinterpretability/builder/compiler.py` (`_environment`). `statuses` is a global, so a template reads `statuses.STATUS_FILL`. `evidence_strength(c)` is a global, and `_macros.html` calls it.
- A macro can read `statuses.STATUS_BADGE[status]` directly.
- The SQL in `registry_stats.py` interpolates `{STRONG_EVIDENCE}`, as the conclusive-null count does.

## Copies that must exist

- `ALLOWED_MODELS` in `.github/workflows/ci.yml` copies the `pyproject.toml` models, so a fork PR can't widen what CI downloads. `tests/test_planner.py::test_configured_models_match_ci_allow_list` checks it.
- `docs/PROTOCOL.md`, the README and the skills quote numbers.
- The test of a copy goes in `tests/test_single_source.py`. Docs, skills and agent files quoting a number go in its `_quotes()` list. A comment at a copy reads, for example, `<!-- Defaults quote ReviewPolicy in core/schemas.py; tests/test_single_source.py checks them. -->`.
- Passing mentions such as "the rejection line, balanced accuracy 0.6" need no test when the docstring names the source.
- The protocol version a feature arrived in ("predates protocol 0.3") can't drift.

## A decision-rule value

A protocol default is a decision-rule change: bump `PROTOCOL_VERSION` and add a `docs/PROTOCOL.md` §6 entry.
