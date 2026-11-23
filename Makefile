.PHONY: install test coverage mutation perf lint typecheck markdown shell vale ste format check check-all stats maps build css search-index sample-run serve loop clean

WIKI_DIR ?= wiki
DIST_DIR ?= dist

# Reinstalls the package itself: when the distribution was renamed to open-interpretability
# (issue #3251), a plain sync removed the `openinterpretability` command with the old distribution.
install:
	uv sync --reinstall-package open-interpretability

# Single-threaded in each process: the tests use tiny models, and thread pools contending
# with other jobs on the machine have hung the suite.
# The tests run side by side on TEST_WORKERS processes (pytest-xdist, #3266): `auto` is one
# per core. `make test TEST_WORKERS=0` runs them one at a time, to chase a failure that
# only shows in parallel.
# The whole output goes to logs/test-last.log too, so a hang's stack dump (pytest-timeout)
# is kept even when only the exit code was read (#4376). pipefail keeps pytest's exit code.
TEST_WORKERS ?= auto
test:
	@mkdir -p logs
	bash -o pipefail -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m openinterpretability.core.heavy_lock pytest -n $(TEST_WORKERS) 2>&1 | tee logs/test-last.log'

# The suite with branch coverage (#3166): it fails below [tool.coverage.report] fail_under.
# Then diff-cover requires [tool.diff_cover] fail_under of the lines this branch changes
# against origin/main, as CI does on a pull request (#5855). About twice as slow as `make test`, so `make test` skips it.
COVERAGE_BASE ?= origin/main
coverage:
	@mkdir -p logs
	bash -o pipefail -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m openinterpretability.core.heavy_lock pytest -n $(TEST_WORKERS) --cov --cov-report=xml --cov-report=term 2>&1 | tee logs/coverage-last.log'
	uv run diff-cover coverage.xml -c pyproject.toml --compare-branch=$(COVERAGE_BASE)

# Mutation testing (scripts/mutation.py). With BASE, for example BASE=origin/main, it mutates
# the package files that the branch changes and runs the tests that import them (#5856).
# Without BASE, it mutates the verdict code of [tool.mutmut] in pyproject.toml (#5763), in
# about 5 minutes. It fails when a mutant survives or no test reaches it. Not in CI. mutmut is
# pinned here, so the lockfile doesn't carry it. `mutmut show <name>` gives a mutant's diff.
MUTMUT ?= mutmut==3.8.1
BASE ?=
mutation:
	rm -rf mutants
	uv run --with $(MUTMUT) python scripts/mutation.py $(if $(BASE),--base $(BASE))

# Wall time, CPU time and peak memory of fixed workloads, checked against
# docs/analyses/perf-baseline.json (scripts/perf_bench.py). Not in CI: runner timings vary.
# The heavy lock keeps another heavy job from overlapping the timings.
perf:
	uv run python -m openinterpretability.core.heavy_lock python scripts/perf_bench.py

# Unused code: a name vulture finds, and vulture_whitelist.py doesn't keep, fails (#3164).
vulture:
	uv run vulture openinterpretability scripts vulture_whitelist.py --min-confidence 60

# mypy, strict, on the packages in [tool.mypy] files (pyproject.toml).
typecheck:
	uv run mypy

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_line_limit.py
	uv run python scripts/check_step_comments.py
	uv run python scripts/check_body_comments.py
	uv run python scripts/check_docstring_args.py
	uv run python scripts/check_docstring_sections.py
	uv run python scripts/check_data_files.py
	uv run python scripts/exchange.py check
	python3 scripts/changelog.py check
	uv run python scripts/papers.py check
	python3 scripts/ifttt_lint.py
	python3 scripts/check_old_name.py
	uv run interrogate openinterpretability scripts
	$(MAKE) vulture
	$(MAKE) typecheck
	$(MAKE) markdown
	$(MAKE) shell
	$(MAKE) vale
	$(MAKE) ste

# Style and relative links of hand-written Markdown; see scripts/check_markdown.py.
markdown:
	uv run python scripts/check_markdown.py

# Shell scripts, with ShellCheck. The shellcheck-py dev dependency ships the binary, so
# `uv sync` installs it. git ls-files finds every tracked .sh file, so a new script is
# checked without a change here. CI sets SHELLCHECK=shellcheck: its venv is already on PATH,
# and `uv run` would sync the locked CUDA torch into it.
SHELLCHECK ?= uv run shellcheck
SHELL_PATHS ?= $(shell git ls-files -co --exclude-standard '*.sh')
shell:
	$(SHELLCHECK) $(SHELL_PATHS)

# Prose of comments and docstrings against our ASD-STE100 rules (.vale/styles/STE) and the
# APA rules (.vale/styles/APA). The fixtures of tests/test_vale_apa.py break the rules on
# purpose, so the run leaves them out. Vale is
# a Go binary, not a Python package: install it with `brew install vale` or from vale.sh.
# `make lint` runs it, so a long sentence fails locally as it does in CI.
# CI runs this target with the version pinned in .github/workflows/ci.yml. Vale exits
# non-zero only on errors, so the STE rules that gate are at level `error`.
VALE_PATHS ?= openinterpretability scripts tests
vale:
	@command -v vale >/dev/null || { echo "vale not found: brew install vale (see vale.sh)"; exit 1; }
	vale --glob='!tests/fixtures/vale/*' $(VALE_PATHS)

# Markdown prose against the structural ASD-STE100 rules: semicolons, sentences over 25
# words, synonym rotation. The linter comes with the vendored asd-ste100 skill and needs
# only the standard library. Passive voice and compound tenses are advisory and never fail.
# A ratchet: add a path here once it passes. The vendored skill's examples break the rules
# on purpose. Papers (papers/<slug>/README.md) get a looser profile, set in
# scripts/ste_prose.py: academic prose relates its claims in longer sentences, so the cap is
# 40 words with a mean of at most 25 (#4202). Every other path keeps the 25-word cap. CHANGELOG.md and its fragments (changelog.d/) are logs, so STE_LOG_PATHS skips synonym rotation
# there: Keep a Changelog requires a "Fixed" heading, and entries quote command names such
# as `verify` and `check`.
STE_PATHS ?= $(shell git ls-files -co --exclude-standard '.claude/skills/*.md' \
	':!.claude/skills/asd-ste100' 'docs/*.md' 'papers/*.md' README.md REVIEW.md CONTRIBUTING.md SECURITY.md \
	.github/pull_request_template.md)
STE_LOG_PATHS ?= CHANGELOG.md $(wildcard changelog.d/*.md)
ste:
	python3 scripts/ste_prose.py $(STE_PATHS)
	python3 .claude/skills/asd-ste100/scripts/ste-lint.py --disable synonym-rotation $(STE_LOG_PATHS)

# What a pull request's check runs: only the branch's changes, unless it changes code.
BASE_REF ?= origin/main
check:
	uv run open-interpretability check --base-ref $(BASE_REF)

# What CI runs on main: every record.
check-all:
	uv run open-interpretability check

# Experiments, units, models, VERIFIED and self-reviewed shares, and protocol versions.
stats:
	uv run open-interpretability stats --wiki-dir $(WIKI_DIR)

# The shared concept map of every configured model, refreshed by each release (#4785, #4586).
# Old poles keep their places. A refit (`open-interpretability map --refit`) is a separate PR.
maps:
	uv run --extra map open-interpretability map --wiki-dir $(WIKI_DIR)

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv run open-interpretability build --wiki-dir $(WIKI_DIR) --output-dir $(DIST_DIR)
	@$(MAKE) --no-print-directory css
	@$(MAKE) --no-print-directory search-index

# The stylesheet, from the classes the pages use (Tailwind's standalone CLI, no Node).
css:
	uv run python -m openinterpretability.builder.css --site $(DIST_DIR)

# Pagefind (a locked dev dependency, so no Node is needed) indexes dist/ for search.
search-index:
	uv run python -m pagefind --site $(DIST_DIR)

# A few real experiments from the wiki covering each verdict, a neuron with several
# experiments, and held-out explanation scores. Records are write-once, so paths are stable.
SAMPLE_RECORDS = $(addprefix wiki/experiments/Qwen--Qwen2.5-0.5B/, \
	L4/N1889-positive/20261019T113221Z.json \
	L9/N1458-negative/20261020T211015Z.json \
	L11/N4255-positive/20261019T050928Z.json \
	L11/N4255-positive/20261020T171549Z.json \
	L12/N4232-negative/20261021T044854Z.json \
	L20/N967-negative/20261021T040406Z.json)

# Builds the site from a small sample of real records (no model or LLM calls).
sample-run:
	rm -rf build/sample-records build/sample-wiki
	mkdir -p build/sample-records
	cp $(SAMPLE_RECORDS) build/sample-records/
	uv run open-interpretability wiki --import-dir build/sample-records --wiki-dir build/sample-wiki
	@$(MAKE) --no-print-directory build WIKI_DIR=build/sample-wiki
	@echo "Open $(DIST_DIR)/index.html, or run 'make serve' for DuckDB/Pagefind (needs HTTP)."

serve:
	uv run python -m http.server --directory $(DIST_DIR) 8000

# Run experiments back to back with headless Claude Code; see scripts/experiment-loop.sh --help.
loop:
	scripts/experiment-loop.sh $(LOOP_ARGS)

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
