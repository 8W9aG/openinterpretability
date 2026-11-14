.PHONY: install test perf lint markdown shell vale ste format check check-all stats build search-index sample-run serve loop clean

WIKI_DIR ?= wiki
DIST_DIR ?= dist

# Reinstalls the package itself: when the distribution was renamed to open-interpretability
# (issue #3251), a plain sync removed the `openinterp` command with the old distribution.
install:
	uv sync --reinstall-package open-interpretability

# Single-threaded: the tests use tiny models, and thread pools contending with other
# jobs on the machine have hung the suite.
test:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m openinterp.core.heavy_lock pytest

# Wall time, CPU time and peak memory of fixed workloads, checked against
# docs/analyses/perf-baseline.json (scripts/perf_bench.py). Not in CI: runner timings vary.
# The heavy lock keeps another heavy job from overlapping the timings.
perf:
	uv run python -m openinterp.core.heavy_lock python scripts/perf_bench.py

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_line_limit.py
	uv run python scripts/check_data_files.py
	uv run python scripts/exchange.py check
	python3 scripts/changelog.py check
	uv run interrogate openinterp
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

# Prose of comments and docstrings against our ASD-STE100 rules (.vale/styles/STE). Vale is
# a Go binary, not a Python package: install it with `brew install vale` or from vale.sh.
# `make lint` runs it, so a long sentence fails locally as it does in CI.
# CI runs this target with the version pinned in .github/workflows/ci.yml. Vale exits
# non-zero only on errors, so the STE rules that gate are at level `error`.
VALE_PATHS ?= openinterp scripts tests
vale:
	@command -v vale >/dev/null || { echo "vale not found: brew install vale (see vale.sh)"; exit 1; }
	vale $(VALE_PATHS)

# Markdown prose against the structural ASD-STE100 rules: semicolons, sentences over 25
# words, synonym rotation. The linter comes with the vendored asd-ste100 skill and needs
# only the standard library. Passive voice and compound tenses are advisory and never fail.
# A ratchet: add a path here once it passes. The vendored skill's examples break the rules
# on purpose. CHANGELOG.md and its fragments (changelog.d/) are logs, so STE_LOG_PATHS skips synonym rotation
# there: Keep a Changelog requires a "Fixed" heading, and entries quote command names such
# as `verify` and `check`.
STE_PATHS ?= $(shell git ls-files -co --exclude-standard '.claude/skills/*.md' \
	':!.claude/skills/asd-ste100' 'docs/*.md' 'papers/*.md' README.md REVIEW.md CONTRIBUTING.md SECURITY.md \
	.github/pull_request_template.md)
STE_LOG_PATHS ?= CHANGELOG.md $(wildcard changelog.d/*.md)
ste:
	python3 .claude/skills/asd-ste100/scripts/ste-lint.py $(STE_PATHS)
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

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv run open-interpretability build --wiki-dir $(WIKI_DIR) --output-dir $(DIST_DIR)
	@$(MAKE) --no-print-directory search-index

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
