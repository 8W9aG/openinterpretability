# The generic Make targets of an autoresearch project (#6630). The project's Makefile sets
# its values, then includes this file:
#
#   DIST_NAME     the project's distribution, which `install` reinstalls.
#   PACKAGES      its Python packages, for vulture, interrogate and Vale.
#   TEST_WRAPPER  a command that each test run goes through, or nothing.
#   BUILD_COMMAND the command that builds the site into $(DIST_DIR).
#   CSS_MODULE    the module that writes the site's stylesheet.
#   PROJECT_LINT  a target of its own checks, which `lint` runs, or nothing.
#
# This file holds no value of a project, so a project can take a new copy of it with no
# merge conflict (tests/test_labinacan_config.py checks that).

ifndef DIST_NAME
$(error Set DIST_NAME before including labinacan.mk)
endif
ifndef PACKAGES
$(error Set PACKAGES before including labinacan.mk)
endif

.PHONY: install test coverage mutation vulture typecheck lint markdown shell vale ste format build css search-index serve clean

WIKI_DIR ?= wiki
DIST_DIR ?= dist

# Reinstalls the project's own distribution. After a rename of the distribution (#3251), a
# plain sync removed its command with the old one.
install:
	uv sync --reinstall-package $(DIST_NAME)

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
	bash -o pipefail -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run $(TEST_WRAPPER) pytest -n $(TEST_WORKERS) 2>&1 | tee logs/test-last.log'

# The suite with branch coverage (#3166): it fails below [tool.coverage.report] fail_under.
# Then diff-cover requires [tool.diff_cover] fail_under of the lines this branch changes
# against origin/main, as CI does on a pull request (#5855). About twice as slow as `make test`, so `make test` skips it.
COVERAGE_BASE ?= origin/main
coverage:
	@mkdir -p logs
	bash -o pipefail -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run $(TEST_WRAPPER) pytest -n $(TEST_WORKERS) --cov --cov-report=xml --cov-report=term 2>&1 | tee logs/coverage-last.log'
	uv run diff-cover coverage.xml -c pyproject.toml --compare-branch=$(COVERAGE_BASE)

# Mutation testing (scripts/mutation.py). With BASE, for example BASE=origin/main, it mutates
# the package files that the branch changes and runs the tests that import them (#5856).
# It runs only the mutants in the changed functions, at most 100 of them (#5857).
# Without BASE, it mutates the verdict code of [tool.mutmut] in pyproject.toml (#5763), in
# about 2 minutes. It fails when a mutant survives or no test reaches it. CI runs it with BASE
# on PRs from authors other than the maintainer (#5857). mutmut is
# pinned in the dev group (pyproject.toml). `mutmut show <name>` gives a mutant's diff.
# RUNS=5 runs it 5 times, each in its own test order (pytest-randomly seeds 1 to 5), and fails
# when a mutant's result changes between runs (#6359), in about 25 minutes. The plugin is
# pinned here only.
BASE ?=
RUNS ?=
ORDER_PLUGIN := pytest-randomly==3.16.0
mutation:
	rm -rf mutants
	uv run $(if $(RUNS),--with $(ORDER_PLUGIN)) python scripts/mutation.py \
		$(if $(BASE),--base $(BASE)) $(if $(RUNS),--runs $(RUNS))

# Unused code: a name vulture finds, and vulture_whitelist.py doesn't keep, fails (#3164).
vulture:
	uv run vulture $(PACKAGES) scripts vulture_whitelist.py --min-confidence 60

# mypy, strict, on the packages in [tool.mypy] files (pyproject.toml).
typecheck:
	uv run mypy

# The project's own checks run after the generic checks of the code and before the slower
# whole-repository ones, as a target that PROJECT_LINT names.
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_line_limit.py
	uv run python scripts/check_step_comments.py
	uv run python scripts/check_body_comments.py
	uv run python scripts/check_docstring_args.py
	uv run python scripts/check_docstring_sections.py
	$(if $(PROJECT_LINT),$(MAKE) --no-print-directory $(PROJECT_LINT))
	uv run python scripts/exchange.py check
	python3 scripts/changelog.py check
	uv run python scripts/papers.py check
	python3 scripts/ifttt_lint.py
	uv run interrogate $(PACKAGES) scripts
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
VALE_PATHS ?= $(PACKAGES) scripts tests
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

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	$(BUILD_COMMAND)
	@$(MAKE) --no-print-directory css
	@$(MAKE) --no-print-directory search-index

# The stylesheet, from the classes the pages use (Tailwind's standalone CLI, no Node).
css:
	uv run python -m $(CSS_MODULE) --site $(DIST_DIR)

# Pagefind (a locked dev dependency, so no Node is needed) indexes dist/ for search.
search-index:
	uv run python -m pagefind --site $(DIST_DIR)

serve:
	uv run python -m http.server --directory $(DIST_DIR) 8000

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
