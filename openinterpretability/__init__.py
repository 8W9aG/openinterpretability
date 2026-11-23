"""Open Interpretability: an agent-run interpretability project and its neuron registry."""

from openinterpretability.core.legacy_names import adopt_legacy_env

__version__ = "0.4.1"
# The name on PyPI, and of the installed distribution (issue #3251). The import name is
# `openinterpretability` (issue #4089).
DISTRIBUTION = "open-interpretability"

adopt_legacy_env()  # the old environment variables, for one release (core/legacy_names.py)
