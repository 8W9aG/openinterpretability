"""Open Interpretability: an agent-run interpretability project and its neuron registry."""

from labinacan import config as _config

from openinterpretability.core.legacy_names import adopt_legacy_env
from openinterpretability.core.paths import CLONE_PYPROJECT, DEFAULT_PYPROJECT, PACKAGED_PYPROJECT

__version__ = "0.6.0"
# The name on PyPI, and of the installed distribution (issue #3251). The import name is
# `openinterpretability` (issue #4089).
DISTRIBUTION = "open-interpretability"

adopt_legacy_env()  # the old environment variables, for one release (core/legacy_names.py)

# labinacan reads this project's values from a pyproject.toml (#6629). It reads the working
# directory's when it has them. Else it reads the copy that an install holds, else the clone's.
_config.use(DEFAULT_PYPROJECT, PACKAGED_PYPROJECT, CLONE_PYPROJECT)
