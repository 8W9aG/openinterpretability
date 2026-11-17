"""The measurement harness: hooks, scans, patching and readouts on the model itself."""

from openinterpretability.harness.attribution import attribution_patching, top_attributed_neurons
from openinterpretability.harness.hooks import (
    NeuronHookRunner,
    activation_histogram,
    isolate_pole,
    pole_strength,
    replace_pole,
)
from openinterpretability.harness.lookahead import measure_lookahead, output_token_projection
from openinterpretability.harness.patcher import (
    AblationResult,
    ActivationPatcher,
    PatchResult,
    logit_diff_metric,
    token_logit_metric,
)
from openinterpretability.harness.scan import top_activating_examples

__all__ = [
    "AblationResult",
    "ActivationPatcher",
    "NeuronHookRunner",
    "PatchResult",
    "activation_histogram",
    "attribution_patching",
    "isolate_pole",
    "logit_diff_metric",
    "measure_lookahead",
    "output_token_projection",
    "pole_strength",
    "replace_pole",
    "token_logit_metric",
    "top_activating_examples",
    "top_attributed_neurons",
]
