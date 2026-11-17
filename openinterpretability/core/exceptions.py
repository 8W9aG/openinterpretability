"""Exception hierarchy for openinterpretability."""


class OpenInterpretabilityError(Exception):
    """Base class for all openinterpretability errors."""


class ArchitectureNotSupportedError(OpenInterpretabilityError):
    """The hook runner could not locate MLP layers on the given model."""


class PatchingError(OpenInterpretabilityError):
    """An activation patching run could not be performed as requested."""


class ShapeMismatchError(PatchingError):
    """Source and target activations have incompatible shapes."""


class AgentError(OpenInterpretabilityError):
    """An LLM agent failed to produce a valid structured response."""


class RegistryError(OpenInterpretabilityError):
    """An audit record on disk is missing or malformed."""


class SyncError(OpenInterpretabilityError):
    """Deploying the static site to object storage failed."""
