"""Exception hierarchy for openinterp."""


class OpenInterpError(Exception):
    """Base class for all openinterp errors."""


class ArchitectureNotSupportedError(OpenInterpError):
    """The hook runner could not locate MLP layers on the given model."""


class PatchingError(OpenInterpError):
    """An activation patching run could not be performed as requested."""


class ShapeMismatchError(PatchingError):
    """Source and target activations have incompatible shapes."""


class AgentError(OpenInterpError):
    """An LLM agent failed to produce a valid structured response."""


class RegistryError(OpenInterpError):
    """An audit record on disk is missing or malformed."""


class SyncError(OpenInterpError):
    """Deploying the static site to object storage failed."""
