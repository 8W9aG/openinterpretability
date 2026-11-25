"""The errors that labinacan raises, so a project's CLI can print them as text (#6628)."""


class LabinacanError(Exception):
    """The base of labinacan's errors."""


class SyncError(LabinacanError):
    """Deploying the static site to object storage failed."""
