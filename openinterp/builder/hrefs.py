"""The site's page paths for models, layers and units, relative to the site's root."""

from __future__ import annotations


def model_href(slug: str) -> str:
    """The model's index page."""
    return f"models/{slug}/index.html"


def circuits_href(slug: str) -> str:
    """The model's circuits page."""
    return f"models/{slug}/circuits.html"


def layer_href(slug: str, layer: int) -> str:
    """The layer's index page."""
    return f"models/{slug}/L{layer}/index.html"


def neuron_href(slug: str, layer: int, idx: int) -> str:
    """The neuron's page (both poles)."""
    return f"models/{slug}/L{layer}/N{idx}.html"


def head_href(slug: str, layer: int, idx: int) -> str:
    """The head's page."""
    return f"models/{slug}/L{layer}/H{idx}.html"


def feature_href(slug: str, layer: int, dictionary: str, idx: int) -> str:
    """The dictionary feature's page."""
    return f"models/{slug}/L{layer}/{dictionary}/F{idx}.html"
