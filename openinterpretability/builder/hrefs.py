"""The site's page paths for models, layers and units, relative to the site's root."""

from __future__ import annotations


def model_href(slug: str) -> str:
    """The model's index page."""
    return f"models/{slug}/index.html"


def circuits_href(slug: str) -> str:
    """The model's circuits page."""
    return f"models/{slug}/circuits.html"


def circuit_href(slug: str, key: str) -> str:
    """One circuit's page, named after its root unit's key (``builder.circuits.unit_key``)."""
    return f"models/{slug}/circuits/{key}.html"


def hub_href(slug: str, key: str) -> str:
    """One hub's page, named after the hub unit's key."""
    return f"models/{slug}/hubs/{key}.html"


def prompt_href(slug: str, prompt_id: str) -> str:
    """A prompt's page, which lists its explanations (#4503)."""
    return f"models/{slug}/explanations/{prompt_id}.html"


def explanation_href(slug: str, prompt_id: str, record_id: str) -> str:
    """One explanation's page, named after its record's UTC stamp (#4503)."""
    return f"models/{slug}/explanations/{prompt_id}/{record_id}.html"


def answer_href(slug: str, prompt_id: str, record_id: str) -> str:
    """One answer experiment's page, named after its record's UTC stamp (#4909)."""
    return f"models/{slug}/explanations/{prompt_id}/answers/{record_id}.html"


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


def superposition_href(slug: str, layer: int, superposition_id: str) -> str:
    """The superposition's page."""
    return f"models/{slug}/L{layer}/S{superposition_id}.html"


def unit_href(ident) -> str:
    """The page of the neuron, head, feature or superposition ``ident`` names."""
    if ident.kind == "dictionary_feature":
        return feature_href(ident.model_slug, ident.layer, ident.dictionary, ident.index)
    if ident.kind == "superposition":
        return superposition_href(ident.model_slug, ident.layer, ident.superposition_id)
    href = head_href if ident.kind == "attn_head" else neuron_href
    return href(ident.model_slug, ident.layer, ident.index)


def experiment_href(slug: str, experiment_id: str) -> str:
    """The experiment's page, at the path of its record under ``wiki/experiments/``.

    ``slug`` is the unit's identifier slug, for a neuron its pole's (``…/L3/N7-positive``).
    """
    return f"models/{slug}/{experiment_id}.html"


def user_href(handle: str) -> str:
    """The page of the person with the public handle ``handle``."""
    return f"users/{handle}.html"
