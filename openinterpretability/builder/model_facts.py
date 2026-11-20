"""What the Models page says about each supported model, read from its ``config.json``.

The config is the one source. The parameter count comes from building the model from
its config on the meta device, so no weights load. The Hub's safetensors count is not
used: for GPT-2 it includes the attention-mask buffers (137M, not 124M).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from openinterpretability.planner_models import config_dims, model_config


@dataclass(frozen=True)
class ModelFacts:
    """The architecture of one model. A field is None when the config does not give it."""

    parameters: int | None
    layers: int | None
    mlp_width: int | None
    activation: str | None


def parameter_count(config: dict) -> int | None:
    """Parameters of the model that ``config`` describes, tied weights counted once.

    Buffers (such as GPT-2's attention mask) are not parameters, so they are not counted.
    Returns None when transformers cannot build the model from the config.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        cfg = AutoConfig.for_model(**config)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception:  # an architecture this transformers version does not know
        return None
    return sum(p.numel() for p in model.parameters())


def facts_from_config(config: dict) -> ModelFacts:
    """The facts that ``config`` gives."""
    dims = config_dims(config)
    activation = config.get("hidden_act") or config.get("activation_function")
    return ModelFacts(
        parameters=parameter_count(config),
        layers=dims[0] if dims else None,
        mlp_width=dims[1] if dims else None,
        activation=activation if isinstance(activation, str) else None,
    )


def load_model_facts(models: Iterable[str], download: bool = True) -> dict[str, ModelFacts]:
    """The facts of each model whose ``config.json`` can be read; others are left out.

    ``download`` is as for :func:`openinterpretability.planner_models.model_config`.
    """
    facts = {}
    for model in models:
        config = model_config(model, download)
        if config is not None:
            facts[model] = facts_from_config(config)
    return facts


def format_parameters(n: int | None) -> str:
    """``124M`` or ``1.54B``: three significant figures; ``—`` when unknown."""
    if n is None:
        return "—"
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


def placeholder_models(configured: Iterable[str], studied: Iterable, facts: dict) -> list:
    """An empty model node for each configured model with no record yet.

    So such a model gets a page. A model whose config gives no layer count is left out:
    its page would have no layers to show.
    """
    from openinterpretability.builder.tree import ModelNode, model_href
    from openinterpretability.core.identifiers import model_slug

    have = {m.name for m in studied}
    nodes = []
    for name in configured:
        f = facts.get(name)
        if name in have or f is None or not f.layers:
            continue
        slug = model_slug(name)
        nodes.append(ModelNode(name, slug, f.layers, model_href(slug), f.mlp_width))
    return nodes


def models_view(configured: Iterable[str], nodes: Iterable, facts: dict, licenses: dict):
    """One row per configured model for the Models page, in the configured order.

    ``nodes`` are the model nodes that have a page. Layers and MLP width fall back to
    what the records give when the config cannot be read.
    """
    by_name = {m.name: m for m in nodes}
    rows = []
    for name in configured:
        node, f = by_name.get(name), facts.get(name)
        rows.append(
            {
                "name": name,
                "href": node.href if node else None,
                "size": format_parameters(f.parameters if f else None),
                "parameters": f.parameters if f else None,
                "layers": (f and f.layers) or (node.n_layers if node else None),
                "mlp_width": (f and f.mlp_width) or (node.mlp_width if node else None),
                "activation": f.activation if f else None,
                "terms": licenses.get(name),
                "experiments": node.experiments if node else 0,
                "verified": (node.verified + node.verified_heads) if node else 0,
            }
        )
    return rows


def render_models(render, models, facts: dict, render_model, draws=()) -> None:
    """Write the Models page, and a page for each configured model with no record yet.

    ``render`` is the compiler's renderer and ``render_model`` writes one model's pages.
    ``draws`` are the random draws and stops. From them ``explore_focus`` gives each model's
    verified count for its card, which shows its pieces bars as the home page does (#3287).
    """
    from openinterpretability.planner import configured_models

    configured = configured_models()
    empty = placeholder_models(configured, models, facts)
    for model in empty:
        render_model(render, model, {}, set())
    rows = models_view(configured, [*models, *empty], facts, render.common["model_licenses"])
    from openinterpretability.builder.stopped import explore_focus

    explore = explore_focus(models, draws)  # proven dead counts as verified, as on the home page
    atlas = render.common["concept_atlas"](models) if "concept_atlas" in render.common else None
    render("models/index.html", "models.html", rows=rows, models=models, explore=explore,
           atlas=atlas is not None)  # fmt: skip
    if atlas is not None:  # every model's poles in the shared concept map (#4586)
        render("models/atlas.html", "atlas.html", atlas=atlas)


def model_page_extras(coverage_rows: dict, explanations, circuits: dict | None = None,
                      concept_map: dict | None = None) -> dict:  # fmt: skip
    """What the model pages show besides the tree, for the site context.

    ``coverage`` is each model's registry coverage (#3837). ``explanations_by_model`` is
    each model's explanations (#4206). Both are keyed by model name. ``circuit_data`` is
    each model's registry circuits and hub norms (#4287), keyed by model slug.
    ``concept_map`` gives a model node's view of the shared concept map, and
    ``concept_atlas`` the atlas's view of it (#4422, #4586).
    """
    from openinterpretability.builder.concept_map_view import atlas_view, map_view
    from openinterpretability.builder.coverage import coverage_view
    from openinterpretability.builder.explanation_pages import by_model

    shown = {"coverage": coverage_view(coverage_rows),
             "concept_map": lambda model: map_view(model, concept_map),
             "concept_atlas": lambda models: atlas_view(models, concept_map)}  # fmt: skip
    return shown | {"explanations_by_model": by_model(explanations), "circuit_data": circuits or {}}
