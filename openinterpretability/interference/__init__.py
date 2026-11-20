"""The interference report: what a change to a model does to what the registry explains (#5230).

A *changed* model is a base model with a patch applied (#5229), or another model with the
same architecture, such as an instruct model. The report measures four levels against the
base: each VERIFIED unit's probes and output, each recorded route, and behaviour. Each level
also runs against two noise floors, the base against itself and a matched random change, and
a change counts only above both.

- :mod:`.keep`: the keep test, each VERIFIED pole's probes on both models.
- :mod:`.units`: every neuron's activation and output, for the footprint and the coverage.
- :mod:`.behaviour`: token-level KL and the answers of explained prompts.
- :mod:`.floors`: the matched random changes.
- :mod:`.report`: the rows, the footprint and the verdict.
"""
