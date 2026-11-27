"""The prompt coverage of many kept sets, scored in batches (#7061).

:func:`openinterpretability.explain_coverage.keep_hooks` mean-ablates every unit but one
kept set, and costs one forward pass per set. A planner that tries every candidate unit
needs thousands of sets for each pick. :class:`KeptSets` runs one row per set in a batch.
Each row has its own boolean masks: which neuron poles and which heads keep their real
values. The result is the same as with ``keep_hooks``, set by set.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from openinterpretability.explain_groups import HEAD

BATCH = 256  # kept sets in one forward pass


class KeptSets:
    """The raw coverage of kept sets on one prompt's answer, with ``explained`` always kept.

    Raw coverage is ``(kept − ablated) / (clean − ablated)``, as
    :func:`openinterpretability.explain_coverage._raw` defines it.
    """

    def __init__(self, runner, means, prompt: str, token: int, position: int,
                 explained: Sequence, batch: int = BATCH):  # fmt: skip
        """Run the clean and the fully ablated passes once.

        Args:
            runner: The model's runner.
            means: The unit means that the ablation uses.
            prompt: The prompt.
            token: The answer token.
            position: The position of the answer's logits.
            explained: The units that every set keeps.
            batch: The kept sets in one forward pass.
        """
        self.runner, self.means, self.token, self.position = runner, means, token, position
        self.explained, self.batch = list(explained), batch
        self.ids = runner.encode(prompt)["input_ids"]
        self.width = int(means.neurons[0].shape[-1])
        with torch.no_grad():
            logits = runner.forward({"input_ids": self.ids})[0, position].float()
        self.clean = float(torch.log_softmax(logits, -1)[token])
        self.ablated = self._log_probs(self._empty())[0]

    def _empty(self) -> tuple[torch.Tensor, ...]:
        """Masks of one row that keeps nothing, not even the explained units.

        Returns:
            The masks, as :meth:`masks` gives them.
        """
        explained, self.explained = self.explained, []
        try:
            return self.masks([[]])
        finally:
            self.explained = explained

    def raw(self, sets: Sequence[Sequence]) -> list[float | None]:
        """The raw coverage of each set, with the explained units kept too.

        Args:
            sets: The kept sets of units.

        Returns:
            Each set's raw coverage. Each value is None when ablating every unit leaves the
            answer as it was.
        """
        if abs(self.clean - self.ablated) <= 1e-9:
            return [None] * len(sets)  # skip building masks that can't give a value
        return self.raw_masks(self.masks(sets)) if sets else []

    def raw_masks(self, masks: tuple[torch.Tensor, ...]) -> list[float | None]:
        """The raw coverage of each row of prebuilt ``masks`` (:meth:`masks`).

        Args:
            masks: The kept poles and heads of each row.

        Returns:
            Each row's raw coverage, or None for each row when there is no span to cover.
        """
        span = self.clean - self.ablated
        rows = masks[0].shape[1]
        if abs(span) <= 1e-9:
            return [None] * rows
        found = self._log_probs(masks)
        return [(kept - self.ablated) / span for kept in found]

    def masks(self, sets: Sequence[Sequence]) -> tuple[torch.Tensor, ...]:
        """Per layer and row: the kept positive poles, negative poles and heads.

        Each row keeps one set and the explained units. A caller may edit the masks, for
        example to drop one unit from each row, and pass them to :meth:`raw_masks`.

        Args:
            sets: The kept sets of units, one for each row.

        Returns:
            The positive, negative and head masks, on the runner's device.
        """
        layers, rows = self.runner.n_layers, len(sets)
        n_heads = self.runner.n_heads if self.means.heads else 1  # a toy model has no heads
        positive = torch.zeros(layers, rows, self.width, dtype=torch.bool)
        negative = torch.zeros(layers, rows, self.width, dtype=torch.bool)
        heads = torch.zeros(layers, rows, n_heads, dtype=torch.bool)
        for row, units in enumerate(sets):
            for layer, index, pole in [*self.explained, *units]:
                mask = heads if pole == HEAD else positive if pole == "positive" else negative
                mask[layer, row, index] = True
        device = self.runner.device
        return positive.to(device), negative.to(device), heads.to(device)

    @torch.no_grad()
    def _log_probs(self, masks: tuple[torch.Tensor, ...]) -> list[float]:
        """The answer's log-probability for each row of ``masks``, ``batch`` rows per pass.

        Args:
            masks: The kept poles and heads of each row.

        Returns:
            Each row's log-probability.
        """
        out: list[float] = []
        for start in range(0, masks[0].shape[1], self.batch):
            positive, negative, heads = (m[:, start : start + self.batch] for m in masks)
            rows = positive.shape[1]
            layers = range(self.runner.n_layers)
            neurons = {la: self._neurons(la, positive, negative) for la in layers}
            heads_at = {la: self._heads(la, heads) for la in layers}
            with self.runner.hooks(neurons, None, heads_at if self.means.heads else None):
                logits = self.runner.forward({"input_ids": self.ids.expand(rows, -1)})
            found = torch.log_softmax(logits[:, self.position].float(), -1)[:, self.token]
            out += [float(v) for v in found]
        return out

    def _neurons(self, layer: int, positive: torch.Tensor, negative: torch.Tensor):
        """The hook that mean-ablates ``layer``'s neurons, past token 0, but the kept poles.

        Args:
            layer: The layer.
            positive: The kept positive poles of each row.
            negative: The kept negative poles of each row.

        Returns:
            The hook.
        """

        def intervene(act: torch.Tensor) -> torch.Tensor:
            """Mean-ablate the activations that no row keeps.

            Args:
                act: The layer's neuron activations, one batch row for each kept set.

            Returns:
                The new activations.
            """
            mean = self.means.neurons[layer].to(act).expand_as(act)
            up, down = positive[layer][:, None, :], negative[layer][:, None, :]
            keep = ((act > 0) & up) | ((act < 0) & down)
            out = torch.where(keep, act, mean).clone()
            out[:, 0, :] = act[:, 0, :]  # the first token stays, as in the attribution
            return out

        return intervene

    def _heads(self, layer: int, heads: torch.Tensor):
        """The hook that mean-ablates ``layer``'s heads, past token 0, but the kept heads.

        Args:
            layer: The layer.
            heads: The kept heads of each row.

        Returns:
            The hook.
        """

        def intervene(z: torch.Tensor) -> torch.Tensor:
            """Mean-ablate the head outputs that no row keeps.

            Args:
                z: The layer's head outputs, one batch row for each kept set.

            Returns:
                The new head outputs.
            """
            width = self.runner.head_dim(layer)
            kept = heads[layer, :, : self.runner.n_heads].repeat_interleave(width, dim=1)
            out = z.clone()
            mean = self.means.heads[layer].to(z).expand_as(z[:, 1:, :])
            out[:, 1:, :] = torch.where(kept[:, None, :], z[:, 1:, :], mean)
            return out

        return intervene
