"""Running a scan's forward pass on Apple's Neural Engine (ANE), through Core ML.

- ``forward.py``: the forward pass, written so the ANE computes it well.
- ``convert.py``: conversion to a compiled Core ML model (coremltools, on Apple-silicon Macs).
- ``cache.py``: compiled models on disk, built once per machine.
- ``backend.py``: :class:`~openinterp.harness.ane.backend.AneScan`, the scan's pass.

docs/analyses/ane-feasibility.md has the measurements behind it.
"""
