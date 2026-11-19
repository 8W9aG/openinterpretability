"""Fix patches: commit-like records of declarative changes to a base model (issue #5229).

A fix is not a new model. It is a patch record that says how to change a base model, as a
git commit says how to change a tree. Patches form a directed acyclic graph through their
``parents``, and branches name patches. The patched model is built only to run the fix's own
checks (#5220 §9.2).

- :mod:`.schema`: the record, its operations and the versioned operation registry.
- :mod:`.tensors`: the content-addressed store of the tensors that operations hold.
- :mod:`.graph`: the canonical hook points and parameters of each architecture.
- :mod:`.gates` and :mod:`.apply`: apply a patch chain to a model in memory, and remove it.
- :mod:`.history`: replay, branches, merges, reverts and rebases.
- :mod:`.store`: patch records and refs in the wiki.
"""
