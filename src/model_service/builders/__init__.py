"""The whole-slide builders, one package each, and what declaring one takes.

- base.py: Builder, BuilderContext, BuilderSettings and ReplicaSettings: what a
  builder implements and gets.
- input_access.py: how a builder's inputs become local files (providers implement it).
- declarations.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders; each builder's spec.py declares it.
- heatmap/: the builder; slides/: slide reading builders share.
"""
