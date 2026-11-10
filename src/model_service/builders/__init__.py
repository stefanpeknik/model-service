"""The whole-slide builders, one package each, and what declaring one takes.

- base.py: Builder, BuilderContext, BuilderSettings and ReplicaSettings: what a
  builder implements and gets.
- input_access.py: how a builder's inputs become local files (providers implement it).
- declarations.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders; each builder's spec.py declares it.
  Offering one in EMPAIA also takes an entry in providers/empaia/ead/builder_io.py
  (BUILDER_IO, or NOT_OFFERED with a reason; a test checks).
- heatmap/, embeddings/: the builders; slides/: slide reading they share.
"""
