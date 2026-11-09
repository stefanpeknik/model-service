"""The whole-slide builders, one package each, and what declaring one takes.

- base.py: Builder, BuilderContext and BuilderSettings, what a builder implements and gets.
- input_access.py: how a builder's inputs become local files (providers implement it).
- declarations.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders; each builder's spec.py declares it.
  Offering one in EMPAIA also takes an entry in providers/empaia/ead_builder_io.py.
- heatmap/, embeddings/: the builders; slides/: slide reading they share.
"""
