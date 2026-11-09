"""The whole-slide builders, one package each, and what declaring one takes.

- base.py: Builder and BuilderContext, what a builder implements and gets.
- declarations.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders; each builder's spec.py declares it.
- heatmap/, embeddings/: the builders; slides/: slide reading they share.
"""
