"""The whole-slide builders, one package each, and what declaring one takes.

- base.py: Builder and BuilderContext, what a builder implements and gets.
- spec.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders.
- heatmap/, embeddings/: the builders; slides/: slide reading they share.
"""
