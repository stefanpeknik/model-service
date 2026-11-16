"""The concrete builders, one package each.

- heatmap/: scores tiles with a binary classifier into a score grid.
- embeddings/: writes tile embeddings from a foundation model to HDF5.

Each package has spec.py (registered in builders/registry.py), builder.py and
processor.py.
"""
