"""Whole-slide builders: the generic code here, the concrete builders below.

Generic files, which every builder uses:

- base.py: Builder, BuilderContext, BuilderSettings and ReplicaSettings: what a
  builder implements and gets.
- input_access.py: how a builder's inputs become local files (providers implement it).
- declarations.py: BuilderSpec and the input and output models a builder declares.
- registry.py: the registered builders; each builder's spec.py declares it.
  Offering one in EMPAIA also takes an entry in providers/empaia/ead/builder_io.py
  (BUILDER_IO, or NOT_OFFERED with a reason; a test checks).
- slides/: slide reading the builders share.
- model_clients/: the clients that call a model and check its answers, one
  module per contract; registry.py maps each contract to its client.

Subpackages:

- implementations/: the concrete builders, one package each (heatmap/, embeddings/).
- replica/: the Ray Serve deployment that hosts a builder; it runs only inside
  Ray replicas, never in the API process.
"""
