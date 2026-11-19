"""Whole-slide builders: what a builder declares and implements.

- base.py: Builder, BuilderContext, BuilderSettings and ReplicaSettings: what a
  builder implements and gets.
- input_access.py: how a builder's inputs become local files (providers implement it).
- declarations.py: BuilderSpec and the input and output models a builder declares.
- directories.py: directories the API and Ray replicas share, open to all users.
- model_clients/: the clients that call a model and check its answers, one
  module per contract; registry.py maps each contract to its client.
"""
