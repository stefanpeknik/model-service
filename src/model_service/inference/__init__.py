"""Inference: the Ray Serve code that serves trained models, and its MLflow loader.

It runs only inside Ray model replicas and imports nothing else from model_service.

Generic files:

- artifacts.py: the MLflow loader the model YAML files name as `_target_`.

Subpackages:

- implementations/: the Ray Serve code that serves trained models; it runs only
  inside Ray replicas.

What a model of each contract declares is contracts/model_declarations.py.
"""
