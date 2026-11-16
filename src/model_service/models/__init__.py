"""Models: the generic code here, the Ray Serve model code below.

Generic files:

- catalog.py: the deployed models, read from the model YAML files.
- clients.py: the clients that call a model and check its answers.
- artifacts.py: the MLflow loader the model YAML files name as `_target_`.

Subpackages:

- implementations/: the Ray Serve code that serves trained models; it runs only
  inside Ray replicas.

What a model of each contract declares is contracts/model_declarations.py.
"""
