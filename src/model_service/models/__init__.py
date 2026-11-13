"""Models as the service sees them, and the code that serves them.

- catalog.py: the deployed models, read from the model YAML files.
- clients.py: the clients that call a model and check its answers.
- artifacts.py: the MLflow loader the model YAML files name as `_target_`.
- implementations/: the Ray Serve model code; it runs only inside Ray replicas.
"""
