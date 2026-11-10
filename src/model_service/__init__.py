"""RationAI model service: runs pathology models on Ray Serve behind one API.

- facade/: the API process (HTTP routes, wiring, raw inference proxy).
- scheduler/: durable executions on PostgreSQL, run inside the facade process.
- builders/: the whole-slide builders and how to declare one.
- builder_deployment/: the Ray Serve deployment that hosts a builder, and its HTTP client.
- inference/: what a model of each contract promises (contracts.py) and the
  clients that call it (model_clients.py).
- providers/: where inputs come from and results go (standalone, EMPAIA).
- wire/: types that cross process boundaries.
- catalog.py: the deployed models, read from the model YAML files.
- models/: the model implementations Ray Serve runs.
"""
