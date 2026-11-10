"""RationAI model service: runs pathology models on Ray Serve behind one API.

- api/: the API process (HTTP routes, wiring, raw inference proxy).
- scheduler/: durable executions on PostgreSQL, run inside the API process.
- builders/: the whole-slide builders and how to declare one.
- builder_deployment/: the Ray Serve deployment that hosts a builder, and its HTTP client.
- inference/: the deployed models (catalog.py, read from the model YAML), what
  each contract promises (model_declarations.py) and the clients that call and
  check them (model_clients.py).
- providers/: where inputs come from and results go (standalone, EMPAIA).
- shared/: types shared across processes (invocation, result, failures, score grid).
- model_implementations/: the Ray Serve code that serves trained models; each
  trained model is one file in deploy/helm/rayservice/models/ naming its implementation.
"""
