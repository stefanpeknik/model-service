"""RationAI model service: runs pathology models on Ray Serve behind one API.

- api/: the API process (HTTP routes, wiring). Raw tiles go straight to Ray
  Serve (`POST /<model-id>/`), not through the API.
- scheduler/: durable executions on PostgreSQL, run inside the API process;
  it calls builders through builder_dispatch.py (interface) and
  http_builder_dispatch.py (over HTTP, see docs/adr/0001).
- builders/: the whole-slide builders and how to declare one.
- builder_deployment/: the Ray Serve deployment that hosts a builder.
- inference/: the deployed models (catalog.py, read from the model YAML) and
  the clients that call and check them (model_clients.py).
- providers/: where inputs come from and results go (standalone, EMPAIA).
- http_common/: auth, error mapping and scheduling calls the API and provider routes share.
- shared/: types shared across processes (invocation, result, failures, model
  declarations, the builder HTTP format, score grid).
- model_implementations/: the Ray Serve code that serves trained models; each
  trained model is one file in deploy/helm/rayservice/models/ naming its implementation.
"""
