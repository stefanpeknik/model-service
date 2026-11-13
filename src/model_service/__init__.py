"""RationAI model service: runs pathology models on Ray Serve behind one API.

- api/: the API process (HTTP routes, wiring). Raw tiles go straight to Ray
  Serve (`POST /<model-id>/`), not through the API.
- scheduler/: durable executions on PostgreSQL; it calls builder replicas over
  HTTP in builder_dispatch.py, which the API's synchronous runs use too.
- builders/: the whole-slide builders and how to declare one.
- builder_deployment/: the Ray Serve deployment that hosts a builder.
- deployed_models/: the deployed models (catalog.py, read from the model YAML) and
  the clients that call and check them (model_clients/).
- providers/: where inputs come from and results go: the provider contract and shared storage.
- http_common/: auth and error mapping of the public API.
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations, the builder HTTP format, score grid).
- model_implementations/: the Ray Serve code that serves trained models; each
  trained model is one file in deploy/helm/rayservice/models/ naming its implementation.
"""
