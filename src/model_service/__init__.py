"""RationAI model service: runs pathology models on Ray Serve behind one API.

- api/: the API process (HTTP routes, wiring). Raw tiles go straight to Ray
  Serve (`POST /<model-id>/`), not through the API.
- scheduler/: durable executions on PostgreSQL, run inside the API process;
  it calls builder replicas over HTTP in builder_dispatch.py (see docs/adr/0001).
- builders/: generic builder code and slides/; the concrete builders in
  builders/implementations/; builders/replica/ is the Ray Serve deployment that
  hosts a builder.
- models/: the deployed models (catalog.py, read from the model YAML), the
  clients that call and check them (clients.py) and the MLflow loader
  (artifacts.py); models/implementations/ is the Ray Serve code that serves
  trained models, each named by one file in deploy/helm/rayservice/models/.
- providers/: where inputs come from and results go (standalone, EMPAIA).
- http_common/: auth, error mapping and scheduling calls the API and provider routes share.
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations, the builder HTTP format, score grid).
"""
