"""RationAI model service: runs pathology models on Ray Serve behind one API.

- api/: the API process (HTTP routes, wiring, the model catalog read from the
  model YAML). Raw tiles go straight to Ray Serve (`POST /<model-id>/`), not
  through the API.
- scheduler/: durable executions on PostgreSQL, run inside the API process;
  it calls builder replicas over HTTP in builder_dispatch.py (see docs/adr/0001).
- builders/: generic builder code and slides/; the concrete builders in
  builders/implementations/; builders/model_clients/ call and check the models;
  builders/replica/ is the Ray Serve deployment that hosts a builder.
- inference/: the Ray Serve code that serves trained models
  (inference/implementations/, each named by one file in
  deploy/helm/rayservice/models/) and the MLflow loader (artifacts.py).
- providers/: where inputs come from and results go (standalone, EMPAIA).
- api/http/: auth, error mapping and scheduling calls the API and provider routes share.
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations, the builder HTTP format, score grid).
"""
