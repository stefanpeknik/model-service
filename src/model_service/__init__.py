"""RationAI model service: runs pathology models on Ray Serve behind one API.

One top-level package per process, plus what they share:

- api/: the API process: HTTP routes (api/http/ for what provider routes share),
  the model catalog read from the model YAML, and the durable scheduler on
  PostgreSQL (api/scheduler/), which calls builder replicas over HTTP. Raw tiles
  go straight to Ray Serve (`POST /<model-id>/`), not through the API.
- builders/: generic builder code and slides/; the concrete builders in
  builders/implementations/; builders/model_clients/ call and check the models;
  builders/replica/ is the Ray Serve deployment that hosts a builder.
- inference/: the Ray Serve code that serves trained models
  (inference/implementations/, each named by one file in
  deploy/helm/rayservice/models/) and the MLflow loader (artifacts.py).
- providers/: where inputs come from and results go (standalone, EMPAIA); they
  plug into both the API and builder replicas.
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations, the builder HTTP format, score grid).
"""
