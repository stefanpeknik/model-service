"""RationAI model service: runs pathology models on Ray Serve.

- inference/: the Ray Serve code that serves trained models
  (inference/implementations/, each named by an application file in
  deploy/helm/rayservice/applications/) and the MLflow loader (artifacts.py).
"""
