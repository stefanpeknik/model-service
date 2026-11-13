"""RationAI model service: runs pathology models and builders on Ray Serve.

- builders/: the whole-slide builders and how to declare one.
- misc/: tissue detection the heatmap builder uses.
- builders/replica/: the Ray Serve deployment that hosts a builder.
- models/: the clients that call the deployed models and check them
  (model_clients/).
- providers/: where inputs come from and results go: shared storage.
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations, the builder HTTP format).
- models/implementations/: the Ray Serve code that serves trained models; each
  trained model is one file in deploy/helm/rayservice/applications/ naming its
  implementation.
"""
