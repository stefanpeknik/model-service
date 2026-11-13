"""RationAI model service: runs pathology models and builders on Ray Serve.

- builders/: the whole-slide builders and how to declare one.
- misc/: tissue detection the heatmap builder uses.
- deployed_models/: the clients that call the deployed models and check them
  (model_clients/).
- contracts/: data shapes processes hand each other (invocation, result, failures, model
  declarations).
- model_implementations/: the Ray Serve code that serves trained models; each
  trained model is one file in deploy/helm/rayservice/applications/ naming its
  implementation.
"""
