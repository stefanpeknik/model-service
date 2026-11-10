"""Model implementations: the Ray Serve code that serves trained models.

A model is a trained model with an ID, declared by one file in
deploy/helm/rayservice/models/ (weights, tile size, mpp, inference contract)
that names the implementation serving it. An implementation here serves every
model of its kind: binary_classifier.py any binary classifier, virchow2.py the
Virchow2 foundation model. inference/ holds the catalog of models and the
clients builders call them with.
"""
