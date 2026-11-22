"""Model implementations: the Ray Serve code that served trained models before the generic host.

Only semantic_segmentation.py is left, for EPISeg's file in
deploy/helm/rayservice/models/ (weights, tile size, mpp, inference contract).
Every other model is a served-model file in catalog/served-models/ on the
generic host in served_model/.
"""
