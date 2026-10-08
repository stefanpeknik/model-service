"""Contain the prostate whole-slide heatmap workflow, tile processing, Ray predictor, and Serve ingress.

prostate.py atomically publishes the completed TIFF; processor.py owns tile computation
and geometry; prostate_serve.py wires the deployed replica.
"""
