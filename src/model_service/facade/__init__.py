"""Contain the public HTTP application and production dependency wiring.

routes/ holds one module per URL prefix and api.py assembles them. app.py wires
the model catalog and HTTP dispatch to Ray Serve.
"""
