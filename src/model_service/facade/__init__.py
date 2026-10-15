"""Contain the public HTTP application and production dependency wiring.

routes/ holds one module per URL prefix and api.py assembles them. app.py wires
PostgreSQL, provider adapters, the worker and HTTP dispatch to Ray Serve.
"""
