"""Contain the public HTTP application and production dependency wiring.

api.py defines routes; runtime.py assembles PostgreSQL, provider adapters, the worker,
and HTTP dispatch to Ray Serve.
"""
