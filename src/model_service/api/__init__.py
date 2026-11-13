"""The API process: public HTTP routes and the process wiring.

- main.py: what uvicorn runs (`app`).
- app_factory.py: create_app assembles the FastAPI app from routes/;
  build_api_process wires the catalog and builder dispatch into it.
- routes/: one module per URL prefix, with its request bodies.
- synchronous_runs.py: runs a builder while the caller waits.
- health.py: Ray Serve readiness.
"""
