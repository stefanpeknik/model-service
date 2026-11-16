"""The API process: public HTTP routes, the process wiring and the scheduler it runs.

- main.py: what uvicorn runs (`app`).
- app_factory.py: create_app assembles the FastAPI app from routes/v1/;
  build_api_process wires the catalog, scheduler and providers into it.
- routes/v1/: one module per /v1/ URL prefix, with its request bodies.
- http/: auth, error mapping and scheduling calls the API and provider routes share.
- model_catalog.py: the deployed models, read from the model YAML files.
- synchronous_runs.py: runs a builder while the caller waits.
- health.py: Ray Serve readiness.
"""
