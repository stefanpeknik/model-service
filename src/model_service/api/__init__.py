"""The API process: public HTTP routes, the process wiring and the scheduler it runs.

- main.py: what uvicorn runs (`app`).
- app_factory.py: create_app assembles the FastAPI app from routes/v1/;
  build_api_process wires the catalog, scheduler and providers into it.
- routes/v1/: one module per /v1/ URL prefix, with its request bodies.
- http/: auth, error mapping and scheduling calls the API and provider routes share.
- model_catalog.py: the deployed models and their bindings, from served-model
  files and, until they migrate, old model files.
- catalog_files.py: reads them from the mounted catalog ConfigMap;
  catalog_mount.py reloads it in place when kubelet updates the files.
- synchronous_runs.py: runs a builder while the caller waits.
- health.py: Ray Serve readiness.
- scheduler/: durable executions on PostgreSQL; it calls builder replicas over
  HTTP in builder_dispatch.py.
"""
