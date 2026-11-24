"""The control plane: public HTTP routes, the process wiring and the scheduler it runs.

- main.py: what uvicorn runs (`app`).
- app_factory.py: create_app assembles the FastAPI app from routes/v1/;
  build_control_plane wires the catalog, scheduler and providers into it.
- routes/v1/: one module per /v1/ URL prefix, with its request bodies.
- http/: auth, error mapping and scheduling calls the control plane and provider routes share.
- catalog_files.py: reads the deployed models and pipelines from the
  mounted catalog ConfigMap; catalog_mount.py reloads it in place when kubelet
  updates the files.
- synchronous_runs.py: runs a pipeline while the caller waits.
- health.py: Ray Serve readiness.
- scheduler/: durable executions on PostgreSQL; it calls the pipeline runner
  over HTTP in dispatch.py.
"""
