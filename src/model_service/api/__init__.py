"""The API process: public HTTP routes, the process wiring and the scheduler it runs.

- main.py: what uvicorn runs (`app`).
- app_factory.py: create_app assembles the FastAPI app from routes/;
  build_api_process wires the catalog, scheduler and providers into it.
- routes/: one module per URL prefix; request_bodies.py holds their bodies.
- http_errors.py: the public error envelope.
- scheduling_http.py: HTTP plumbing for scheduling routes, providers' too
  (Idempotency-Key header, scheduler refusals as HTTP errors).
- synchronous_runs.py: runs a builder while the caller waits (SynchronousBuilderRunner).
- health.py: Ray Serve readiness.
"""
