"""The API process: public HTTP routes, the process wiring and the scheduler it runs.

- wiring.py: build_api_process, builds the process (catalog, scheduler, providers).
- http_app.py: create_app, assembles the FastAPI app from routes/.
- routes/: one module per URL prefix; request_bodies.py holds their bodies.
- http_errors.py: the public error envelope.
- submission.py: admission shared by routes, providers' too (idempotency key, refusals).
- raw_inference_proxy.py: forwards raw tiles to models.
- synchronous_runs.py: runs a builder while the caller waits (SynchronousBuilderRunner).
- capacity.py, health.py: Ray Serve state for executions and readiness.
"""
