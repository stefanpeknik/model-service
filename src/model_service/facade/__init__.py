"""The API process: public HTTP routes and the process wiring.

- wiring.py: create_facade, builds the process (catalog, scheduler, providers).
- http_app.py: create_app, assembles the FastAPI app from routes/.
- routes/: one module per URL prefix; requests.py holds their bodies.
- http_errors.py: the public error envelope.
- raw_inference_proxy.py: forwards raw tiles to models.
- synchronous_runs.py: runs a builder while the caller waits (BuilderRunner).
- capacity.py, health.py: Ray Serve state for executions and readiness.
"""
