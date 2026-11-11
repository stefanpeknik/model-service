"""The Ray Serve deployment that hosts one builder: what runs inside the replica.

- serve.py: the Ray Serve ingress (Helm imports serve:app).
- runner.py: runs one invocation: checks, deadline, cleanup.
- http_handler.py: answers a run request and maps exceptions to HTTP errors.

Its HTTP format is contracts/builder_http.py; the client that calls it is
scheduler/builder_dispatch.py.
"""
