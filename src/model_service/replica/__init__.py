"""The Ray Serve deployment that hosts one builder, and the client that calls it.

- serve.py: the Ray Serve ingress (Helm imports serve:app).
- runner.py: runs one invocation: checks, deadline, cleanup.
- protocol.py: the request and error format of its HTTP API.
- client.py: how the facade and scheduler call a replica.
- settings.py: replica settings (threads, concurrency).
"""
