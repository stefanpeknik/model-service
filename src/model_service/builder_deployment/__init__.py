"""The Ray Serve deployment that hosts one builder, and the client that calls it.

- serve.py: the Ray Serve ingress (Helm imports serve:app).
- runner.py: runs one invocation: checks, deadline, cleanup.
- protocol.py: the request and error format of its HTTP API.
- client.py: HTTPBuilderDispatch, how the API and scheduler call a replica
  (implements scheduler/builder_dispatch.py).
- settings.py: replica settings (a builder's BuilderSettings plus storage roots).
"""
