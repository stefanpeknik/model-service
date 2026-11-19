"""Data shapes processes hand each other; every process may import them.

- invocation.py: a builder invocation (inputs, access, configuration, deadline).
- caller_request.py: the request a caller sends, which becomes an invocation.
- result.py: a builder's result.
- model_declarations.py: what a model of each inference contract declares.
- builder_http.py: the builder deployment's HTTP request and error format.
- failures.py: failure codes and errors every process shares.
- log_context.py: the JSON log format and the execution id log lines carry.
- json_values.py: strict JSON values.
"""
