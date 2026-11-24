"""Data shapes processes hand each other; every process may import them.

- invocation.py: an invocation (inputs, access, configuration, deadline).
- caller_request.py: the request a caller sends, which becomes an invocation.
- result.py: a run's result.
- pipeline_http.py: the pipeline runner's HTTP request and error format.
- failures.py: failure codes and errors every process shares.
- log_context.py: the JSON log format and the execution id log lines carry.
- json_values.py: strict JSON values.
"""
