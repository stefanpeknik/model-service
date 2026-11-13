"""Data shapes processes hand each other; every process may import them.

- invocation.py: a builder invocation (inputs, access, configuration, deadline).
- result.py: a builder's result; score_grid.py: the heatmap's file format.
- model_declarations.py: what a model of each inference contract declares.
- builder_http.py: the builder deployment's HTTP request and error format.
- failures.py: failure codes and errors every process shares.
- json_values.py: strict JSON values.
"""
