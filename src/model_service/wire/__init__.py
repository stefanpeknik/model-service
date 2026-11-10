"""Types that cross process boundaries; every process may import them.

- invocation.py: a builder invocation (inputs, access, configuration, deadline).
- caller_request.py: the request a caller sends, which becomes an invocation.
- result.py: a builder's result; score_grid.py: the heatmap's file format.
- failures.py: failure codes and errors every process shares.
- json_values.py: strict JSON values.
"""
