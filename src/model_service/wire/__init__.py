"""Types that cross process boundaries; every process may import them.

- invocation.py: a builder invocation (inputs, access, configuration, deadline).
- builder_request.py: the public request a builder invocation is made from.
- result.py: a builder's result; score_grid.py: the heatmap's file format.
- failures.py: failure codes and errors every process shares.
- json_values.py: strict JSON values.
"""
