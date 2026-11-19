"""The API process and the scheduler it runs.

- model_catalog.py: the deployed models, read from the model YAML files.
- scheduler/: durable executions; it calls builder replicas over HTTP in
  builder_dispatch.py.
"""
