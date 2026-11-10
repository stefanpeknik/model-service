"""Where builder inputs come from and results go: standalone shared storage and EMPAIA.

- provider.py: Provider, and a list of every method a provider writes and the
  module that defines each (scheduler/provider_adapter.py, builders/input_access.py).
- registry.py: the providers this deployment composes.
- standalone/: the standalone provider and its shared-storage file access.
- empaia/: the EMPAIA provider.
"""
