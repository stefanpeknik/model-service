"""Where builder inputs come from and results go: standalone shared storage and EMPAIA.

- provider.py: Provider, and a list of every method a provider writes.
- job_lifecycle.py: ProviderAdapter, how the scheduler admits and finishes a
  provider's jobs.
- registry.py: the providers this deployment composes.
- standalone/: the standalone provider and its shared-storage file access.
- empaia/: the EMPAIA provider.
"""
