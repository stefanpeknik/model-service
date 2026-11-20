"""Where builder inputs come from and results go: standalone shared storage.

- provider.py: the provider contract: Provider, ProviderAdapter (how the
  scheduler admits and finishes its jobs), and every method a provider writes.
- registry.py: the providers this deployment composes.
- standalone/: the standalone provider and its shared-storage file access.
"""
