"""Where builder inputs come from and results go: standalone shared storage and EMPAIA.

A provider implements builders/input_access.py (local input files for a
builder) and scheduler/provider_adapter.py (admitting and finishing jobs).

- provider.py: Provider, what a provider contributes to each process.
- registry.py: the providers this deployment composes.
- standalone/: the standalone provider and its shared-storage file access.
- empaia/: the EMPAIA provider.
"""
