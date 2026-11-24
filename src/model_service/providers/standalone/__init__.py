"""The standalone provider: callers name files on shared storage and read results there.

- provider.py: the provider, loaded by both processes; shared-storage input access.
- adapter.py: admission and finish, control plane only.
- shared_storage.py: paths below the allowed roots and the attempts directory.
"""
