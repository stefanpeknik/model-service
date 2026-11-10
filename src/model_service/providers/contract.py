"""Everything a provider implements or receives, in one place to read.

A provider is a `Provider` (provider.py) that contributes, per process:

- to the facade, a `ProviderAdapter` that admits and finishes its jobs
  (defined in scheduler/provider_adapter.py, because the scheduler calls it);
- to builder replicas, an `InputAccessFactory` that turns its inputs into local
  files (defined in builders/input_access.py, because builders consume it).

The interfaces live with their consumers; this module re-exports them.
"""

from model_service.builders.input_access import InputAccess, InputAccessFactory
from model_service.providers.provider import FacadeContribution, Provider
from model_service.scheduler.provider_adapter import (
    PreparedInvocation,
    Priority,
    ProviderAdapter,
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
    SubmissionIdentity,
)


__all__ = [
    "FacadeContribution",
    "InputAccess",
    "InputAccessFactory",
    "PreparedInvocation",
    "Priority",
    "Provider",
    "ProviderAdapter",
    "ProviderCancellation",
    "ProviderFailure",
    "ProviderFinishFailureError",
    "ProviderSuccess",
    "SubmissionIdentity",
]
