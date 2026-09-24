from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from workflow_execution.contracts import JsonValue, WorkflowOutput


type MutationGuard = Callable[[], Awaitable[None]]


class EmpaiaGatewayError(RuntimeError):
    def __init__(
        self,
        safe_message: str,
        *,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True, slots=True)
class EmpaiaJobContext:
    api_url: str
    job_id: str
    credential: str

    def protected_values(self) -> dict[str, JsonValue]:
        return {
            "api_url": self.api_url,
            "job_id": self.job_id,
            "credential": self.credential,
        }


class EmpaiaGateway(Protocol):
    def set_before_mutation(self, callback: MutationGuard) -> None: ...

    async def get_configuration(self) -> dict[str, JsonValue]: ...

    async def materialize_input(self, name: str, destination: str) -> None: ...

    async def publish_output(self, name: str, output: WorkflowOutput) -> None: ...

    async def finalize(self) -> None: ...

    async def report_failure(self, message: str) -> None: ...

    async def close(self) -> None: ...


class EmpaiaGatewayFactory(Protocol):
    def __call__(self, context: EmpaiaJobContext) -> EmpaiaGateway: ...


def load_gateway_factory(import_path: str) -> EmpaiaGatewayFactory:
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("EMPAIA gateway factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError("EMPAIA gateway factory is not callable")
    return factory
