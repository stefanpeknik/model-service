"""Who is calling, and may they do this? Pluggable; nothing is enforced by default.

An `Authenticator` turns a request into a `Principal` or rejects it (401). An
`Authorizer` decides whether that principal may perform an `Action`, on the
execution it names if any (403). The public API and the EMPAIA route each get
their own `Access`, so EMPAIA's service token and direct users can be checked
differently. Both default to letting everyone in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import Depends, Request

from model_service.facade.routes.common import ApiErrorCode, http_error


if TYPE_CHECKING:
    from collections.abc import Mapping


class Action(StrEnum):
    READ_CATALOG = "read_catalog"
    RUN = "run"
    SCHEDULE = "schedule"
    READ_EXECUTION = "read_execution"
    CANCEL_EXECUTION = "cancel_execution"
    SUBMIT_EMPAIA_JOB = "submit_empaia_job"


@dataclass(frozen=True, slots=True)
class Principal:
    """The caller; `subject` is None for anonymous requests."""

    subject: str | None
    claims: Mapping[str, object] = field(default_factory=dict)


ANONYMOUS = Principal(None)


class AuthenticationError(Exception):
    """The request carries no acceptable credentials."""


class AuthorizationError(Exception):
    """The principal may not perform the action."""


class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> Principal: ...


class Authorizer(Protocol):
    async def authorize(
        self, principal: Principal, action: Action, execution_id: str | None
    ) -> None:
        """Return to allow; raise AuthorizationError to refuse."""
        ...


class AllowAnonymous:
    async def authenticate(self, request: Request) -> Principal:
        return ANONYMOUS


class AllowAll:
    async def authorize(
        self, principal: Principal, action: Action, execution_id: str | None
    ) -> None:
        return None


@dataclass(frozen=True, slots=True)
class Access:
    authenticator: Authenticator = field(default_factory=AllowAnonymous)
    authorizer: Authorizer = field(default_factory=AllowAll)

    def require(self, action: Action) -> Any:
        """A route dependency checking `action`, on the route's `execution_id` if it has one."""

        async def check(request: Request) -> Principal:
            try:
                principal = await self.authenticator.authenticate(request)
            except AuthenticationError as error:
                raise http_error(
                    401,
                    ApiErrorCode.UNAUTHENTICATED,
                    "authentication is required",
                    headers={"WWW-Authenticate": "Bearer"},
                ) from error
            try:
                await self.authorizer.authorize(
                    principal, action, request.path_params.get("execution_id")
                )
            except AuthorizationError as error:
                raise http_error(
                    403, ApiErrorCode.FORBIDDEN, "this action is not allowed"
                ) from error
            return principal

        return Depends(check)
