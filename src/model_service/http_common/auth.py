"""Who is calling, and may they do this? Pluggable; nothing is enforced by default.

An `Authenticator` turns a request into a `Principal` or rejects it (401). An
`Authorizer` decides whether that principal may perform an `Action`, on the
execution it names if any (403). The public API and each provider's routes
get their own `Access`, so a provider's service token (an external job service's, say) and
direct users can be checked differently. All default to letting everyone in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fastapi import Depends, Request

from model_service.http_common.errors import ApiErrorCode, http_error


class Action(StrEnum):
    """What a caller wants to do; the authorizer decides per action."""

    READ_CATALOG = "read_catalog"
    RUN = "run"
    SCHEDULE = "schedule"
    READ_EXECUTION = "read_execution"
    CANCEL_EXECUTION = "cancel_execution"
    # A job a provider's own route submits, e.g. an external job service.
    SUBMIT_PROVIDER_JOB = "submit_provider_job"


@dataclass(frozen=True, slots=True)
class Principal:
    """The caller; `subject` is None for anonymous requests."""

    subject: str | None


class AuthenticationError(Exception):
    """The request carries no acceptable credentials."""


class AuthorizationError(Exception):
    """The principal may not perform the action."""


class Authenticator(ABC):
    """Identifies the caller of a request."""

    @abstractmethod
    async def authenticate(self, request: Request) -> Principal:
        """Identify the caller of a request.

        Args:
            request: The incoming HTTP request.

        Returns:
            The calling principal.

        Raises:
            AuthenticationError: If the request carries no acceptable credentials.
        """
        ...


class Authorizer(ABC):
    """Decides whether a caller may perform an action."""

    @abstractmethod
    async def authorize(
        self, principal: Principal, action: Action, execution_id: str | None
    ) -> None:
        """Decide whether the principal may perform the action; return to allow.

        Args:
            principal: The authenticated caller.
            action: What the caller wants to do.
            execution_id: The execution the route names, or None.

        Raises:
            AuthorizationError: If the principal may not perform the action.
        """
        ...


class AllowAnonymous(Authenticator):
    """Authenticator that accepts every request as anonymous."""

    async def authenticate(self, request: Request) -> Principal:
        """Accept every request as anonymous."""
        return Principal(None)


class AllowAll(Authorizer):
    """Authorizer that allows every action."""

    async def authorize(
        self, principal: Principal, action: Action, execution_id: str | None
    ) -> None:
        """Allow every action."""
        return None


@dataclass(frozen=True, slots=True)
class Access:
    """The API's authentication and authorization; open by default.

    Attributes:
        authenticator: Identifies callers.
        authorizer: Decides what they may do.
    """

    authenticator: Authenticator = field(default_factory=AllowAnonymous)
    authorizer: Authorizer = field(default_factory=AllowAll)

    def require(self, action: Action) -> Any:
        """Make a route dependency that authenticates the caller and authorizes `action`.

        The check runs on the route's `execution_id` path parameter if it has one.
        The dependency answers 401 (with `WWW-Authenticate: Bearer`) when
        authentication fails and 403 when authorization fails.

        Args:
            action: The action the route performs.

        Returns:
            A FastAPI `Depends` marker resolving to the caller's `Principal`.
        """

        async def check(request: Request) -> Principal:
            """Authenticate the caller and authorize the action.

            Args:
                request: The incoming request.

            Returns:
                The caller's principal.

            Raises:
                HTTPException: 401 when authentication fails, 403 when the action is
                    not allowed.
            """
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
