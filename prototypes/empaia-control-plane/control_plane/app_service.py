"""Call AppService synchronously with file-backed credentials and classified response failures.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO
    from urllib.request import OpenerDirector

    from control_plane.domain import JsonValue


_MAX_RESPONSE_BYTES = 1_048_576
_APPLICATION_ERROR_STATUSES = {400, 404, 409, 422}
_TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class AppServiceError(RuntimeError):
    """Base class that deliberately excludes response bodies and credentials."""


class AppServiceApplicationError(AppServiceError):
    """The EMPAIA job or its declared data cannot satisfy the app contract."""


class AppServiceTransientError(AppServiceError):
    """The fixed AppService dependency may succeed when retried."""


class AppServiceInfrastructureError(AppServiceError):
    """The service integration or response contract is invalid."""


class TokenProvider(Protocol):
    """Supply an AppService credential without coupling HTTP calls to its storage location."""

    def get_token(self) -> str:
        """Return the current credential or raise a classified credential-access error.

        Returns:
            str: Current nonempty AppService credential after checking configured
                credential separation.
        """
        ...


@dataclass(frozen=True, slots=True)
class FileTokenProvider:
    """Reads a mounted credential just before each outbound operation.

    Attributes:
        path: Shared file reference; results carry its path rather than bytes.
        forbidden_paths: Credential files that must differ from the AppService token.
    """

    path: Path
    forbidden_paths: tuple[Path, ...] = ()

    def get_token(self) -> str:
        """Read the credential on every use and verify it differs from configured forbidden trust-domain credentials.

        Returns:
            str: Current nonempty AppService credential after checking configured
                credential separation.

        Raises:
            AppServiceInfrastructureError: The prototype integration settings or
                response contract are invalid.
            AppServiceTransientError: The prototype provider or credential source is
                temporarily unavailable.
        """
        try:
            token = self.path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise AppServiceTransientError(
                "AppService credential is temporarily unavailable"
            ) from error
        if not token:
            raise AppServiceInfrastructureError("AppService credential is empty")
        for path in self.forbidden_paths:
            try:
                forbidden = path.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise AppServiceTransientError(
                    "credential separation cannot be verified"
                ) from error
            if forbidden and secrets.compare_digest(token, forbidden):
                raise AppServiceInfrastructureError(
                    "AppService credential is not isolated from another trust domain"
                )
        return token


@dataclass(frozen=True, slots=True)
class JobConfiguration:
    """Keep provider global and customer configuration sections separate for parameter selection.

    Attributes:
        global_values: Provider global configuration section.
        customer_values: Provider customer configuration section.
    """

    global_values: dict[str, JsonValue]
    customer_values: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DirectAccess:
    """Carry slide identity, required WSI plugin, provider storage path, and direct-access support.

    Attributes:
        slide_id: External whole-slide input identifier.
        storage_path: Provider shared file path or authorized storage locator.
        supported: Whether this provider input supports direct shared access.
        required_plugin_id: WSI representation plugin required to read the provider
            input.
    """

    slide_id: str
    required_plugin_id: str
    storage_path: str
    supported: bool


class _RejectRedirects(HTTPRedirectHandler):
    """Disable redirects so authenticated requests do not silently cross AppService origins."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        """Return no follow-up request for an HTTP redirect.

        Args:
            req: Original urllib request supplied to the redirect hook.
            fp: Original redirect response stream supplied by urllib.
            code: Stable failure identifier, or HTTP status when classifying an upstream
                response.
            msg: Redirect status text supplied by urllib.
            headers: HTTP metadata to filter, send, or include in an error response.
            newurl: Redirect target that this prototype deliberately refuses to follow.

        Returns:
            Request | None: A redirected request; this rejection hook returns no follow-
                up request. None indicates an absent, unsupported, or unmatched value as
                described above.
        """
        return None


class AppServiceClient:
    """Small v3 client pinned to one deployment-configured AppService origin."""

    def __init__(
        self,
        base_url: str,
        token_provider: TokenProvider,
        *,
        timeout_seconds: float = 10.0,
        allow_insecure_http: bool = False,
        opener: OpenerDirector | None = None,
    ) -> None:
        """Validate an HTTP(S) origin and positive timeout, requiring explicit prototype opt-in for plaintext transport.

        Args:
            base_url: AppService base origin and optional path prefix.
            token_provider: Credential source consulted for each prototype AppService
                request.
            timeout_seconds: Maximum wait for cleanup, polling, or a classified
                operation.
            allow_insecure_http: Whether this historical prototype may send credentials
                over plaintext HTTP.
            opener: Optional urllib opener used to inject transport behavior in
                prototype tests.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if timeout_seconds <= 0:
            raise ValueError("AppService timeout must be positive")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("AppService URL must be an HTTP(S) origin with a path")
        if parsed.scheme == "http" and not allow_insecure_http:
            raise ValueError(
                "plaintext AppService transport requires an explicit POC opt-in"
            )
        normalized_path = parsed.path.rstrip("/")
        self._base_url = urlunsplit(
            (parsed.scheme, parsed.netloc, normalized_path, "", "")
        )
        self._tokens = token_provider
        self._timeout_seconds = timeout_seconds
        self._opener = opener or build_opener(_RejectRedirects())

    def get_configuration(self, job_id: str) -> JobConfiguration:
        """GET the job configuration and require valid flat global/customer sections.

        Args:
            job_id: External EMPAIA job identifier.

        Returns:
            JobConfiguration: Validated global and customer provider configuration
                sections.
        """
        value = self._request_json("GET", f"/v3/{quote(job_id, safe='')}/configuration")
        global_values = _flat_configuration_section(value, "global")
        customer_values = _flat_configuration_section(value, "customer")
        return JobConfiguration(global_values, customer_values)

    def get_direct_access(self, job_id: str, input_name: str) -> DirectAccess:
        """GET and validate shared WSI access metadata for a named job input.

        Args:
            job_id: External EMPAIA job identifier.
            input_name: Named external job input to resolve.

        Returns:
            DirectAccess: Validated provider shared path and direct-access support
                metadata.

        Raises:
            AppServiceInfrastructureError: The prototype integration settings or
                response contract are invalid.
        """
        value = self._request_json(
            "GET",
            "/v3/rationai/"
            f"{quote(job_id, safe='')}/inputs/{quote(input_name, safe='')}"
            "/direct-access",
        )
        slide_id = _required_string(value, "slide_id")
        required_plugin_id = _required_string(value, "required_plugin_id")
        storage_path = _required_string(value, "storage_path")
        supported = value.get("direct_access_supported")
        if not isinstance(supported, bool):
            raise AppServiceInfrastructureError(
                "AppService direct-access response has an invalid support flag"
            )
        return DirectAccess(
            slide_id=slide_id,
            required_plugin_id=required_plugin_id,
            storage_path=storage_path,
            supported=supported,
        )

    def report_failure(self, job_id: str, user_message: str) -> None:
        """PUT a nonempty safe failure message of at most 255 characters to the external job.

        Args:
            job_id: External EMPAIA job identifier.
            user_message: Safe external job failure message, bounded by the provider
                contract.

        Returns:
            None: No return value.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if not user_message or len(user_message) > 255:
            raise ValueError(
                "AppService failure message must contain at most 255 chars"
            )
        self._request_json(
            "PUT",
            f"/v3/{quote(job_id, safe='')}/failure",
            {"user_message": user_message},
        )

    def _request_json(
        self, method: str, path: str, body: dict[str, JsonValue] | None = None
    ) -> dict[str, Any]:
        """Send authenticated JSON with a bounded response size, reject redirects, and classify transport, status, and decoding failures.

        Args:
            method: HTTP method used by the request or test client.
            path: File, storage locator, or HTTP endpoint selected by this operation.
            body: Typed or JSON request payload accepted by this boundary.

        Returns:
            dict[str, Any]: Decoded, serialized, or recorded fields described above.

        Raises:
            AppServiceInfrastructureError: The prototype integration settings or
                response contract are invalid.
            AppServiceTransientError: The prototype provider or credential source is
                temporarily unavailable.
        """
        token = self._tokens.get_token()
        payload = (
            json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
            if body is not None
            else None
        )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._base_url}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            self._raise_http_error(error.code)
        except (TimeoutError, URLError, OSError) as error:
            raise AppServiceTransientError(
                "AppService request could not be completed"
            ) from error
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise AppServiceInfrastructureError("AppService response is too large")
        if not raw:
            return {}
        try:
            value: Any = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AppServiceInfrastructureError(
                "AppService returned invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise AppServiceInfrastructureError(
                "AppService response must be a JSON object"
            )
        return value

    @staticmethod
    def _raise_http_error(status_code: int) -> None:
        """Map rejected jobs, temporary statuses, and integration failures to separate prototype error categories.

        Args:
            status_code: HTTP status to return or classify.

        Returns:
            None: No return value.

        Raises:
            AppServiceApplicationError: The prototype provider rejects the external
                job's data or configuration.
            AppServiceInfrastructureError: The prototype integration settings or
                response contract are invalid.
            AppServiceTransientError: The prototype provider or credential source is
                temporarily unavailable.
        """
        if status_code in _APPLICATION_ERROR_STATUSES:
            raise AppServiceApplicationError(
                f"AppService rejected the EMPAIA job (HTTP {status_code})"
            )
        if status_code in _TRANSIENT_STATUSES:
            raise AppServiceTransientError(
                f"AppService is temporarily unavailable (HTTP {status_code})"
            )
        raise AppServiceInfrastructureError(
            f"AppService integration failed (HTTP {status_code})"
        )


def _flat_configuration_section(
    value: dict[str, Any], name: str
) -> dict[str, JsonValue]:
    """Require named primitive-valued configuration entries and reject non-finite numeric values.

    Args:
        value: Value being validated, serialized, converted, or checked by this helper.
        name: Input/output name or configuration field selected by this operation.

    Returns:
        dict[str, JsonValue]: Named primitive provider values, with non-finite numbers
            rejected.

    Raises:
        AppServiceApplicationError: The prototype provider rejects the external job's
            data or configuration.
        AppServiceInfrastructureError: The prototype integration settings or response
            contract are invalid.
    """
    section = value.get(name)
    if not isinstance(section, dict):
        raise AppServiceInfrastructureError(
            f"AppService configuration has no valid {name} section"
        )
    result: dict[str, JsonValue] = {}
    for key, item in section.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item, (str, int, float, bool))
        ):
            raise AppServiceApplicationError(
                "job configuration contains an unsupported parameter"
            )
        if isinstance(item, float) and not (-float("inf") < item < float("inf")):
            raise AppServiceApplicationError(
                "job configuration contains a non-finite number"
            )
        result[key] = item
    return result


def _required_string(value: dict[str, Any], name: str) -> str:
    """Extract a nonempty stripped response field or raise an integration error.

    Args:
        value: Value being validated, serialized, converted, or checked by this helper.
        name: Input/output name or configuration field selected by this operation.

    Returns:
        str: Nonempty stripped provider response field.

    Raises:
        AppServiceInfrastructureError: The prototype integration settings or response
            contract are invalid.
    """
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise AppServiceInfrastructureError(f"AppService response has no valid {name}")
    return item.strip()
