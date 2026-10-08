"""Allow only configured EMPAIA app/version mappings during prototype admission.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    from control_plane.domain import AppMapping


class AppRegistry:
    """Allow-list of registered EMPAIA apps controlled by the deployment."""

    def __init__(self, mappings: Iterable[AppMapping]) -> None:
        """Require a nonempty mapping set with unique app/version pairs.

        Args:
            mappings: Configured app contracts or provider-to-worker storage
                translations.

        Raises:
            ValueError: A value fails the validation described above.
        """
        self._mappings: dict[tuple[str, str], AppMapping] = {}
        for mapping in mappings:
            key = (mapping.app_id, mapping.app_version)
            if key in self._mappings:
                raise ValueError(
                    f"duplicate app mapping for {mapping.app_id!r} "
                    f"{mapping.app_version!r}"
                )
            self._mappings[key] = mapping
        if not self._mappings:
            raise ValueError("at least one app mapping is required")

    def get(self, app_id: str, app_version: str) -> AppMapping | None:
        """Return the configured mapping for an exact app/version pair or None if disabled.

        Args:
            app_id: Registered EMPAIA application identifier.
            app_version: Exact registered application version to select.

        Returns:
            AppMapping | None: Enabled app/version contract selected by the registry.
                None indicates an absent, unsupported, or unmatched value as described
                above.
        """
        return self._mappings.get((app_id, app_version))
