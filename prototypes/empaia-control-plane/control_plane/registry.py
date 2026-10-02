from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    from control_plane.domain import AppMapping


class AppRegistry:
    """Allow-list of registered EMPAIA apps controlled by the deployment."""

    def __init__(self, mappings: Iterable[AppMapping]) -> None:
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
        return self._mappings.get((app_id, app_version))
