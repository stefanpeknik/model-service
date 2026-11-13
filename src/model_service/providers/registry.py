"""The providers this deployment composes, in order; a new provider is added here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.standalone.provider import PROVIDER as STANDALONE


if TYPE_CHECKING:
    from model_service.providers.provider import Provider


PROVIDERS: tuple[Provider, ...] = (STANDALONE,)
