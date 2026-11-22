"""The document kinds of this repo's catalog, in resolution order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.catalog.analyses import Analyses
from model_service.catalog.served_models import ServedModels


if TYPE_CHECKING:
    from model_service.catalog.kinds import DocumentKind


DEFAULT_KINDS: tuple[DocumentKind, ...] = (ServedModels(), Analyses())
"""Served models first: analyses refer to them."""
