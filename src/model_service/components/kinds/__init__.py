"""The seven component kinds, one small base class each."""

from model_service.components.kinds.converter import InputConverter
from model_service.components.kinds.loader import Loader
from model_service.components.kinds.publisher import Commit, Publisher
from model_service.components.kinds.runtime import (
    BatchFunction,
    Runtime,
    RuntimeContext,
)
from model_service.components.kinds.sink import Sink, SinkOptions
from model_service.components.kinds.source import Opened, Source
from model_service.components.kinds.step import Step


KINDS = (Loader, Runtime, Step, Source, Sink, InputConverter, Publisher)

__all__ = [
    "KINDS",
    "BatchFunction",
    "Commit",
    "InputConverter",
    "Loader",
    "Opened",
    "Publisher",
    "Runtime",
    "RuntimeContext",
    "Sink",
    "SinkOptions",
    "Source",
    "Step",
]
