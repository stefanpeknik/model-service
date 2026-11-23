"""Helpers for testing a component in isolation, without a cluster.

- `fake_slide(path)`: a small tiled TIFF that OpenSlide and pyvips open;
- `fake_frame()`: the frame row a slide source gives, for sinks;
- `fake_model(output=…)`: a served-model stand-in for `model.` dimensions,
  batch calls and the host's `infer(tile)`;
- `run_component(component, data)`: run a step, source or sink once;
- `FakeGateway(directory)`: a fake EMPAIA job for converters and publishers;
- `conformance.check(component, work_dir)`: the suite of the component's kind.

Heavy libraries (pyvips) are imported only when a helper needs them.
"""

from model_service.testing import conformance
from model_service.testing.empaia import FakeGateway
from model_service.testing.models import FakeModel, fake_model
from model_service.testing.running import SourceRun, fake_batch, run_component
from model_service.testing.slides import fake_frame, fake_slide


__all__ = [
    "FakeGateway",
    "FakeModel",
    "SourceRun",
    "conformance",
    "fake_batch",
    "fake_frame",
    "fake_model",
    "fake_slide",
    "run_component",
]
