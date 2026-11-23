"""The served model's Serve routes, called on the ingress app without a Ray cluster."""

from __future__ import annotations

from typing import Any

from model_service.served_model import serve


class Host:
    """A host stand-in that answers `describe()`."""

    def describe(self) -> dict[str, Any]:
        return {
            "id": "m1",
            "digest": "abc",
            "weights_revision": "/m.onnx",
            "execution_providers": ["CPUExecutionProvider"],
        }


def test_get_describe_answers_what_the_replica_runs() -> None:
    replica = object.__new__(serve.ServedModelDeployment.func_or_class)
    replica.host = Host()  # type: ignore[assignment]

    route = next(
        r for r in serve.fastapi.routes if getattr(r, "path", "") == "/describe"
    )
    described = route.endpoint(replica)  # type: ignore[attr-defined]

    assert route.methods == {"GET"}  # type: ignore[attr-defined]
    assert described == Host().describe()
