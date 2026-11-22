"""`ProvGigaPath` and the generic host give bit-identical results.

Both run Prov-GigaPath's files (today's model file and the catalog's
served-model file) on CPU, with only the weights swapped for a small random
ViT shaped like Prov-GigaPath's tile encoder. The tiles keep today's 224 px,
so the host's `pre:` steps must reproduce the resize to 256 and the crop to
224 exactly. Deleted with `ProvGigaPath` once Prov-GigaPath passes its live
check.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lz4.frame
import numpy as np
import pytest
import timm
from PIL import Image

from model_service.catalog import load_catalog
from model_service.inference.implementations import prov_gigapath
from model_service.served_model import serve
from model_service.served_model.deployment import serve_application
from tests.support import vit_weights
from tests.support.model_documents import model_file
from tests.support.off_serve import OffServe, off_serve


if TYPE_CHECKING:
    from collections.abc import Iterator


REPO = Path(__file__).resolve().parents[2]
TILE = 224
BATCH = 4


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return vit_weights.prov_gigapath_like(tmp_path_factory.mktemp("prov-gigapath"))


@pytest.fixture(scope="module")
def old(weights: Path) -> Iterator[OffServe]:
    document = model_file(REPO / "deploy/helm/rayservice/models/prov-gigapath.yaml")
    config = document["deployment"]["user_config"] | {"max_batch_size": BATCH}
    create_model = timm.create_model

    def from_stand_in(name: str, **kwargs: Any) -> Any:
        assert name == "hf_hub:prov-gigapath/prov-gigapath"
        return create_model(f"local-dir:{weights}", **kwargs)

    with contextlib.ExitStack() as stack:
        # Only ProvGigaPath's `reconfigure` builds its model from the stand-in.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(timm, "create_model", from_stand_in)
            running = stack.enter_context(
                off_serve(prov_gigapath.ProvGigaPath, prov_gigapath.fastapi, config)
            )
        yield running


@pytest.fixture(scope="module")
def new(weights: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OffServe]:
    entry = load_catalog(REPO / "catalog").entries("served-models")["prov-gigapath"]
    (deployment,) = serve_application(
        entry, cache_root=str(tmp_path_factory.mktemp("new-cache"))
    )["deployments"]
    config: dict[str, Any] = deployment["user_config"]
    declaration = config["declaration"]
    config["declaration"] = declaration | {
        "weights": {"use": "local", "path": str(weights)},
        "batching": declaration["batching"] | {"max_size": BATCH},
    }
    with off_serve(serve.ServedModelDeployment, serve.fastapi, config) as running:
        yield running


def tiles() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    batch = list(rng.integers(0, 256, (BATCH - 2, 3, TILE, TILE), dtype=np.uint8))
    return [np.zeros((3, TILE, TILE), np.uint8), *batch, np.full_like(batch[0], 255)]


def channels_last(tile: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(tile.transpose(1, 2, 0))


def bits(arrays: list[np.ndarray]) -> list[tuple[str, tuple[int, ...], bytes]]:
    return [(str(a.dtype), a.shape, a.tobytes()) for a in arrays]


def test_embed_gives_the_same_features_alone_and_in_a_batch(
    old: OffServe, new: OffServe
) -> None:
    batch = tiles()
    model = old.instance
    features = old.batch(
        "predict", [model.transforms(Image.fromarray(channels_last(t))) for t in batch]
    )
    batched = [f.float().numpy() for f in features]
    alone = [old.call("embed", tile) for tile in batch]

    assert bits(new.gather("embed", batch)) == bits(batched)
    assert bits([new.call("embed", tile) for tile in batch]) == bits(alone)


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-output-dtype": "float16"}],
    ids=["float32", "float16"],
)
def test_the_raw_route_replies_with_the_same_bytes(
    old: OffServe, new: OffServe, headers: dict[str, str]
) -> None:
    for tile in tiles():
        body = lz4.frame.compress(channels_last(tile))

        expected, got = old.post(body, headers), new.post(body, headers)

        assert expected.status_code == 200
        assert (got.status_code, got.headers["x-output-shape"], got.content) == (
            expected.status_code,
            expected.headers["x-output-shape"],
            expected.content,
        )


def test_get_config_reports_the_same_fields(old: OffServe, new: OffServe) -> None:
    assert new.call("get_config") == old.call("get_config")
