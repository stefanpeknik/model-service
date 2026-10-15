"""Check that the model catalog derives bindings from deployment YAML documents."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from builders.contracts import (
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
    UnknownModelError,
)
from model_service.catalog import CatalogError, ModelCatalog


def spatial_document(model_id: str = "prostate-classifier-1") -> dict[str, object]:
    return {
        "inference": {"contract": "spatial-scores"},
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "models.binary_classifier:app",
            "deployments": [
                {
                    "name": "BinaryClassifier",
                    "user_config": {
                        "tile_size": 512,
                        "output_tile_size": 1,
                        "n_channels": 1,
                        "mpp": 0.468,
                        "max_batch_size": 64,
                        "model": {"_target_": "providers.model_provider:mlflow"},
                    },
                }
            ],
        },
    }


def embedding_document(model_id: str = "virchow2") -> dict[str, object]:
    return {
        "inference": {
            "contract": "tile-embedding",
            "mpp": 0.5,
            "summaries": ["global", "local_mean"],
        },
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "models.virchow2:app",
            "deployments": [
                {"name": "Virchow2", "user_config": {"tile_size": 224}},
            ],
        },
    }


class ModelCatalogTests(unittest.TestCase):
    def test_spatial_parameters_come_from_user_config(self) -> None:
        catalog = ModelCatalog.from_documents([spatial_document()])

        self.assertEqual(
            catalog.get("prostate-classifier-1"),
            SpatialScoresModel(
                id="prostate-classifier-1",
                tile_size=512,
                output_tile_size=1,
                n_channels=1,
                mpp=0.468,
            ),
        )

    def test_embedding_combines_user_config_and_inference_declaration(self) -> None:
        catalog = ModelCatalog.from_documents([embedding_document()])

        self.assertEqual(
            catalog.get("virchow2"),
            TileEmbeddingModel(
                id="virchow2",
                tile_size=224,
                mpp=0.5,
                summaries=("global", "local_mean"),
            ),
        )

    def test_compatible_builders_follow_the_contract(self) -> None:
        catalog = ModelCatalog.from_documents(
            [spatial_document(), embedding_document()]
        )

        self.assertEqual(
            ModelCatalog.compatible_builders(catalog.get("prostate-classifier-1")),
            ["heatmap"],
        )
        self.assertEqual(
            ModelCatalog.compatible_builders(catalog.get("virchow2")),
            ["tile-embeddings"],
        )
        self.assertEqual(
            [model.contract for model in catalog],
            [InferenceContract.SPATIAL_SCORES, InferenceContract.TILE_EMBEDDING],
        )
        self.assertEqual(len(catalog), 2)

    def test_unknown_model_is_a_lookup_error(self) -> None:
        catalog = ModelCatalog.from_documents([spatial_document()])

        with self.assertRaises(UnknownModelError):
            catalog.get("missing")

    def test_reads_rendered_json_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model-catalog.json"
            path.write_text(json.dumps([embedding_document()]), encoding="utf-8")

            catalog = ModelCatalog.from_file(path)

            self.assertEqual(catalog.get("virchow2").tile_size, 224)
            with self.assertRaises(CatalogError):
                ModelCatalog.from_file(Path(directory) / "missing.json")
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(CatalogError):
                ModelCatalog.from_file(path)

    def test_invalid_declarations_are_rejected(self) -> None:
        def changed(document: dict[str, object], change: object) -> object:
            copy = deepcopy(document)
            change(copy)  # type: ignore[operator]
            return copy

        def serve(document: dict[str, object]) -> dict[str, object]:
            return document["serve"]  # type: ignore[return-value]

        def user_config(document: dict[str, object]) -> dict[str, object]:
            return serve(document)["deployments"][0]["user_config"]  # type: ignore[index]

        def inference(document: dict[str, object]) -> dict[str, object]:
            return document["inference"]  # type: ignore[return-value]

        cases = {
            "wrong route prefix": [
                changed(
                    spatial_document(),
                    lambda d: serve(d).update(route_prefix="/elsewhere"),
                )
            ],
            "missing user_config key": [
                changed(spatial_document(), lambda d: user_config(d).pop("mpp"))
            ],
            "mpp declared twice": [
                changed(spatial_document(), lambda d: inference(d).update(mpp=0.5))
            ],
            "missing embedding mpp": [
                changed(embedding_document(), lambda d: inference(d).pop("mpp"))
            ],
            "missing embedding summaries": [
                changed(embedding_document(), lambda d: inference(d).pop("summaries"))
            ],
            "unknown contract": [
                changed(
                    spatial_document(),
                    lambda d: inference(d).update(contract="segmentation-mask"),
                )
            ],
            "invalid parameter": [
                changed(spatial_document(), lambda d: user_config(d).update(mpp=-1))
            ],
            "no deployments": [
                changed(spatial_document(), lambda d: serve(d).update(deployments=[]))
            ],
            "unknown top-level key": [
                changed(spatial_document(), lambda d: d.update(extra=True))
            ],
            "duplicate model IDs": [spatial_document(), spatial_document()],
            "not a list": {"models": []},
        }
        for name, documents in cases.items():
            with self.subTest(name), self.assertRaises(CatalogError):
                ModelCatalog.from_documents(documents)


if __name__ == "__main__":
    unittest.main()


# Methods a builder calls through a deployment handle, per contract.
_HANDLE_METHODS = {
    InferenceContract.SPATIAL_SCORES: ("get_config", "predict"),
    InferenceContract.TILE_EMBEDDING: ("get_config", "embed"),
}
MODEL_FILES = sorted(
    Path(__file__).parents[1].joinpath("helm/rayservice/models").rglob("*.yaml")
)


class ModelImplementationTests(unittest.TestCase):
    def test_every_model_file_points_to_an_implementation_of_its_contract(
        self,
    ) -> None:
        import importlib

        import yaml

        self.assertTrue(MODEL_FILES)
        for path in MODEL_FILES:
            with self.subTest(path.name):
                document = yaml.safe_load(path.read_text(encoding="utf-8"))
                contract = InferenceContract(document["inference"]["contract"])
                module_name = document["serve"]["import_path"].partition(":")[0]
                deployment = getattr(
                    importlib.import_module(module_name),
                    document["serve"]["deployments"][0]["name"],
                )
                implementation = deployment.func_or_class
                for method in _HANDLE_METHODS[contract]:
                    self.assertTrue(
                        callable(getattr(implementation, method, None)),
                        f"{module_name} lacks {method} for {contract}",
                    )
