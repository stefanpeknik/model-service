"""Check that the model catalog derives bindings from deployment YAML documents."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.contracts.model_declarations import (
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.deployed_models.catalog import CatalogError, ModelCatalog
from tests.support.model_documents import embedding_document, spatial_document


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
            ),
        )

    def test_a_provider_block_is_kept_for_its_provider(self) -> None:
        document = {**spatial_document(), "listing": {"name": "Prostate"}}

        catalog = ModelCatalog.from_documents([document], ["listing", "other"])

        model_id = next(iter(catalog)).id
        self.assertEqual(
            catalog.provider_block("listing"), {model_id: {"name": "Prostate"}}
        )
        self.assertEqual(catalog.provider_block("other"), {})

    def test_a_block_no_provider_takes_is_rejected(self) -> None:
        document = {**spatial_document(), "listing": {"name": "Prostate"}}

        with self.assertRaises(CatalogError):
            ModelCatalog.from_documents([document])

    def test_an_invalid_model_file_is_named_in_the_error(self) -> None:
        broken = spatial_document()
        broken["serve"] = {**broken["serve"], "route_prefix": 7}

        with self.assertRaisesRegex(CatalogError, repr(broken["serve"]["name"])):
            ModelCatalog.from_documents([spatial_document(), broken])

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


MODEL_FILES = sorted(
    Path(__file__).parents[2].joinpath("deploy/helm/rayservice/models").rglob("*.yaml")
)


class ModelImplementationTests(unittest.TestCase):
    def test_every_model_file_points_to_an_implementation_of_its_contract(
        self,
    ) -> None:
        import importlib

        import yaml

        from model_service.deployed_models.model_clients import CLIENTS

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
                # The methods the contract's model client, if any, calls by handle.
                client = CLIENTS.get(contract)
                for method in client.methods if client else ():
                    self.assertTrue(
                        callable(getattr(implementation, method, None)),
                        f"{module_name} lacks {method} for {contract}",
                    )


if __name__ == "__main__":
    unittest.main()
