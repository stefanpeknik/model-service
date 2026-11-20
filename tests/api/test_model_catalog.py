"""Check that the model catalog derives bindings from deployment YAML documents."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.api.model_catalog import CatalogError, ModelCatalog
from model_service.contracts.model_declarations import (
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from tests.support.model_documents import (
    embedding_document,
    implementation,
    model_file,
    spatial_document,
)


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

    def test_a_provider_block_is_kept_for_its_provider(self) -> None:
        document = {**spatial_document(), "listing": {"name": "Prostate"}}

        catalog = ModelCatalog.from_documents([document], ["listing", "other"])

        model_id = next(iter(catalog)).id
        self.assertEqual(
            catalog.provider_block("listing"), {model_id: {"name": "Prostate"}}
        )
        self.assertEqual(catalog.provider_block("other"), {})

    def test_a_model_file_left_out_is_named_in_an_error_log(self) -> None:
        broken = spatial_document("broken")
        broken["deployment"] = {"user_config": 7}

        with self.assertLogs("model_service.api.model_catalog", "ERROR") as logs:
            catalog = ModelCatalog.from_documents([embedding_document(), broken])

        self.assertEqual([model.id for model in catalog], ["virchow2"])
        self.assertIn("'broken'", logs.output[0])

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

    def test_invalid_model_files_are_left_out(self) -> None:
        def changed(document: dict[str, object], change: object) -> object:
            copy = deepcopy(document)
            change(copy)  # type: ignore[operator]
            return copy

        def user_config(document: dict[str, object]) -> dict[str, object]:
            return document["deployment"]["user_config"]  # type: ignore[index]

        cases = {
            "missing user_config key": changed(
                spatial_document(), lambda d: user_config(d).pop("mpp")
            ),
            "missing embedding mpp": changed(
                embedding_document(), lambda d: user_config(d).pop("mpp")
            ),
            "missing embedding summaries": changed(
                embedding_document(), lambda d: user_config(d).pop("summaries")
            ),
            "unknown contract": changed(
                spatial_document(), lambda d: d.update(contract="segmentation-mask")
            ),
            "invalid parameter": changed(
                spatial_document(), lambda d: user_config(d).update(mpp=-1)
            ),
            "no deployment": changed(spatial_document(), lambda d: d.pop("deployment")),
            "no implementation": changed(
                spatial_document(), lambda d: d.pop("implementation")
            ),
            # A module path, not `module.Class` in inference/implementations.
            "an import path as implementation": changed(
                spatial_document(),
                lambda d: d.update(
                    implementation="model_service.inference.binary_classifier:app"
                ),
            ),
            # The earlier shape; its `serve` block is no provider's.
            "a serve block": changed(
                spatial_document(), lambda d: d.update(serve={"name": "x"})
            ),
            "a block no provider takes": changed(
                spatial_document(), lambda d: d.update(listing={"name": "Prostate"})
            ),
        }
        for name, document in cases.items():
            with (
                self.subTest(name),
                self.assertLogs("model_service.api.model_catalog", "ERROR"),
            ):
                self.assertEqual(len(ModelCatalog.from_documents([document])), 0)

    def test_a_repeated_model_id_keeps_the_first_file(self) -> None:
        first = spatial_document()
        repeated = deepcopy(first)
        user_config = repeated["deployment"]["user_config"]
        user_config["mpp"] = 1.0

        with self.assertLogs("model_service.api.model_catalog", "ERROR"):
            catalog = ModelCatalog.from_documents([first, repeated])

        self.assertEqual(catalog.get("prostate-classifier-1").mpp, 0.468)
        self.assertEqual(len(catalog), 1)

    def test_a_catalog_that_is_not_a_list_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            ModelCatalog.from_documents({"models": []})


MODEL_FILES = sorted(
    Path(__file__).parents[2].joinpath("deploy/helm/rayservice/models").rglob("*.yaml")
)


class ModelImplementationTests(unittest.TestCase):
    def test_every_model_file_points_to_an_implementation_of_its_contract(
        self,
    ) -> None:
        from model_service.builders.model_clients.registry import CLIENTS

        self.assertTrue(MODEL_FILES)
        for path in MODEL_FILES:
            with self.subTest(path.name):
                document = model_file(path)
                contract = InferenceContract(document["contract"])
                _, deployment = implementation(document)
                # The methods the contract's model client calls by handle.
                for method in CLIENTS[contract].methods:
                    self.assertTrue(
                        callable(getattr(deployment.func_or_class, method, None)),
                        f"{document['implementation']} lacks {method} for {contract}",
                    )


if __name__ == "__main__":
    unittest.main()
