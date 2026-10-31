"""URL allowlist and workspace cleanup."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from model_service.builders.contracts import BuilderResult, SharedArtifact
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.input_access import job_workspace
from model_service.providers.empaia.provider import EmpaiaProviderAdapter
from model_service.scheduler.contracts import (
    ExecutionError,
    InvalidScheduleRequestError,
    ProviderName,
    ScheduleRequest,
)
from model_service.scheduler.providers import (
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
)


def unused_gateway_factory(_context: object) -> object:
    raise AssertionError("the gateway must not be created")


class ProviderSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_unlisted_appservice_never_receives_the_credential(self) -> None:
        provider = EmpaiaProviderAdapter(
            unused_gateway_factory,  # type: ignore[arg-type]
            allowed_api_urls=("https://empaia.example/app-api",),
        )
        for api_url in (
            "https://attacker.example",
            "https://empaia.example/app-api-evil",
            "https://empaia.example",
        ):
            with (
                self.subTest(api_url=api_url),
                self.assertRaises(InvalidScheduleRequestError),
            ):
                await provider.prepare(
                    ScheduleRequest(
                        builder_id="heatmap",
                        model_id="prostate-classifier-1",
                        provider="empaia",
                        body={
                            "installation_id": "installation-1",
                            "api_url": api_url,
                            "job_id": "job-1",
                            "credential": "token",
                        },
                    ),
                    datetime.now(UTC) + timedelta(hours=1),
                )

    async def test_an_empty_allowlist_refuses_every_job(self) -> None:
        provider = EmpaiaProviderAdapter(unused_gateway_factory, allowed_api_urls=())  # type: ignore[arg-type]

        with self.assertRaises(InvalidScheduleRequestError):
            await provider.prepare(
                ScheduleRequest(
                    builder_id="heatmap",
                    model_id="prostate-classifier-1",
                    provider=ProviderName.EMPAIA,
                    body={
                        "installation_id": "https://empaia.example/app-api",
                        "api_url": "https://empaia.example/app-api",
                        "job_id": "job-1",
                        "credential": "token",
                    },
                ),
                datetime.now(UTC) + timedelta(hours=1),
            )

    def job_workspaces(self, root: Path) -> tuple[Path, Path, Path]:
        """Two attempts of job "j" (one with a heatmap) and another job's workspace."""
        job = job_workspace(root, "https://empaia.invalid", "j")
        artifact = job / "attempt-2" / "outputs" / "heatmap.npz"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"npz")
        (job / "attempt-1" / "outputs").mkdir(parents=True)
        other_job = job_workspace(root, "https://empaia.invalid", "other")
        other_job.mkdir()
        return job, artifact, other_job

    async def test_every_attempt_workspace_is_removed_after_publication(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, artifact, other_job = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            await provider.finish(
                {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"},
                ProviderSuccess(
                    BuilderResult(
                        outputs={
                            "heatmap": SharedArtifact(
                                path=str(artifact), media_type="application/x-npz"
                            )
                        }
                    )
                ),
            )

            gateway.finalize.assert_awaited_once()
            self.assertFalse(job.exists())
            self.assertTrue(other_job.exists())

    async def test_failed_jobs_free_their_workspace_even_if_reporting_fails(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.report_failure.side_effect = EmpaiaGatewayError(
            "EMPAIA AppService is temporarily unavailable", retryable=True
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, _artifact, other_job = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            with self.assertRaises(ProviderFinishFailureError):
                await provider.finish(
                    {
                        "api_url": "https://empaia.invalid",
                        "job_id": "j",
                        "credential": "c",
                    },
                    ProviderFailure(ExecutionError("builder_failed", "it failed")),
                )

            self.assertFalse(job.exists())
            self.assertTrue(other_job.exists())

    async def test_a_publication_that_fails_keeps_outputs_for_its_retry(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.finalize.side_effect = EmpaiaGatewayError(
            "EMPAIA AppService is temporarily unavailable", retryable=True
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _job, artifact, _other = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            with self.assertRaises(ProviderFinishFailureError):
                await provider.finish(
                    {
                        "api_url": "https://empaia.invalid",
                        "job_id": "j",
                        "credential": "c",
                    },
                    ProviderSuccess(BuilderResult(outputs={})),
                )

            self.assertTrue(artifact.exists())
