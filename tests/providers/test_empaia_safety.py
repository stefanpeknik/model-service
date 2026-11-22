"""URL allowlist and workspace cleanup."""

from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from model_service.api.scheduler.public import (
    ExecutionError,
    InvalidScheduleRequestError,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
    ScheduleRequest,
)
from model_service.api.scheduler.targets import ExecutionTarget
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
from model_service.providers.empaia.jobs.inputs import job_workspace
from tests.support.empaia_fixtures import slide_info, write_score_grid


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
                    provider="empaia",
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
        write_score_grid(artifact)
        (job / "attempt-1" / "outputs").mkdir(parents=True)
        other_job = job_workspace(root, "https://empaia.invalid", "other")
        other_job.mkdir()
        return job, artifact, other_job

    async def test_every_attempt_workspace_is_removed_after_publication(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.slide_info.return_value = slide_info()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, artifact, other_job = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,
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
                    ),
                    ExecutionTarget.builder("heatmap", "prostate-classifier-1"),
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
                lambda _context: gateway,
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

    @unittest.skipIf(os.geteuid() == 0, "root removes read-only directories")
    async def test_a_workspace_that_cannot_be_removed_is_reported(self) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.slide_info.return_value = slide_info()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _job, artifact, _other = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )
            # As if another user had created it without letting others in.
            artifact.parent.chmod(0o555)
            try:
                with self.assertLogs("model_service", "WARNING"):
                    await provider.finish(
                        {
                            "api_url": "https://empaia.invalid",
                            "job_id": "j",
                            "credential": "c",
                        },
                        ProviderSuccess(
                            BuilderResult(
                                outputs={
                                    "heatmap": SharedArtifact(
                                        path=str(artifact),
                                        media_type="application/x-npz",
                                    )
                                }
                            ),
                            ExecutionTarget.builder("heatmap", "prostate-classifier-1"),
                        ),
                    )
            finally:
                artifact.parent.chmod(0o755)

            gateway.finalize.assert_awaited_once()
            self.assertTrue(artifact.exists())

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
                lambda _context: gateway,
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
                    ProviderSuccess(
                        BuilderResult(outputs={}),
                        ExecutionTarget.builder("heatmap", "prostate-classifier-1"),
                    ),
                )

            self.assertTrue(artifact.exists())
