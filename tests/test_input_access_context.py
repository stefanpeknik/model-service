from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi import FastAPI
from pydantic import TypeAdapter, ValidationError

from execution_scheduler.contracts import InvalidScheduleRequestError, ScheduleRequest
from integrations.empaia.gateway import EmpaiaJobContext
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from integrations.empaia.provider import EmpaiaProviderAdapter
from integrations.shared_storage import SharedStorageAccessAdapter
from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import (
    WorkflowCatalog,
    WorkflowDefinition,
    WorkflowValueDefinition,
)
from workflow_execution.contracts import (
    InlineOutput,
    InputAccessContext,
    InvalidWorkflowInvocation,
    WorkflowExecutionContext,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.dispatch import (
    HTTPWorkflowDispatch,
    RemoteWorkflowFailureError,
)
from workflow_execution.http import WorkflowRunRequest, run_workflow_http
from workflow_execution.serialization import invocation_to_dict


def invocation(kind: str, state: dict) -> WorkflowInvocation:
    return WorkflowInvocation(
        inputs={},
        resolved_configuration={},
        input_access=InputAccessContext(kind=kind, state=state),
        deadline_at=datetime.now(UTC) + timedelta(minutes=1),
    )


class InputAccessContextTests(unittest.IsolatedAsyncioTestCase):
    def test_shared_storage_rejects_missing_coerced_and_unexpected_fields(self) -> None:
        adapter = SharedStorageAccessAdapter()
        for state in (
            {},
            {"root": None},
            {"root": True},
            {"root": 1},
            {"root": ""},
            {"root": "/shared", "credential": "secret"},
            {"root": "/shared", "processing_attempt_id": "caller-attempt"},
        ):
            with (
                self.subTest(state=state),
                self.assertRaises(InvalidWorkflowInvocation),
            ):
                adapter.create(
                    invocation("shared_storage", state), WorkflowExecutionContext()
                )

    def test_runtime_context_rejects_invalid_attempts_and_non_json_secrets(
        self,
    ) -> None:
        for attempt in ("", True, 1):
            with self.subTest(attempt=attempt), self.assertRaises(ValidationError):
                WorkflowExecutionContext(processing_attempt_id=attempt)
        with self.assertRaises(ValidationError):
            WorkflowExecutionContext(protected_context={"invalid": float("nan")})
        with self.assertRaises(ValidationError):
            TypeAdapter(WorkflowExecutionContext).validate_python({"unexpected": True})

    async def test_empaia_validates_state_and_runtime_before_creating_gateway(
        self,
    ) -> None:
        def unexpected_gateway(_context):
            self.fail("invalid context must not open an AppService gateway")

        public = {"api_url": "https://empaia.invalid", "job_id": "job"}
        protected = {**public, "credential": "secret"}
        runtime = WorkflowExecutionContext(
            processing_attempt_id="attempt", protected_context=protected
        )
        cases = [
            ({}, runtime),
            ({**public, "job_id": False}, runtime),
            ({**public, "credential": "secret"}, runtime),
            ({**public, "processing_attempt_id": "attempt"}, runtime),
            ({**public, "job_id": "another-job"}, runtime),
            ({**public, "api_url": "https://another.invalid"}, runtime),
            (public, WorkflowExecutionContext(processing_attempt_id="attempt")),
            (public, WorkflowExecutionContext(protected_context=protected)),
            (
                public,
                WorkflowExecutionContext(
                    processing_attempt_id="attempt",
                    protected_context={**protected, "credential": 1},
                ),
            ),
        ]
        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessAdapter(unexpected_gateway, Path(directory))
            for state, context in cases:
                with (
                    self.subTest(state=state),
                    self.assertRaises(InvalidWorkflowInvocation) as raised,
                ):
                    await adapter.create(invocation("empaia", state), context)
                self.assertNotIn("secret", str(raised.exception))

    def test_empaia_admission_uses_strict_payload_schema_without_echoing_credentials(
        self,
    ) -> None:
        def unexpected_gateway(_context):
            self.fail("identifying a schedule must not contact AppService")

        provider = EmpaiaProviderAdapter(WorkflowCatalog.default(), unexpected_gateway)
        valid = {
            "installation_id": "installation",
            "api_url": "https://empaia.invalid",
            "job_id": "job",
            "credential": "secret",
        }
        for body in (
            {**valid, "job_id": 1},
            {**valid, "credential": False},
            {**valid, "unexpected": "secret"},
        ):
            with (
                self.subTest(body=body),
                self.assertRaises(InvalidScheduleRequestError) as raised,
            ):
                provider.identify(
                    ScheduleRequest(
                        workflow_id="prostate-heatmap", provider="empaia", body=body
                    )
                )
            self.assertNotIn("secret", str(raised.exception))

    def test_context_representations_do_not_expose_credentials(self) -> None:
        protected = {
            "api_url": "https://empaia.invalid",
            "job_id": "job",
            "credential": "secret-token",
        }
        self.assertNotIn(
            "secret-token", repr(EmpaiaJobContext.model_validate(protected))
        )
        self.assertNotIn(
            "secret-token", repr(WorkflowExecutionContext(protected_context=protected))
        )

    async def test_http_dispatch_round_trip_keeps_runtime_out_of_durable_invocation(
        self,
    ) -> None:
        seen_credentials = []
        closed = []

        class Gateway:
            async def close(self):
                closed.append(True)

        def gateway_factory(context):
            seen_credentials.append(context.credential)
            return Gateway()

        class Workflow:
            async def run(self, _context):
                return WorkflowResult({"score": InlineOutput(1)})

        with TemporaryDirectory() as directory:
            application = WorkflowApplication(
                WorkflowDefinition(
                    "probe", {}, {"score": WorkflowValueDefinition("inline")}
                ),
                Workflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(gateway_factory, Path(directory))
                },
            )
            api = FastAPI()
            seen_requests = []

            @api.post("/workflows/probe/", response_model=WorkflowResult)
            async def run(body: WorkflowRunRequest) -> WorkflowResult:
                seen_requests.append(body)
                return await run_workflow_http(
                    application, body, body.execution_context
                )

            public = {"api_url": "https://empaia.invalid", "job_id": "job"}
            request = invocation("empaia", public)
            before = invocation_to_dict(request)
            runtime = WorkflowExecutionContext(
                processing_attempt_id="attempt",
                protected_context={**public, "credential": "secret"},
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api)
            ) as client:
                dispatch = HTTPWorkflowDispatch(client, origin="http://workflow")
                result = await dispatch.run("probe", runtime, request)
                with self.assertRaises(RemoteWorkflowFailureError) as raised:
                    await dispatch.run(
                        "probe",
                        WorkflowExecutionContext(processing_attempt_id="attempt"),
                        request,
                    )

            self.assertEqual(result, WorkflowResult({"score": InlineOutput(1)}))
            self.assertEqual(seen_requests[0].execution_context, runtime)
            self.assertEqual(seen_requests[0].input_access.state, public)
            self.assertEqual(invocation_to_dict(seen_requests[0]), before)
            self.assertEqual(invocation_to_dict(request), before)
            self.assertEqual(seen_credentials, ["secret"])
            self.assertEqual(closed, [True])
            self.assertEqual(raised.exception.code, "invalid_workflow_invocation")
            self.assertFalse(raised.exception.retryable)
            self.assertNotIn("secret", str(raised.exception))

    async def test_direct_workflow_http_request_without_runtime_remains_valid(
        self,
    ) -> None:
        class Workflow:
            async def run(self, context):
                return WorkflowResult(
                    {"path": InlineOutput(str(context.output_path("result.bin")))}
                )

        with TemporaryDirectory() as directory:
            application = WorkflowApplication(
                WorkflowDefinition(
                    "probe", {}, {"path": WorkflowValueDefinition("inline")}
                ),
                Workflow(),
            )
            api = FastAPI()

            @api.post("/", response_model=WorkflowResult)
            async def run(body: WorkflowRunRequest) -> WorkflowResult:
                return await run_workflow_http(
                    application, body, body.execution_context
                )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://workflow"
            ) as client:
                response = await client.post(
                    "/",
                    json=invocation_to_dict(
                        invocation("shared_storage", {"root": directory})
                    ),
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["outputs"]["path"]["value"],
                str(Path(directory) / "result.bin"),
            )
