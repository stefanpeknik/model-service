"""Check tagged workflow input/output values preserved across JSON conversion."""

import unittest
from datetime import UTC, datetime

from workflow_execution.contracts import WorkflowInputKind, WorkflowOutputKind
from workflow_execution.serialization import (
    invocation_from_dict,
    invocation_to_dict,
    result_from_dict,
    result_to_dict,
)


class WorkflowSerializationTests(unittest.TestCase):
    """Exercise tagged workflow input/output values preserved across JSON conversion."""

    def test_shared_kind_enums_preserve_invocation_wire_strings(self) -> None:
        """Verify shared kind enums preserve invocation wire strings.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        body = {
            "inputs": {
                "inline": {"kind": "inline", "value": 7},
                "slide": {"kind": "shared_path", "path": "/shared/slide.tiff"},
                "provider": {"kind": "provider", "name": "slide"},
            },
            "resolved_configuration": {},
            "input_access": {"kind": "test", "state": {}},
            "deadline_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        }
        invocation = invocation_from_dict(body)
        self.assertIs(invocation.inputs["inline"].kind, WorkflowInputKind.INLINE)
        self.assertIs(invocation.inputs["slide"].kind, WorkflowInputKind.SHARED_PATH)
        self.assertIs(invocation.inputs["provider"].kind, WorkflowInputKind.PROVIDER)
        self.assertEqual(invocation_to_dict(invocation), body)

    def test_shared_kind_enums_preserve_result_wire_strings(self) -> None:
        """Verify shared kind enums preserve result wire strings.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        body = {
            "outputs": {
                "score": {"kind": "inline", "value": 0.75},
                "heatmap": {
                    "kind": "shared_artifact",
                    "path": "/shared/result.tiff",
                    "media_type": "image/tiff",
                },
            }
        }
        result = result_from_dict(body)
        self.assertIs(result.outputs["score"].kind, WorkflowOutputKind.INLINE)
        self.assertIs(
            result.outputs["heatmap"].kind, WorkflowOutputKind.SHARED_ARTIFACT
        )
        self.assertEqual(result_to_dict(result), body)
