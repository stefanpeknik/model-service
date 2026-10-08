"""Opt-in regression checks run inside the built scheduler container."""

from __future__ import annotations

import getpass
import os
import pwd
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("RUN_SCHEDULER_IMAGE_CHECKS") == "1",
    "RUN_SCHEDULER_IMAGE_CHECKS=1 requires the built scheduler image",
)
class SchedulerImageTests(unittest.TestCase):
    def test_migration_user_exists_without_environment_or_runtime_injection(self):
        self.assertEqual(os.getuid(), 10001)
        self.assertEqual(os.getgid(), 10001)
        self.assertEqual(pwd.getpwuid(os.getuid()).pw_name, "scheduler")
        with patch.dict(os.environ):
            for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
                os.environ.pop(name, None)
            self.assertEqual(getpass.getuser(), "scheduler")
