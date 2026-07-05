"""CLI tests."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from mnemosyne_brain.app.cli import main


class CliTestCase(unittest.TestCase):
    """Verifies the CLI sends one message through the graph."""

    def test_cli_prints_response_track_and_no_capsule_for_local_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "mnemosyne_cli.sqlite3")
            previous_db_path = os.environ.get("MNEMOSYNE_DB_PATH")
            os.environ["MNEMOSYNE_DB_PATH"] = db_path
            output = io.StringIO()
            try:
                with redirect_stdout(output):
                    exit_code = main(["Remember that Pav loves architecture diagrams"])
            finally:
                if previous_db_path is None:
                    os.environ.pop("MNEMOSYNE_DB_PATH", None)
                else:
                    os.environ["MNEMOSYNE_DB_PATH"] = previous_db_path

        rendered = output.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("Assistant: Local answer: Remember that Pav loves architecture diagrams", rendered)
        self.assertIn("Track: trk_", rendered)
        self.assertIn("Capsule: none", rendered)
