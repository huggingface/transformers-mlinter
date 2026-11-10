# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from mlinter import DEFAULT_ENABLED_TRF_RULES, TRF_RULES, Violation


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "scan_transformers.py"
_spec = importlib.util.spec_from_file_location("scan_transformers", SCRIPT)
assert _spec is not None and _spec.loader is not None
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

DEFAULT_RULE = sorted(DEFAULT_ENABLED_TRF_RULES)[0]
DISABLED_RULE = sorted(set(TRF_RULES) - DEFAULT_ENABLED_TRF_RULES)[0]
MODELING = Path("src/transformers/models/foo/modeling_foo.py")


def _violation(rule_id: str, line: int, message: str = "something is off") -> Violation:
    return Violation(file_path=MODELING, line_number=line, rule_id=rule_id, message=f"{rule_id}: {message}")


class ScanTransformersTest(unittest.TestCase):
    def _main(self, violations: list[Violation], *argv: str) -> tuple[int, str]:
        stdout = StringIO()
        with (
            patch.object(scan, "_collect", return_value=([MODELING], violations)),
            patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": ""}),
            redirect_stdout(stdout),
            redirect_stderr(StringIO()),
        ):
            code = scan.main(list(argv))
        return code, stdout.getvalue()

    def test_defaults_only_keeps_default_enabled_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "base.json"
            code, stdout = self._main(
                [_violation(DEFAULT_RULE, 3), _violation(DISABLED_RULE, 5)], "--defaults-only", str(out)
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout, "")
            records = json.loads(out.read_text())
            self.assertEqual([(r["rule_id"], r["line"]) for r in records], [(DEFAULT_RULE, 3)])
            # Nothing but the JSON: the base run must not overwrite the report.
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["base.json"])

    def test_without_baseline_the_scan_only_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, stdout = self._main([_violation(DEFAULT_RULE, 3)], "--output-dir", tmp)
        self.assertEqual(code, 0)
        self.assertNotIn("## Gate", stdout)

    def test_baseline_fails_on_a_new_default_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "base.json"
            baseline.write_text(json.dumps(scan._default_findings([_violation(DEFAULT_RULE, 3)])))
            code, stdout = self._main(
                [_violation(DEFAULT_RULE, 3), _violation(DEFAULT_RULE, 9, "new finding")],
                "--output-dir",
                tmp,
                "--baseline",
                str(baseline),
            )
            report = (Path(tmp) / "scan-report.md").read_text()
        self.assertEqual(code, 1)
        self.assertIn("**1** on the base branch, **2** with this change, **1** new", stdout)
        self.assertIn(f"`{MODELING}:9` — {DEFAULT_RULE}: new finding", stdout)
        self.assertNotIn(f"`{MODELING}:3` — {DEFAULT_RULE}: something", stdout.split("# transformers-mlinter scan")[0])
        self.assertTrue(report.startswith("## Gate"))

    def test_baseline_passes_when_nothing_is_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "base.json"
            baseline.write_text(json.dumps(scan._default_findings([_violation(DEFAULT_RULE, 3)])))
            # A reworded message, a disabled rule's finding and a fixed violation are all fine.
            code, stdout = self._main(
                [_violation(DEFAULT_RULE, 3, "reworded"), _violation(DISABLED_RULE, 7)],
                "--output-dir",
                tmp,
                "--baseline",
                str(baseline),
            )
        self.assertEqual(code, 0)
        self.assertIn("No new violations", stdout)

    def test_a_rule_newly_enabled_by_default_counts_as_new(self):
        # The base did not run the rule by default, so it has no record of its findings.
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "base.json"
            baseline.write_text("[]\n")
            code, _ = self._main([_violation(DEFAULT_RULE, 3)], "--output-dir", tmp, "--baseline", str(baseline))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
