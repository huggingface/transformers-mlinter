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

"""Tests for `.ai/context-script`, the repo context injected into PR reviews."""

import importlib.util
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".ai" / "context-script"


def _load_context_script():
    """Import the hook by path: it is an executable without a `.py` suffix."""
    loader = SourceFileLoader("context_script", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


context_script = _load_context_script()


def added(path):
    return {"path": path, "status": "added"}


def modified(path):
    return {"path": path, "status": "modified"}


class NumberingStateTest(unittest.TestCase):
    """The reviewer must never be told that the rule this PR adds already exists.

    ai-reviewer runs the hook against the PR's own checkout, so a plain listing of
    `mlinter/` counts the new rule as pre-existing and reports the next free id one
    too high — which is how three reviews in a row asked for a renumbering that
    would have opened a permanent gap in the sequence.
    """

    def test_baseline_comes_from_the_base_branch_when_git_can_answer(self):
        files = [added("mlinter/trf055.py")]
        with patch.object(context_script, "_git_rule_numbers", return_value=list(range(1, 55))):
            baseline, source = context_script.baseline_rule_numbers(files)
        self.assertEqual(baseline[-1], 54)
        self.assertEqual(source, "the base branch")

    def test_baseline_falls_back_to_the_checkout_minus_this_prs_modules(self):
        # The helper sandbox binds only the checkout, so git cannot always reach
        # a linked worktree's object store. The local listing must still exclude
        # the PR's own new module.
        files = [added("mlinter/trf055.py")]
        with (
            patch.object(context_script, "_git_rule_numbers", return_value=None),
            patch.object(context_script, "existing_rule_numbers", return_value=list(range(1, 56))),
        ):
            baseline, source = context_script.baseline_rule_numbers(files)
        self.assertEqual(baseline[-1], 54, "the PR's own trf055 must not count as pre-existing")
        self.assertIn("minus the modules this PR adds", source)

    def test_next_free_id_is_the_one_this_pr_uses(self):
        files = [added("mlinter/trf055.py")]
        with (
            patch.object(context_script, "_git_rule_numbers", return_value=None),
            patch.object(context_script, "existing_rule_numbers", return_value=list(range(1, 56))),
        ):
            section = context_script.numbering_section(files)
        self.assertIn("Next free rule ID: **TRF055**", section)
        self.assertIn("Rule IDs this PR adds: TRF055", section)
        self.assertIn("do not collide", section)
        self.assertNotIn("ID collision", section)

    def test_modifying_a_rule_module_does_not_claim_its_id(self):
        # Touching trf019.py does not add id 19, so it must not be subtracted from
        # the baseline either.
        files = [modified("mlinter/trf019.py")]
        self.assertEqual(context_script.added_rule_numbers(files), set())
        with (
            patch.object(context_script, "_git_rule_numbers", return_value=None),
            patch.object(context_script, "existing_rule_numbers", return_value=list(range(1, 55))),
        ):
            baseline, _ = context_script.baseline_rule_numbers(files)
        self.assertIn(19, baseline)

    def test_real_collision_is_still_reported(self):
        files = [added("mlinter/trf042.py")]
        with patch.object(context_script, "_git_rule_numbers", return_value=list(range(1, 55))):
            section = context_script.numbering_section(files)
        self.assertIn("ID collision", section)
        self.assertIn("TRF042", section)

    def test_status_absent_from_the_payload_still_yields_added_ids(self):
        # Older ai-reviewer payloads carry only `path`; fall back to treating a
        # touched rule module as added rather than reporting nothing.
        self.assertEqual(
            context_script.added_rule_numbers([{"path": "mlinter/trf055.py"}]),
            {55},
        )

    def test_non_rule_paths_and_malformed_entries_are_ignored(self):
        files = [added("mlinter/rules.toml"), added("tests/test_mlinter.py"), "not-a-dict", {}]
        self.assertEqual(context_script.added_rule_numbers(files), set())


if __name__ == "__main__":
    unittest.main()
