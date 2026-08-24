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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf019_mod, date, mlinter, patch


class TRF019Test(RuleTestCase):
    # --- TRF019: ModelNameProcessorKwargs must not define _defaults ---

    def test_trf019_flags_non_empty_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {"return_tensors": "pt"},
    }
    text_kwargs: FooTokenizerKwargs
    images_kwargs: FooImageProcessorKwargs
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)
        self.assertIn("_defaults", trf019[0].message)
        self.assertIn("processor_config.json", trf019[0].message)

    def test_trf019_no_violation_without_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: FooTokenizerKwargs
    images_kwargs: FooImageProcessorKwargs
"""
        trf019 = self._run(mlinter.TRF019, source, file_name="processing_foo.py")
        self.assertEqual(len(trf019), 0)

    def test_trf019_no_violation_with_empty_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}
    text_kwargs: FooTokenizerKwargs
"""
        trf019 = self._run(mlinter.TRF019, source, file_name="processing_foo.py")
        self.assertEqual(len(trf019), 0)

    def test_trf019_ignores_non_processing_files(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        for file_name in ("image_processing_foo.py", "modeling_foo.py", "configuration_foo.py"):
            file_path = Path(f"src/transformers/models/foo/{file_name}")
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
            trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
            self.assertEqual(len(trf019), 0, f"Expected no violation in {file_name}")

    def test_trf019_ignores_non_processor_kwargs_classes(self):
        source = """
class FooConfig:
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        trf019 = self._run(mlinter.TRF019, source, file_name="processing_foo.py")
        self.assertEqual(len(trf019), 0)

    def test_trf019_allowlisted_model_skipped(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.dict(mlinter.TRF_MODEL_DIR_ALLOWLISTS, {mlinter.TRF019: {"foo"}}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_flags_multiple_kwargs_classes(self):
        source = """
class FooTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"truncation": True}}

class FooVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"images_kwargs": {"do_resize": True}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 2)

    def test_trf019_cutoff_exempts_file_committed_before_cutoff(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-10"),
            patch.object(_helpers_mod, "model_contribution_date", return_value=date(2025, 1, 1)),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_cutoff_flags_file_committed_on_or_after_cutoff(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-09"),
            patch.object(_helpers_mod, "model_contribution_date", return_value=date(2026, 6, 10)),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)

    def test_trf019_cutoff_flags_file_not_in_git(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-10"),
            patch.object(_helpers_mod, "model_contribution_date", return_value=None),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)

    def test_trf019_no_cutoff_always_flags(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_trf019_mod, "CUTOFF_DATE", ""):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)
