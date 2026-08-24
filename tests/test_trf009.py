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


from tests.rule_test_utils import Path, RuleTestCase, mlinter, patch


class TRF009Test(RuleTestCase):
    # --- TRF009: cross-model imports (old TRF013) ---

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_flags_cross_model_import_in_modeling_file(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `llama`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_allows_same_model_import_in_modeling_file(self, _mock):
        source = """
from .configuration_foo import FooConfig
from transformers.models.foo.configuration_foo import FooConfig as FooConfigAlias
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(trf009, [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_ignores_modular_files(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(trf009, [])
