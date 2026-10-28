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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, date, mlinter, patch


class TRF060Test(RuleTestCase):
    # --- TRF060: model files must import from transformers with relative imports ---

    def _trf060(self, source: str, file_path: str = "src/transformers/models/foo/modeling_foo.py", contributed=None):
        with patch.object(_helpers_mod, "model_contribution_date", return_value=contributed):
            violations = mlinter.analyze_file(Path(file_path), source, enabled_rules={mlinter.TRF060})
        return [v for v in violations if v.rule_id == mlinter.TRF060]

    def test_trf060_flags_absolute_import_of_shared_module(self):
        violations = self._trf060("from transformers.modeling_utils import PreTrainedModel\n")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 1)
        self.assertIn("`from ...modeling_utils import PreTrainedModel`", violations[0].message)

    def test_trf060_flags_top_level_package_import(self):
        violations = self._trf060("from transformers import initialization as init\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("`from ... import initialization as init`", violations[0].message)

    def test_trf060_suggests_sibling_module_for_own_model(self):
        violations = self._trf060("from transformers.models.foo.configuration_foo import FooConfig\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("`from .configuration_foo import FooConfig`", violations[0].message)

    def test_trf060_flags_modular_import_from_another_model(self):
        source = "from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP\n"
        violations = self._trf060(source, file_path="src/transformers/models/foo/modular_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`from ..llama.modeling_llama import LlamaAttention, LlamaMLP`", violations[0].message)

    def test_trf060_elides_long_import_lists(self):
        source = "from transformers.models.llama.modeling_llama import A, B, C, D\n"
        violations = self._trf060(source, file_path="src/transformers/models/foo/modular_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`from ..llama.modeling_llama import ...`", violations[0].message)

    def test_trf060_flags_plain_import_statement(self):
        violations = self._trf060("import transformers.utils.logging as hf_logging\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("`from ...utils import logging as hf_logging`", violations[0].message)

    def test_trf060_flags_import_inside_function(self):
        source = """
def foo():
    from transformers.utils import logging
"""
        violations = self._trf060(source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 3)

    def test_trf060_allows_relative_and_third_party_imports(self):
        source = """
import torch
from torch import nn
from transformers_extra import thing

from ...modeling_utils import PreTrainedModel
from ..llama.modeling_llama import LlamaAttention
from .configuration_foo import FooConfig
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_ignores_files_outside_a_transformers_checkout(self):
        # In a standalone model repo `transformers` is a dependency: absolute is the only valid form.
        source = "from transformers.modeling_utils import PreTrainedModel\n"
        self.assertEqual(self._trf060(source, file_path="my_model/modeling_foo.py"), [])

    def test_trf060_ignores_conversion_scripts(self):
        source = "from transformers import FooConfig, FooModel\n"
        path = "src/transformers/models/foo/convert_foo_to_hf.py"
        self.assertEqual(self._trf060(source, file_path=path), [])

    def test_trf060_cutoff_exempts_old_model(self):
        source = "from transformers.modeling_utils import PreTrainedModel\n"
        self.assertEqual(self._trf060(source, contributed=date(2020, 1, 1)), [])

    def test_trf060_suppression_comment(self):
        source = "from transformers.modeling_utils import PreTrainedModel  # trf-ignore: TRF060\n"
        self.assertEqual(self._trf060(source), [])
