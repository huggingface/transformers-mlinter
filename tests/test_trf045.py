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


class TRF045Test(RuleTestCase):
    # --- TRF045: forward must not declare legacy output_*/return_dict parameters ---

    def test_trf045_flags_legacy_output_parameters(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids, output_attentions=None, output_hidden_states=None, return_dict=None):
        return input_ids
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF045})
        trf045 = [v for v in violations if v.rule_id == mlinter.TRF045]
        self.assertEqual(len(trf045), 1)
        self.assertIn("output_attentions, output_hidden_states, return_dict", trf045[0].message)

    def test_trf045_no_violation_with_kwargs_signature(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids, **kwargs):
        return input_ids
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF045})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF045], [])

    def test_trf045_cutoff_exempts_old_model(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids, return_dict=None):
        return input_ids
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2020, 1, 1)):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF045})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF045], [])

    def test_trf045_ignores_non_forward_methods(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_encoder_outputs(self, input_ids, return_dict=None):
        return input_ids
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF045})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF045], [])
