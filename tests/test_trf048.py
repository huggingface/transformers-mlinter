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


from tests.rule_test_utils import Path, RuleTestCase, mlinter


class TRF048Test(RuleTestCase):
    # --- TRF048: _tied_weights_keys must be a dict ---

    def test_trf048_flags_list_form(self):
        source = """
class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
"""
        trf048 = self._run(mlinter.TRF048, source)
        self.assertEqual(len(trf048), 1)
        self.assertIn("dict", trf048[0].message)

    def test_trf048_allows_dict_form(self):
        source = """
class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF048})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF048], [])
