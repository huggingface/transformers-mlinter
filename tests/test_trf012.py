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


from tests.rule_test_utils import RuleTestCase, mlinter


class TRF012Test(RuleTestCase):
    # --- TRF012: _init_weights should use init primitives ---

    def test_trf012_flags_inplace_module_weight_ops(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        module.weight.normal_(mean=0.0, std=0.02)
"""
        trf012 = self._run(mlinter.TRF012, source)
        self.assertEqual(len(trf012), 1)
        self.assertIn("in-place operation on a module's weight", trf012[0].message)

    def test_trf012_allows_init_primitives(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        init.normal_(module.weight, mean=0.0, std=0.02)
"""
        trf012 = self._run(mlinter.TRF012, source)
        self.assertEqual(trf012, [])
