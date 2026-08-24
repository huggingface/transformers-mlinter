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


class TRF046Test(RuleTestCase):
    # --- TRF046: forward must not write module attributes ---

    def test_trf046_flags_self_assignment_in_forward(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        self.sequence_length = hidden_states.shape[1]
        return hidden_states
"""
        trf046 = self._run(mlinter.TRF046, source)
        self.assertEqual(len(trf046), 1)
        self.assertIn("FooModel.forward writes self.sequence_length", trf046[0].message)

    def test_trf046_flags_augmented_assignment(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        self.call_count += 1
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF046})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF046]), 1)

    def test_trf046_allows_locals_and_init_assignments(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF046})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF046], [])

    def test_trf046_respects_suppression_comment(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        # trf-ignore: TRF046
        self.cached_states = hidden_states
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF046})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF046], [])
