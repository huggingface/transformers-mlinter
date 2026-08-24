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


from tests.rule_test_utils import RuleTestCase, _helpers_mod, mlinter, patch


class TRF027Test(RuleTestCase):
    # --- TRF027: no bare assert in model files ---

    def _trf027(self, source, file_name="modeling_foo.py"):
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            return self._run(mlinter.TRF027, source, file_name=file_name)

    def test_trf027_flags_assert(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        assert hidden_states.dim() == 3
        return hidden_states
"""
        violations = self._trf027(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("assert", violations[0].message)

    def test_trf027_accepts_raise_and_skips_other_files(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        if hidden_states.dim() != 3:
            raise ValueError("expected 3D")
        return hidden_states
"""
        self.assertEqual(self._trf027(source), [])
        assert_source = "def f(x):\n    assert x\n"
        self.assertEqual(self._trf027(assert_source, file_name="processing_foo.py"), [])

    def test_trf027_respects_suppression(self):
        source = """
def f(x):
    # trf-ignore: TRF027
    assert x
"""
        self.assertEqual(self._trf027(source), [])
