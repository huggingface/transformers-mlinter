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


class TRF037Test(RuleTestCase):
    # --- TRF037: no torch.einsum in modeling (opt-in) ---

    def test_trf037_flags_einsum_with_equation(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, q, k):
        return torch.einsum("bqhc,bkhc->bhqk", q, k)
"""
        violations = self._run(mlinter.TRF037, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("bqhc,bkhc->bhqk", violations[0].message)

    def test_trf037_is_disabled_by_default(self):
        self.assertNotIn(mlinter.TRF037, mlinter.DEFAULT_ENABLED_TRF_RULES)

    def test_trf037_accepts_explicit_matmul(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, q, k):
        return q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
"""
        self.assertEqual(self._run(mlinter.TRF037, source), [])
