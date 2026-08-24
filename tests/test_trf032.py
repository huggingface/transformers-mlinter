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


class TRF032Test(RuleTestCase):
    # --- TRF032: masked fill must use torch.finfo(dtype).min ---

    def test_trf032_flags_magic_negative(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, scores, mask):
        return scores.masked_fill(~mask, -1e9)
"""
        violations = self._run(mlinter.TRF032, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("finfo", violations[0].message)

    def test_trf032_accepts_finfo_and_small_values(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, scores, mask):
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        pad = torch.full_like(scores, -1.0)
        return scores + pad
"""
        self.assertEqual(self._run(mlinter.TRF032, source), [])

    def test_trf032_reports_once_per_call(self):
        source = """
def f(scores, mask):
    return scores.masked_fill(~mask, -1e9).masked_fill(~mask, -1e4)
"""
        self.assertEqual(len(self._run(mlinter.TRF032, source)), 2)
