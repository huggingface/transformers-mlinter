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


class TRF056Test(RuleTestCase):
    # --- TRF056: no .item()/.tolist() inside forward ---

    def test_trf056_flags_tolist_in_forward(self):
        source = """
class FooVisionModel(FooPreTrainedModel):
    def forward(self, hidden_states, grid_thw):
        for grid, item in zip(grid_thw.tolist(), hidden_states):
            self.merger(item, size=grid)
"""
        violations = self._run(mlinter.TRF056, source)
        self.assertEqual(len(violations), 1)
        self.assertIn(".tolist()", violations[0].message)

    def test_trf056_flags_item_in_forward(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids, num_tokens):
        for _ in range(num_tokens.sum().item()):
            input_ids = self.layer(input_ids)
"""
        violations = self._run(mlinter.TRF056, source)
        self.assertEqual(len(violations), 1)
        self.assertIn(".item()", violations[0].message)

    def test_trf056_exempts_split_sizes(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, query_states, cu_seqlens):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        return torch.split(query_states, lengths.tolist(), dim=2)
"""
        self.assertEqual(self._run(mlinter.TRF056, source), [])

    def test_trf056_exempts_split_sizes_through_a_local(self):
        source = """
class FooVisionModel(FooPreTrainedModel):
    def forward(self, hidden_states, grid_thw):
        split_sizes = grid_thw.prod(dim=-1).tolist()
        return torch.split(hidden_states, split_sizes, dim=1)
"""
        self.assertEqual(self._run(mlinter.TRF056, source), [])

    def test_trf056_ignores_calls_outside_forward(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.sizes = config.sizes.tolist()

    def post_process(self, logits):
        return logits.argmax(-1).tolist()
"""
        self.assertEqual(self._run(mlinter.TRF056, source), [])

    def test_trf056_respects_suppression(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, counts):
        # trf-ignore: TRF056
        sizes = counts.tolist()
        return sizes
"""
        self.assertEqual(self._run(mlinter.TRF056, source), [])
