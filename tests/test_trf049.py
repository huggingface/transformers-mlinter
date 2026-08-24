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


class TRF049Test(RuleTestCase):
    # --- TRF049: weight initialization belongs in _init_weights, not __init__ ---

    def test_trf049_flags_nn_init_in_init(self):
        source = """
class FooEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(config.num_positions, config.hidden_size))
        nn.init.trunc_normal_(self.position_embedding, std=config.initializer_range)
"""
        trf049 = self._run(mlinter.TRF049, source)
        self.assertEqual(len(trf049), 1)
        self.assertIn("FooEmbeddings.__init__ initializes weight values", trf049[0].message)

    def test_trf049_flags_inplace_init_on_own_parameter(self):
        source = """
class FooLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.02)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF049})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF049]), 1)

    def test_trf049_allows_init_weights_method(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FooEmbeddings):
            init.trunc_normal_(module.position_embedding, std=self.config.initializer_range)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF049})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF049], [])

    def test_trf049_allows_plain_allocation(self):
        source = """
class FooEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(config.num_positions, config.hidden_size))
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF049})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF049], [])
