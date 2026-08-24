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


class TRF030Test(RuleTestCase):
    # --- TRF030: config attribute chain depth ---

    def test_trf030_flags_three_level_config_chain(self):
        source = """
class FooAtomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = FooLayerNorm(config.diffusion_config.atom_encoder_config.hidden_size)
"""
        violations = self._run(mlinter.TRF030, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("3 levels", violations[0].message)

    def test_trf030_accepts_one_and_two_hops(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.a = config.hidden_size
        self.b = config.text_config.hidden_size
        self.c = self.config.vision_config.num_attention_heads
"""
        self.assertEqual(self._run(mlinter.TRF030, source), [])

    def test_trf030_reports_once_per_line(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.a = config.x.y.hidden_size + config.x.y.intermediate_size
"""
        self.assertEqual(len(self._run(mlinter.TRF030, source)), 1)
