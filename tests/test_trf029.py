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


class TRF029Test(RuleTestCase):
    # --- TRF029: config plus a redundant config field ---

    def test_trf029_flags_redundant_config_arguments(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
"""
        violations = self._run(mlinter.TRF029, source)
        self.assertEqual(len(violations), 1)
        for name in ("embed_dim", "num_heads", "dropout"):
            self.assertIn(name, violations[0].message)

    def test_trf029_accepts_config_only_and_layer_idx(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, layer_idx=None, device=None, **kwargs):
        super().__init__()
        self.embed_dim = config.hidden_size
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])

    def test_trf029_ignores_modules_without_config(self):
        source = """
class FooRotary(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        self.head_dim = head_dim
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])
