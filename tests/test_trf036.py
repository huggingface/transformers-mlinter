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


class TRF036Test(RuleTestCase):
    # --- TRF036: no nn.Sequential in modeling ---

    def test_trf036_flags_sequential(self):
        source = """
class FooMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU())
"""
        violations = self._run(mlinter.TRF036, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("nn.Sequential", violations[0].message)

    def test_trf036_accepts_explicit_submodules(self):
        source = """
class FooMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
"""
        self.assertEqual(self._run(mlinter.TRF036, source), [])
