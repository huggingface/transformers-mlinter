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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, mlinter, patch


class TRF024Test(RuleTestCase):
    # --- TRF024: layer dimensions must come from the config ---

    def _trf024(self, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF024})
        return [v for v in violations if v.rule_id == mlinter.TRF024]

    def test_trf024_flags_hardcoded_dimensions(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(768, 3072, bias=False)
        self.norm = nn.LayerNorm(3072)
        self.embed = nn.Embedding(32000, config.hidden_size)
"""
        violations = self._trf024(source)
        self.assertEqual(len(violations), 3)
        self.assertIn("768", violations[0].message)
        self.assertIn("nn.Linear", violations[0].message)

    def test_trf024_flags_keyword_dimensions_and_sequence_shapes(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(in_features=config.hidden_size, out_features=4096)
        self.norm = nn.LayerNorm((1024,))
"""
        violations = self._trf024(source)
        self.assertEqual(len(violations), 2)

    def test_trf024_allows_config_values_and_small_literals(self):
        source = """
class FooHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.binary = nn.Linear(config.hidden_size, 2)
        self.patch = nn.Conv2d(3, config.hidden_size, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.group = nn.GroupNorm(32, config.hidden_size)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_ignores_operator_shape_arguments(self):
        source = """
class FooConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=128, padding=64)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_ignores_unrelated_linear_attribute(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = self.registry.Linear(768, 768)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_respects_suppression_and_file_type(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # trf-ignore: TRF024
        self.proj = nn.Linear(768, 3072)
"""
        self.assertEqual(self._trf024(source), [])
        plain = "class FooConfig(PreTrainedConfig):\n    proj = nn.Linear(768, 3072)\n"
        self.assertEqual(self._trf024(plain, file_name="configuration_foo.py"), [])
