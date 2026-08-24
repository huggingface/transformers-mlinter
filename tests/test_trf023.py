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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf023_mod, date, mlinter, patch


class TRF023Test(RuleTestCase):
    # --- TRF023: config fields must use canonical dimension names ---

    def _trf023(self, source, file_name="configuration_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF023})
        return [v for v in violations if v.rule_id == mlinter.TRF023]

    def test_trf023_flags_legacy_dataclass_fields(self):
        source = """
@strict(accept_kwargs=True)
class FooConfig(PreTrainedConfig):
    d_model: int = 1024
    d_ff: int = 4096
    n_heads: int = 16
    n_layers: int = 24
"""
        violations = self._trf023(source)
        self.assertEqual(len(violations), 4)
        messages = " ".join(v.message for v in violations)
        for legacy, canonical in (
            ("d_model", "hidden_size"),
            ("d_ff", "intermediate_size"),
            ("n_heads", "num_attention_heads"),
            ("n_layers", "num_hidden_layers"),
        ):
            self.assertIn(f"`{legacy}`", messages)
            self.assertIn(f"`{canonical}`", messages)

    def test_trf023_flags_legacy_init_assignment(self):
        source = """
class FooConfig(PreTrainedConfig):
    def __init__(self, n_embd=768, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = n_embd
"""
        violations = self._trf023(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("hidden_size", violations[0].message)

    def test_trf023_accepts_canonical_names(self):
        source = """
@strict(accept_kwargs=True)
class FooConfig(PreTrainedConfig):
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    head_dim: int = 64
    num_heads: int = 16
    num_layers: int = 24
    embed_dim: int = 512
"""
        self.assertEqual(self._trf023(source), [])

    def test_trf023_reports_each_legacy_field_once(self):
        source = """
class FooConfig(PreTrainedConfig):
    def __init__(self, d_model=1024, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
"""
        self.assertEqual(len(self._trf023(source)), 1)

    def test_trf023_ignores_non_config_classes_and_files(self):
        source = """
class FooAttention(nn.Module):
    d_model: int = 1024
"""
        self.assertEqual(self._trf023(source), [])
        config_source = "class FooConfig(PreTrainedConfig):\n    d_model: int = 1024\n"
        self.assertEqual(self._trf023(config_source, file_name="modeling_foo.py"), [])

    def test_trf023_respects_suppression(self):
        source = """
class FooConfig(PreTrainedConfig):
    # trf-ignore: TRF023
    d_model: int = 1024
"""
        self.assertEqual(self._trf023(source), [])

    def test_trf023_exempts_models_before_cutoff(self):
        source = "class FooConfig(PreTrainedConfig):\n    d_model: int = 1024\n"
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2023, 1, 1)):
            with patch.object(_trf023_mod, "CUTOFF_DATE", "2026-06-20"):
                violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF023})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF023], [])
