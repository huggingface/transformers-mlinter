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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf059_mod, mlinter, patch, tempfile


class TRF059Test(RuleTestCase):
    # --- TRF059: moe_tp_experts forward signature ---

    def _run_trf059(self, source):
        with patch.object(_trf059_mod, "_MOE_TP_MODEL_DIRS", {"foo"}):
            return self._run(mlinter.TRF059, source)

    def test_trf059_discovers_models_from_tp_plan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir) / "src" / "transformers" / "models"
            model_dir = models_root / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "configuration_foo.py").write_text(
                'base_model_tp_plan = {"layers.*.experts": "moe_tp_experts"}\n', encoding="utf-8"
            )
            with (
                patch.object(_trf059_mod, "MODELS_ROOT", models_root),
                patch.object(_helpers_mod, "MODELS_ROOT", models_root),
            ):
                self.assertEqual(_trf059_mod._model_dirs_with_moe_tp_experts(), {"foo"})

    def test_trf059_accepts_canonical_signature(self):
        source = """
class FooExperts(nn.Module):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states
"""
        self.assertEqual(self._run_trf059(source), [])

    def test_trf059_accepts_common_routing_aliases(self):
        source = """
class FooExperts(nn.Module):
    def forward(self, x, selected_experts, routing_weights, implementation=None):
        return x
"""
        self.assertEqual(self._run_trf059(source), [])

    def test_trf059_flags_wrong_routing_argument_order(self):
        source = """
class FooExperts(nn.Module):
    def forward(self, hidden_states, top_k_weights, top_k_index):
        return hidden_states
"""
        violations = self._run_trf059(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("expected arg 2 to be top-k expert indices", violations[0].message)

    def test_trf059_flags_missing_routing_arguments(self):
        source = """
class FooExperts(nn.Module):
    def forward(self, hidden_states):
        return hidden_states
"""
        violations = self._run_trf059(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("expected at least 3 positional arguments after self", violations[0].message)

    def test_trf059_accepts_inherited_experts_forward(self):
        source = """
class FooExperts(BaseExperts):
    pass
"""
        self.assertEqual(self._run_trf059(source), [])

    def test_trf059_accepts_transitively_inherited_experts_forward(self):
        source = """
class IntermediateModule(BaseExperts):
    pass

class FooExperts(IntermediateModule):
    pass
"""
        self.assertEqual(self._run_trf059(source), [])

    def test_trf059_respects_class_suppression_without_forward(self):
        source = """
# trf-ignore: TRF059
class FooExperts(nn.Module):
    pass
"""
        self.assertEqual(self._run_trf059(source), [])

    def test_trf059_respects_llama4_allowlist(self):
        source = """
class Llama4TextExperts(nn.Module):
    def forward(self, hidden_states):
        return hidden_states
"""
        file_path = Path("src/transformers/models/llama4/modeling_llama4.py")
        with patch.object(_trf059_mod, "_MOE_TP_MODEL_DIRS", {"llama4"}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF059})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF059], [])
