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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf034_mod, date, mlinter, patch, tempfile


class TRF034Test(RuleTestCase):
    # --- TRF034: ModuleList layers must be GradientCheckpointingLayer ---

    def test_trf034_flags_plain_module_layer(self):
        source = """
class FooDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
"""
        violations = self._run(mlinter.TRF034, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("FooDecoderLayer", violations[0].message)
        self.assertIn("GradientCheckpointingLayer", violations[0].message)

    def test_trf034_accepts_gradient_checkpointing_layer(self):
        source = """
class FooDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_does_not_report_a_base_class_a_grandfathered_model_owns(self):
        """The parent model owns the structure, so the cutoff has to be read against the parent's file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir) / "src" / "transformers" / "models"
            (models_root / "rt_detr").mkdir(parents=True)
            (models_root / "d_fine").mkdir()
            (models_root / "rt_detr" / "modeling_rt_detr.py").write_text(
                "class RTDetrRepVggBlock(nn.Module):\n    pass\n", encoding="utf-8"
            )
            modular_path = models_root / "d_fine" / "modular_d_fine.py"
            source = """
from ..rt_detr.modeling_rt_detr import RTDetrRepVggBlock


class DFineRepVggBlock(RTDetrRepVggBlock):
    pass


class DFineEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([DFineRepVggBlock(config) for _ in range(2)])
"""

            def run(parent_date):
                # The model being linted has no contribution date, so it is never grandfathered itself.
                def contribution_date(path):
                    return parent_date if "rt_detr" in str(path) else None

                with (
                    patch.object(_helpers_mod, "MODELS_ROOT", models_root),
                    patch.object(_helpers_mod, "model_contribution_date", side_effect=contribution_date),
                    patch.object(_trf034_mod, "CUTOFF_DATE", "2026-06-20"),
                ):
                    violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})
                return [violation for violation in violations if violation.rule_id == mlinter.TRF034]

            # rt_detr predates the cutoff: d_fine cannot fix RTDetrRepVggBlock, so nothing is reported.
            self.assertEqual(run(date(2024, 1, 1)), [])
            # A parent the cutoff does not cover is a parent that can be fixed, so the finding stands.
            self.assertEqual(len(run(date(2026, 7, 1))), 1)

    def test_trf034_still_reports_a_base_owned_by_the_model_being_linted(self):
        """Inheriting inside your own model is no excuse, whatever the model's own date says."""
        source = """
class FooBaseLayer(nn.Module):
    pass


class FooDecoderLayer(FooBaseLayer):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(len(self._run(mlinter.TRF034, source)), 1)

    def test_trf034_follows_local_inheritance(self):
        source = """
class FooBaseLayer(GradientCheckpointingLayer):
    pass


class FooDecoderLayer(FooBaseLayer):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_allows_imported_modular_layer_base(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir) / "src" / "transformers" / "models"
            llama_dir = models_root / "llama"
            foo_dir = models_root / "foo"
            llama_dir.mkdir(parents=True)
            foo_dir.mkdir()
            llama_path = llama_dir / "modeling_llama.py"
            llama_path.write_text(
                """
class LlamaDecoderLayer(GradientCheckpointingLayer):
    pass
""",
                encoding="utf-8",
            )
            source = """
from ..llama.modeling_llama import LlamaDecoderLayer


class FooDecoderLayer(LlamaDecoderLayer):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
            modular_path = foo_dir / "modular_foo.py"
            violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})
            self.assertEqual([violation for violation in violations if violation.rule_id == mlinter.TRF034], [])

            llama_path.write_text(
                """
class LlamaDecoderLayer(nn.Module):
    pass
""",
                encoding="utf-8",
            )
            violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})

        self.assertEqual(len([violation for violation in violations if violation.rule_id == mlinter.TRF034]), 1)

    def test_trf034_reports_imported_modular_layer_base_that_resolves_to_plain_module(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir) / "src" / "transformers" / "models"
            rt_detr_dir = models_root / "rt_detr"
            foo_dir = models_root / "foo"
            rt_detr_dir.mkdir(parents=True)
            foo_dir.mkdir()
            (rt_detr_dir / "modeling_rt_detr.py").write_text(
                """
class RTDetrDecoderLayer(nn.Module):
    pass
""",
                encoding="utf-8",
            )
            source = """
from ..rt_detr.modeling_rt_detr import RTDetrDecoderLayer


class FooDecoderLayer(RTDetrDecoderLayer):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
            modular_path = foo_dir / "modular_foo.py"
            violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})

        trf034 = [violation for violation in violations if violation.rule_id == mlinter.TRF034]
        self.assertEqual(len(trf034), 1)
        self.assertIn("FooDecoderLayer", trf034[0].message)

    def test_trf034_ignores_non_layer_modulelists(self):
        source = """
class FooExpert(nn.Module):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.experts = nn.ModuleList([FooExpert(config) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(3)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])
