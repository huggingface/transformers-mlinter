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


from tests.rule_test_utils import Path, RuleTestCase, mlinter, tempfile


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
