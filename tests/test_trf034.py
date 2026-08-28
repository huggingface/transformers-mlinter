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
    # --- TRF034: trunk layers in a ModuleList must be GradientCheckpointingLayer ---
    #
    # Every fixture expecting a finding has to describe a model the finding can happen to: one that
    # turns `supports_gradient_checkpointing` on, holding a layer that does the token mixing.

    def test_trf034_flags_plain_module_layer(self):
        source = """
class FooAttention(nn.Module):
    pass


class FooDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

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
class FooAttention(nn.Module):
    pass


class FooDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_follows_local_inheritance(self):
        source = """
class FooAttention(nn.Module):
    pass


class FooBaseLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooDecoderLayer(FooBaseLayer):
    pass


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_ignores_non_layer_modulelists(self):
        source = """
class FooExpert(nn.Module):
    pass


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.experts = nn.ModuleList([FooExpert(config) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(3)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    # --- the model has to be checkpointable for the finding to describe anything ---

    def test_trf034_skips_model_that_never_enables_checkpointing(self):
        # `PreTrainedModel.supports_gradient_checkpointing` defaults to False, so this model raises
        # from `gradient_checkpointing_enable()` rather than skipping the layer.
        source = """
class FooAttention(nn.Module):
    pass


class FooDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_skips_model_that_disables_checkpointing_explicitly(self):
        source = """
class FooAttention(nn.Module):
    pass


class FooDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_reads_support_flag_from_sibling_modeling_file(self):
        # A modular file does not repeat the XxxPreTrainedModel carrying the flag, so the model's
        # generated modeling file has to be read before the rule concludes the model opts out.
        with tempfile.TemporaryDirectory() as tmp_dir:
            foo_dir = Path(tmp_dir) / "src" / "transformers" / "models" / "foo"
            foo_dir.mkdir(parents=True)
            (foo_dir / "modeling_foo.py").write_text(
                """
class FooPreTrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True
""",
                encoding="utf-8",
            )
            source = """
class FooAttention(nn.Module):
    pass


class FooDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = FooAttention(config)


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
            violations = mlinter.analyze_file(foo_dir / "modular_foo.py", source, enabled_rules={mlinter.TRF034})

        trf034 = [violation for violation in violations if violation.rule_id == mlinter.TRF034]
        self.assertEqual(len(trf034), 1)
        self.assertIn("FooDecoderLayer", trf034[0].message)

    # --- only the token-mixing trunk is in scope ---

    def test_trf034_skips_conv_only_auxiliary_stack(self):
        # A conv backbone, a DPT head and a vocoder upsampler are ModuleLists of `*Layer` classes
        # too. Trading compute for memory there is the model author's call, not a defect.
        source = """
class FooConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(config.hidden_size, config.hidden_size, 3)
        self.activation = ACT2FN[config.hidden_act]


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.neck = nn.ModuleList([FooConvLayer(config) for _ in range(config.num_neck_layers)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    def test_trf034_flags_attention_free_trunk_by_its_mixing_module(self):
        # FocalNet-style: the trunk mixes tokens in a modulation module and has no attention at all.
        source = """
class FooModulation(nn.Module):
    pass


class FooLayer(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.modulation = FooModulation(config, dim)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooLayer(config, config.hidden_size) for _ in range(config.depths[0])])
"""
        violations = self._run(mlinter.TRF034, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("FooLayer", violations[0].message)

    def test_trf034_finds_mixing_through_a_delegating_child_block(self):
        # Florence2-style: the block holds no attention itself, the child it delegates to does.
        source = """
class FooAttention(nn.Module):
    pass


class FooSpatialBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = FooAttention(config)


class FooVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_block = FooSpatialBlock(config)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([FooVisionBlock(config) for _ in range(config.depths[0])])
"""
        violations = self._run(mlinter.TRF034, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("FooVisionBlock", violations[0].message)

    def test_trf034_does_not_read_attention_mentions_as_a_mixing_module(self):
        # `config._attn_implementation` and an `attention_mask` argument say nothing about what the
        # layer is built from; counting those would mark nearly every layer in the library as trunk.
        source = """
class FooConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 3)
        self.impl = config._attn_implementation

    def forward(self, hidden_states, attention_mask=None):
        return self.conv(hidden_states)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooConvLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    # --- checkpointing a layer that carries running statistics would corrupt them ---

    def test_trf034_skips_layer_holding_running_statistics(self):
        # Recomputation runs the forward a second time, folding every batch into the running stats
        # twice, so complying here would trade a memory saving for wrong statistics.
        source = """
class FooAttention(nn.Module):
    pass


class FooBatchNormConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = FooAttention(config)
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, 3)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.postnet = nn.ModuleList([FooBatchNormConvLayer(config) for _ in range(config.postnet_layers)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source), [])

    # --- base chains that cross model files ---

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
class LlamaAttention(nn.Module):
    pass


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
""",
                encoding="utf-8",
            )
            source = """
from ..llama.modeling_llama import LlamaDecoderLayer


class FooDecoderLayer(LlamaDecoderLayer):
    pass


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
            modular_path = foo_dir / "modular_foo.py"
            violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})
            self.assertEqual([violation for violation in violations if violation.rule_id == mlinter.TRF034], [])

            llama_path.write_text(
                """
class LlamaAttention(nn.Module):
    pass


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
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
class RTDetrAttention(nn.Module):
    pass


class RTDetrDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RTDetrAttention(config)
""",
                encoding="utf-8",
            )
            source = """
from ..rt_detr.modeling_rt_detr import RTDetrDecoderLayer


class FooDecoderLayer(RTDetrDecoderLayer):
    pass


class FooModel(FooPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
            modular_path = foo_dir / "modular_foo.py"
            violations = mlinter.analyze_file(modular_path, source, enabled_rules={mlinter.TRF034})

        trf034 = [violation for violation in violations if violation.rule_id == mlinter.TRF034]
        self.assertEqual(len(trf034), 1)
        self.assertIn("FooDecoderLayer", trf034[0].message)
