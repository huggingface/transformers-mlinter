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


class TRF021Test(RuleTestCase):
    # --- TRF021: scalar tensors must be filled on-device, not copied from host ---

    def _trf021(self, modeling_source: str, config_source: str | None = None) -> list:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            if config_source is not None:
                (model_dir / "configuration_foo.py").write_text(config_source, encoding="utf-8")
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF021})
            return [v for v in violations if v.rule_id == mlinter.TRF021]

    def test_trf021_flags_scalar_config_field_copied_to_device(self):
        config_source = """
class FooConfig(PreTrainedConfig):
    image_token_id: int | None = 258880
"""
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def get_placeholder_mask(self, input_ids, inputs_embeds):
        return (
            inputs_embeds
            == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
        ).all(-1)
"""
        trf021 = self._trf021(modeling_source, config_source)
        self.assertEqual(len(trf021), 1)
        self.assertIn("torch.full((), self.config.image_token_id", trf021[0].message)

    def test_trf021_allows_torch_full_rewrite(self):
        config_source = """
class FooConfig(PreTrainedConfig):
    image_token_id: int | None = 258880
"""
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def get_placeholder_mask(self, input_ids, inputs_embeds):
        return (
            inputs_embeds
            == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
        ).all(-1)
"""
        self.assertEqual(self._trf021(modeling_source, config_source), [])

    def test_trf021_flags_numeric_literal_and_finfo_scalars(self):
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states, attention_mask):
        floor = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        ceiling = torch.tensor(torch.finfo(hidden_states.dtype).min, device=hidden_states.device)
        return floor, ceiling
"""
        trf021 = self._trf021(modeling_source)
        self.assertEqual(len(trf021), 2)

    def test_trf021_skips_sequence_valued_config_field(self):
        # eos_token_id may be a list, so `torch.full((), ...)` is not a valid rewrite.
        config_source = """
class FooConfig(PreTrainedConfig):
    eos_token_id: int | list[int] | None = 2
    class_thresholds: list[float] | None = None
"""
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        stop = torch.tensor(self.config.eos_token_id, dtype=torch.long, device=inputs_embeds.device)
        thresholds = torch.tensor(self.config.class_thresholds, device=inputs_embeds.device)
        return stop, thresholds
"""
        self.assertEqual(self._trf021(modeling_source, config_source), [])

    def test_trf021_skips_list_literals_and_unresolved_names(self):
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds, spatial_shapes):
        shapes = torch.tensor([1, 2, 3], device=inputs_embeds.device)
        unknown = torch.tensor(spatial_shapes, device=inputs_embeds.device)
        return shapes, unknown
"""
        self.assertEqual(self._trf021(modeling_source), [])

    def test_trf021_skips_cpu_device_and_missing_device(self):
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        pinned = torch.tensor(0.0, device="cpu")
        indexed = torch.tensor(0.0, device="cpu:0")
        wrapped = torch.tensor(0.0, device=torch.device("cpu"))
        keyworded = torch.tensor(0.0, device=torch.device(type="cpu"))
        hostside = torch.tensor(0.0)
        return pinned, indexed, wrapped, keyworded, hostside
"""
        self.assertEqual(self._trf021(modeling_source), [])

    def test_trf021_flags_accelerator_whose_name_contains_cpu(self):
        # The host check matches the device type exactly, so a backend merely containing "cpu" in
        # its name is still an accelerator and the copy still breaks graph capture.
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        named = torch.tensor(0.0, device="mycpu")
        wrapped = torch.tensor(0.0, device=torch.device("cpuplus:0"))
        return named, wrapped
"""
        self.assertEqual(len(self._trf021(modeling_source)), 2)

    def test_trf021_skips_construction_time_methods(self):
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config, device):
        super().__init__(config)
        self.register_buffer("scale", torch.tensor(1.0, device=device))

    def _init_weights(self, module):
        module.gate = torch.tensor(0.0, device=module.weight.device)
"""
        self.assertEqual(self._trf021(modeling_source), [])

    def test_trf021_resolves_locals_and_self_attributes(self):
        config_source = """
class FooConfig(PreTrainedConfig):
    image_token_id: int = 32000
    min_depth: float = 0.001
"""
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.min_depth = config.min_depth

    def forward(self, inputs_embeds):
        image_token_id = self.config.image_token_id
        mask = torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
        floor = torch.tensor(self.min_depth, device=inputs_embeds.device)
        return mask, floor
"""
        trf021 = self._trf021(modeling_source, config_source)
        self.assertEqual(len(trf021), 2)

    def test_trf021_follows_config_attribute_map_alias(self):
        config_source = """
class FooConfig(PreTrainedConfig):
    attribute_map = {"image_token_id": "image_token_index"}
    image_token_index: int = 32000
"""
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        return torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
"""
        self.assertEqual(len(self._trf021(modeling_source, config_source)), 1)

    def test_trf021_resolves_the_config_class_the_model_targets(self):
        # Two config classes in one file annotate `image_token_id` differently. Only the class the
        # modeling class actually targets may decide whether the value is a scalar.
        config_source = """
class FooTextConfig(PreTrainedConfig):
    image_token_id: list[int] | None = None

class FooConfig(PreTrainedConfig):
    image_token_id: int = 32000
"""
        modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig

class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        return torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
"""
        self.assertEqual(len(self._trf021(modeling_source, config_source)), 1)

        text_modeling_source = """
class FooTextPreTrainedModel(PreTrainedModel):
    config_class = FooTextConfig

class FooTextModel(FooTextPreTrainedModel):
    def forward(self, inputs_embeds):
        return torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
"""
        self.assertEqual(self._trf021(text_modeling_source, config_source), [])

    def test_trf021_respects_suppression_comment(self):
        modeling_source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds):
        # trf-ignore: TRF021
        return torch.tensor(0.0, device=inputs_embeds.device)
"""
        self.assertEqual(self._trf021(modeling_source), [])

    def test_trf021_skips_non_modeling_files(self):
        source = """
class FooProcessor(ProcessorMixin):
    def __call__(self, images, device):
        return torch.tensor(0.0, device=device)
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF021})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF021], [])
