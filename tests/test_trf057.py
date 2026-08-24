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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf057_mod, date, mlinter, patch, tempfile


class TRF057Test(RuleTestCase):
    # --- TRF057: @auto_docstring on the documented classes and their entry points ---

    def _trf057_modular(self, modular_source, generated_files):
        """Check a modular file against generated siblings, given as {file name: source}."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            for name, source in generated_files.items():
                banner = f"# {_helpers_mod.GENERATED_FILE_MARKER} src/transformers/models/foo/modular_foo.py\n"
                (model_dir / name).write_text(banner + source, encoding="utf-8")
            file_path = model_dir / "modular_foo.py"
            with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
                violations = mlinter.analyze_file(file_path, modular_source, enabled_rules={mlinter.TRF057})
            return [v for v in violations if v.rule_id == mlinter.TRF057]

    def test_trf057_flags_undecorated_model_class(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
"""
        violations = self._run(mlinter.TRF057, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooModel` is a public class", violations[0].message)

    def test_trf057_accepts_decorated_model_class(self):
        source = """
@auto_docstring
class FooForConditionalGeneration(FooPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_ignores_inner_model_class_names(self):
        source = """
class FooEncoder(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

class FooTextTransformer(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_ignores_generic_task_mixin_subclasses(self):
        source = """
class FooForSequenceClassification(GenericForSequenceClassification, FooPreTrainedModel): ...

class FooForTokenClassification(GenericForTokenClassification, FooPreTrainedModel): ...
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_flags_undecorated_output_class(self):
        source = """
@dataclass
class FooModelOutputWithPast(ModelOutput):
    logits: torch.FloatTensor | None = None
"""
        violations = self._run(mlinter.TRF057, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooModelOutputWithPast` is a public class", violations[0].message)
        self.assertIn("Add @auto_docstring above @dataclass", violations[0].message)

    def test_trf057_asks_for_the_innermost_position_without_an_inner_decorator(self):
        source = """
@register_config
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
"""
        violations = self._run(mlinter.TRF057, source, file_name="configuration_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("Add @auto_docstring as the innermost decorator", violations[0].message)

    def test_trf057_asks_for_the_strict_position(self):
        source = """
@strict(accept_kwargs=True)
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
"""
        violations = self._run(mlinter.TRF057, source, file_name="configuration_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("Add @auto_docstring above @strict", violations[0].message)

    def test_trf057_prefers_the_strict_position_over_the_dataclass_one(self):
        source = """
@strict
@dataclass
class FooModelOutputWithPast(ModelOutput):
    logits: torch.FloatTensor | None = None
"""
        violations = self._run(mlinter.TRF057, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("Add @auto_docstring above @strict", violations[0].message)

    def test_trf057_asks_for_the_dataclass_position_in_a_modular_file(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModelOutputWithPast

class FooModelOutputWithPast(LlamaModelOutputWithPast):
    logits: torch.FloatTensor | None = None
"""
        generated = """
@dataclass
class FooModelOutputWithPast(ModelOutput):
    logits: torch.FloatTensor | None = None
"""
        violations = self._trf057_modular(modular_source, {"modeling_foo.py": generated})
        self.assertEqual(len(violations), 1)
        self.assertIn("Add @auto_docstring above @dataclass", violations[0].message)

    def test_trf057_flags_undecorated_config_class(self):
        source = """
@strict(accept_kwargs=True)
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
"""
        violations = self._run(mlinter.TRF057, source, file_name="configuration_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooConfig` is a public class", violations[0].message)

    def test_trf057_accepts_decorated_config_class(self):
        source = """
@strict(accept_kwargs=True)
@auto_docstring(checkpoint="acme/foo")
class FooTextConfig(FooConfig):
    hidden_size: int = 768
"""
        self.assertEqual(self._run(mlinter.TRF057, source, file_name="configuration_foo.py"), [])

    def test_trf057_flags_undecorated_processor_class(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}

class FooProcessor(ProcessorMixin):
    @auto_docstring
    def __call__(self, images=None, text=None, **kwargs):
        return BatchFeature()
"""
        violations = self._run(mlinter.TRF057, source, file_name="processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooProcessor` is a public class", violations[0].message)

    def test_trf057_ignores_classes_without_bases(self):
        source = """
class AutoProcessor:
    def __call__(self, *args, **kwargs):
        raise EnvironmentError("Use from_pretrained")
"""
        self.assertEqual(self._run(mlinter.TRF057, source, file_name="processing_foo.py"), [])

    def test_trf057_flags_undecorated_forward(self):
        source = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        violations = self._run(mlinter.TRF057, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooModel.forward` is a public method", violations[0].message)

    def test_trf057_accepts_decorated_forward(self):
        source = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_flags_undecorated_feature_getters(self):
        source = """
@auto_docstring
class FooForConditionalGeneration(FooPreTrainedModel):
    def get_image_features(self, pixel_values):
        return self.vision_tower(pixel_values)

    def get_video_features(self, pixel_values_videos):
        return self.vision_tower(pixel_values_videos)
"""
        violations = self._run(mlinter.TRF057, source)
        self.assertEqual(len(violations), 2)
        self.assertIn("get_image_features", violations[0].message)
        self.assertIn("get_video_features", violations[1].message)

    def test_trf057_ignores_inner_layer_forward(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states

class FooDecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states):
        return hidden_states
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_flags_undecorated_preprocess(self):
        source = """
@auto_docstring
class FooImageProcessor(TorchvisionBackend):
    def preprocess(self, images, **kwargs):
        return images
"""
        violations = self._run(mlinter.TRF057, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooImageProcessor.preprocess` is a public method", violations[0].message)

    def test_trf057_accepts_decorated_preprocess(self):
        source = """
@auto_docstring
class FooImageProcessor(PilBackend):
    @auto_docstring
    def preprocess(self, images, **kwargs):
        return images
"""
        self.assertEqual(self._run(mlinter.TRF057, source, file_name="image_processing_pil_foo.py"), [])

    def test_trf057_flags_undecorated_processor_call(self):
        source = """
@auto_docstring
class FooProcessor(ProcessorMixin):
    def __call__(self, images=None, text=None, **kwargs):
        return BatchFeature()
"""
        violations = self._run(mlinter.TRF057, source, file_name="processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooProcessor.__call__`", violations[0].message)

    def test_trf057_flags_processor_whose_base_is_another_model(self):
        source = """
from ..gemma3.processing_gemma3 import Gemma3Processor

@auto_docstring
class FooProcessor(Gemma3Processor):
    def __call__(self, images=None, text=None, **kwargs):
        return BatchFeature()
"""
        violations = self._run(mlinter.TRF057, source, file_name="processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooProcessor.__call__`", violations[0].message)

    def test_trf057_ignores_helper_call_in_an_image_processing_file(self):
        source = """
class FooMaskingGenerator:
    def __call__(self):
        return self.mask
"""
        self.assertEqual(self._run(mlinter.TRF057, source, file_name="image_processing_foo.py"), [])

    def test_trf057_ignores_video_processors(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    def preprocess(self, videos, **kwargs):
        return videos
"""
        self.assertEqual(self._run(mlinter.TRF057, source, file_name="video_processing_foo.py"), [])

    def test_trf057_respects_suppression(self):
        source = """
# trf-ignore: TRF057
class FooModel(FooPreTrainedModel):
    # trf-ignore: TRF057
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        self.assertEqual(self._run(mlinter.TRF057, source), [])

    def test_trf057_exempts_models_before_cutoff(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2023, 1, 1)):
            with patch.object(_trf057_mod, "CUTOFF_DATE", "2026-06-20"):
                violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF057})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF057], [])

    def test_trf057_modular_class_and_method_inherit_the_decorator(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    def forward(self, input_ids):
        return super().forward(input_ids)
"""
        generated = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    @auto_docstring
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        self.assertEqual(self._trf057_modular(modular_source, {"modeling_foo.py": generated}), [])

    def test_trf057_flags_modular_class_that_ships_undecorated(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    pass
"""
        generated = """
class FooModel(FooPreTrainedModel):
    pass
"""
        violations = self._trf057_modular(modular_source, {"modeling_foo.py": generated})
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooModel` in modeling_foo.py has no @auto_docstring", violations[0].message)
        self.assertIn("No parent class", violations[0].message)

    def test_trf057_flags_modular_override_that_ships_undecorated(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    def forward(self, input_ids):
        return super().forward(input_ids)
"""
        generated = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        violations = self._trf057_modular(modular_source, {"modeling_foo.py": generated})
        self.assertEqual(len(violations), 1)
        self.assertIn("`FooModel.forward` in modeling_foo.py has no @auto_docstring", violations[0].message)
        self.assertIn("No parent method", violations[0].message)

    def test_trf057_reports_dropped_parent_decorators(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    @capture_outputs
    def forward(self, input_ids):
        return super().forward(input_ids)
"""
        generated = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    @capture_outputs
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        violations = self._trf057_modular(modular_source, {"modeling_foo.py": generated})
        self.assertEqual(len(violations), 1)
        self.assertIn("Declaring @capture_outputs", violations[0].message)

    def test_trf057_accepts_modular_decorator_before_regeneration(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

@auto_docstring
class FooModel(LlamaModel):
    @auto_docstring
    def forward(self, input_ids):
        return super().forward(input_ids)
"""
        stale_generated = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        self.assertEqual(self._trf057_modular(modular_source, {"modeling_foo.py": stale_generated}), [])

    def test_trf057_ignores_modular_method_deleted_by_the_converter(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    def get_video_features(self):
        raise AttributeError("Not needed for Foo")
"""
        generated = """
@auto_docstring
class FooModel(FooPreTrainedModel):
    @auto_docstring
    def forward(self, input_ids):
        return self.layers(input_ids)
"""
        self.assertEqual(self._trf057_modular(modular_source, {"modeling_foo.py": generated}), [])

    def test_trf057_finds_the_generated_file_by_banner_not_by_name(self):
        modular_source = """
from ..llava.image_processing_pil_llava import LlavaImageProcessorPil

class FooImageProcessorPil(LlavaImageProcessorPil):
    def preprocess(self, images, **kwargs):
        return images
"""
        generated = """
@auto_docstring
class FooImageProcessorPil(PilBackend):
    def preprocess(self, images, **kwargs):
        return images
"""
        violations = self._trf057_modular(modular_source, {"image_processing_pil_foo.py": generated})
        self.assertEqual(len(violations), 1)
        self.assertIn("in image_processing_pil_foo.py", violations[0].message)

    def test_trf057_ignores_modular_class_absent_from_the_generated_files(self):
        modular_source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    def forward(self, input_ids):
        return super().forward(input_ids)
"""
        self.assertEqual(self._trf057_modular(modular_source, {}), [])
