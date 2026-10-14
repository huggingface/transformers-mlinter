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


class TRF060Test(RuleTestCase):
    # --- TRF060: don't capture every hidden state to read only some of them ---

    def test_trf060_flags_forced_kwargs_with_single_index(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        return self.multi_modal_projector(selected_image_feature)
"""
        violations = self._run(mlinter.TRF060, source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 4)
        self.assertIn("get_image_features", violations[0].message)
        self.assertIn("hidden_states_to_capture", violations[0].message)

    def test_trf060_flags_comprehension_over_feature_layers(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in self.config.vision_feature_layer]
        return torch.cat(hs_pool, dim=-1)
"""
        self.assertEqual(len(self._run(mlinter.TRF060, source)), 1)

    def test_trf060_flags_keyword_flag_and_alias(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_img_features(self, img_embeds, attention_mask):
        img_processor_output = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_states = img_processor_output.hidden_states
        return hidden_states[self.layer_idx]
"""
        violations = self._run(mlinter.TRF060, source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 5)

    def test_trf060_flags_chained_read_on_the_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values):
        return self.vision_model(pixel_values, output_hidden_states=True).hidden_states[-2]
"""
        self.assertEqual(len(self._run(mlinter.TRF060, source)), 1)

    def test_trf060_flags_modular_file_with_imported_base(self):
        # The rule reads only the function's own body, so an imported base class changes nothing.
        source = """
from ..llava.modeling_llava import LlavaModel


class FooModel(LlavaModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        return image_outputs.hidden_states[vision_feature_layer]
"""
        self.assertEqual(len(self._run(mlinter.TRF060, source, file_name="modular_foo.py")), 1)

    def test_trf060_allows_hidden_states_to_capture(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        vision_feature_layer, kwargs["output_hidden_states"] = hidden_states_to_capture(
            vision_feature_layer, self.config.vision_config, kwargs.get("output_hidden_states")
        )
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        return image_outputs.hidden_states[vision_feature_layer]
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_allows_stacking_the_whole_tuple(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_values):
        outputs = self.semantic_model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        stacked = torch.stack([h for h in hidden_states], dim=1)
        return stacked.mean(dim=1), hidden_states[-1]
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_allows_slicing_or_passing_the_tuple_on(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        return self.decoder(outputs.hidden_states)

    def other(self, pixel_values):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        return [h[0] for h in outputs.hidden_states[1:]]
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_allows_filter_output_hidden_states_backbones(self):
        source = """
class FooBackbone(FooPreTrainedModel, BackboneMixin):
    @can_return_tuple
    @filter_output_hidden_states
    def forward(self, pixel_values):
        outputs = self.foo(pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)
        return BackboneOutput(feature_maps=feature_maps)
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_ignores_unrelated_hidden_states_local(self):
        # `hidden_states` is assigned before the forced call and never aliases its output tuple.
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, pixel_values, **kwargs):
        hidden_states = self.embeddings(pixel_values)
        first = hidden_states[0]
        kwargs["output_hidden_states"] = True
        outputs = self.encoder(first, **kwargs)
        return self.head(outputs.hidden_states)
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_ignores_kwargs_forced_after_the_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, pixel_values, **kwargs):
        outputs = self.encoder(pixel_values, **kwargs)
        kwargs["output_hidden_states"] = True
        return outputs.hidden_states[-1]
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])

    def test_trf060_ignores_configuration_files(self):
        source = """
def helper(model, x):
    return model(x, output_hidden_states=True).hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF060, source, file_name="configuration_foo.py"), [])

    def test_trf060_suppression(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, **kwargs):
        # trf-ignore: TRF060
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        return image_outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF060, source), [])
