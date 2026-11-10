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


class TRF067Test(RuleTestCase):
    # --- TRF067: don't capture every hidden state to read only some of them ---

    def test_trf067_flags_forced_kwargs_with_single_index(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        return self.multi_modal_projector(selected_image_feature)
"""
        violations = self._run(mlinter.TRF067, source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 4)
        self.assertIn("get_image_features", violations[0].message)
        self.assertIn("select_hidden_states", violations[0].message)

    def test_trf067_flags_comprehension_over_feature_layers(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in self.config.vision_feature_layer]
        return torch.cat(hs_pool, dim=-1)
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_flags_keyword_flag_and_alias(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_img_features(self, img_embeds, attention_mask):
        img_processor_output = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_states = img_processor_output.hidden_states
        return hidden_states[self.layer_idx]
"""
        violations = self._run(mlinter.TRF067, source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 5)

    def test_trf067_flags_chained_read_on_the_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values):
        return self.vision_model(pixel_values, output_hidden_states=True).hidden_states[-2]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_flags_modular_file_with_imported_base(self):
        # The rule reads only the function's own body, so an imported base class changes nothing.
        source = """
from ..llava.modeling_llava import LlavaModel


class FooModel(LlavaModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        return image_outputs.hidden_states[vision_feature_layer]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source, file_name="modular_foo.py")), 1)

    def test_trf067_allows_select_hidden_states_decorator(self):
        source = """
class FooModel(FooPreTrainedModel):
    @merge_with_config_defaults
    @can_return_tuple
    @select_hidden_states("vision_feature_layer")
    @auto_docstring
    def get_image_features(self, pixel_values, vision_feature_layer=None, **kwargs):
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        return image_outputs.hidden_states[vision_feature_layer]

    @select_hidden_states(lambda self: [i + 1 for i in self.extract_layers])
    def forward(self, pixel_values, **kwargs):
        outputs = self.clip.get_image_features(pixel_values, **kwargs)
        return [outputs.hidden_states[i + 1] for i in self.extract_layers]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_allows_selecting_by_index(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_img_features(self, img_embeds):
        outputs = self.img_processor(img_embeds, output_hidden_states=self.layer_idx)
        return outputs.hidden_states[self.layer_idx]

    def get_text_features(self, input_ids, **kwargs):
        kwargs["output_hidden_states"] = kwargs.get("output_hidden_states") or -1
        text_outputs = self.text_model(input_ids, **kwargs)
        return text_outputs.hidden_states[-1][:, 0]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_allows_stacking_the_whole_tuple(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, input_values):
        outputs = self.semantic_model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        stacked = torch.stack([h for h in hidden_states], dim=1)
        return stacked.mean(dim=1), hidden_states[-1]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_allows_slicing_or_passing_the_tuple_on(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        return self.decoder(outputs.hidden_states)

    def other(self, pixel_values):
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        return [h[0] for h in outputs.hidden_states[1:]]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_allows_filter_output_hidden_states_backbones(self):
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
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_ignores_unrelated_hidden_states_local(self):
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
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_ignores_kwargs_forced_after_the_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, pixel_values, **kwargs):
        outputs = self.encoder(pixel_values, **kwargs)
        kwargs["output_hidden_states"] = True
        return outputs.hidden_states[-1]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_ignores_configuration_files(self):
        source = """
def helper(model, x):
    return model(x, output_hidden_states=True).hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source, file_name="configuration_foo.py"), [])

    def test_trf067_suppression(self):
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, **kwargs):
        # trf-ignore: TRF067
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        return image_outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    # --- Regressions from review on huggingface/transformers-mlinter#85 ---

    def test_trf067_alias_stops_at_rebind(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs = self.enc(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hidden_states = hidden_states[-2]
        hidden_states = self.proj(hidden_states)
        return hidden_states
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_result_name_stops_at_rebind(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs = self.enc(x, output_hidden_states=True)
        first = outputs.hidden_states[-2]
        outputs = self.other(x, output_hidden_states=True)
        return first, torch.stack(outputs.hidden_states)
"""
        violations = self._run(mlinter.TRF067, source)
        self.assertEqual([violation.line_number for violation in violations], [4])

    def test_trf067_flags_annotated_result(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs: BaseModelOutput = self.enc(x, output_hidden_states=True)
        return outputs.hidden_states[-2]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_flags_kwargs_update_and_setdefault(self):
        for force in ("kwargs.update(output_hidden_states=True)", 'kwargs.setdefault("output_hidden_states", True)'):
            with self.subTest(force=force):
                source = f"""
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        {force}
        outputs = self.enc(x, **kwargs)
        return outputs.hidden_states[-2]
"""
                self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_ignores_forced_kwargs_reset_before_the_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["output_hidden_states"] = False
        outputs = self.enc(x, **kwargs)
        return outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_ignores_kwargs_forced_in_another_branch(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        if self.training:
            kwargs["output_hidden_states"] = True
            return self.loss(x, **kwargs)
        outputs = self.enc(x, **kwargs)
        return outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_flags_kwargs_forced_inside_an_enclosing_block(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        with torch.no_grad():
            kwargs["output_hidden_states"] = True
            outputs = self.enc(x, **kwargs)
        return [outputs.hidden_states[i + 1] for i in self.extract_layers]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_allows_handing_the_tuple_on(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs = self.enc(x, output_hidden_states=True)
        feat = outputs.hidden_states[-2]
        return FooOutput(last=feat, hidden_states=outputs.hidden_states)
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_still_flags_returning_the_whole_output(self):
        # The tuple itself isn't handed on, only the output holding it: that's the get_image_features pattern.
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        image_outputs.pooler_output = self.projector(image_outputs.hidden_states[vision_feature_layer])
        return image_outputs
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_allows_keyword_consumers(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs = self.enc(x, output_hidden_states=True)
        last = outputs.hidden_states[-1]
        return last, self.weighted_sum(hidden_states=outputs.hidden_states)
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_allows_indexing_every_position(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        hs = self.enc(x, output_hidden_states=True).hidden_states
        out = ()
        for i in range(len(hs)):
            out += (hs[i],)
        return out

    def stages(self, x):
        hs = self.enc(x, output_hidden_states=True).hidden_states
        return [hs[idx] for idx, stage in enumerate(self.stage_names)]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_suppression_above_a_multiline_call(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        # trf-ignore: TRF067
        outputs = self.enc(
            x,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_shared_forced_kwargs_need_every_call_to_index(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        kwargs["output_hidden_states"] = True
        selected = self.enc(x, **kwargs).hidden_states[-2]
        stacked = torch.stack(self.other(x, **kwargs).hidden_states)
        return selected, stacked
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_flags_kwargs_forced_under_a_condition(self):
        # internvl, qianfan_ocr: the force only happens on one path, but on that path it leaks.
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        if vision_feature_layer != -1:
            kwargs["output_hidden_states"] = True
        vision_outputs = self.vision_tower(pixel_values=pixel_values, **kwargs)
        if vision_feature_layer == -1:
            return vision_outputs.last_hidden_state
        return vision_outputs.hidden_states[vision_feature_layer]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_follows_kwargs_rebuilt_from_themselves(self):
        # mistral3: dropping `None` values keeps the forced flag.
        source = """
class FooModel(FooPreTrainedModel):
    def get_image_features(self, pixel_values, vision_feature_layer, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        return image_outputs.hidden_states[vision_feature_layer]
"""
        self.assertEqual(len(self._run(mlinter.TRF067, source)), 1)

    def test_trf067_ignores_kwargs_replaced_by_a_fresh_dict(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs = {"return_dict": True}
        outputs = self.enc(x, **kwargs)
        return outputs.hidden_states[-2]
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])

    def test_trf067_ignores_reads_in_a_nested_function(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, x):
        outputs = self.enc(x, output_hidden_states=True)

        def pick(outputs):
            return outputs.hidden_states[-2]

        return pick(outputs)
"""
        self.assertEqual(self._run(mlinter.TRF067, source), [])
