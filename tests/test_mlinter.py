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

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlinter as public_api
from mlinter import _helpers as _helpers_mod
from mlinter import _version as _version_mod
from mlinter import mlinter
from mlinter import trf011 as _trf011_mod


TEST_PP_PLAN_MODULES = {"foo": {"embed_tokens", "final_layer_norm", "layers", "norm"}}


def _write_custom_rules_toml(
    tmp_dir: Path, *, trf001_description: str | None = None, trf001_default_enabled: bool | None = None
) -> Path:
    text = mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8")
    if trf001_description is not None:
        text = text.replace(
            'description = "Class-level config_class on <Model>PreTrainedModel should match <Model>Config naming."',
            f'description = "{trf001_description}"',
            1,
        )
    if trf001_default_enabled is not None:
        replacement = "true" if trf001_default_enabled else "false"
        text = text.replace("default_enabled = true", f"default_enabled = {replacement}", 1)

    custom_rules_path = tmp_dir / "custom_rules.toml"
    custom_rules_path.write_text(text, encoding="utf-8")
    return custom_rules_path


class CheckModelingStructureTest(unittest.TestCase):
    # --- TRF001: config_class naming consistency (old TRF003) ---

    def test_trf001_valid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF001})
        trf001 = [v for v in violations if v.rule_id == mlinter.TRF001]
        self.assertEqual(trf001, [])

    def test_trf001_invalid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = BarConfig
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF001})
        trf001 = [v for v in violations if v.rule_id == mlinter.TRF001]
        self.assertEqual(len(trf001), 1)
        self.assertIn("config_class is BarConfig, expected FooConfig", trf001[0].message)

    # --- TRF002: base_model_prefix (old TRF004) ---

    def test_trf002_valid_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF002})
        trf002 = [v for v in violations if v.rule_id == mlinter.TRF002]
        self.assertEqual(trf002, [])

    def test_trf002_invalid_empty_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = ""
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF002})
        trf002 = [v for v in violations if v.rule_id == mlinter.TRF002]
        self.assertEqual(len(trf002), 1)
        self.assertIn("non-empty canonical token", trf002[0].message)

    # --- TRF003: capture_output enforcement (reworked old TRF005) ---

    def test_trf003_flags_old_return_dict_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        if not return_dict:
            return (x,)
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF003})
        trf003 = [v for v in violations if v.rule_id == mlinter.TRF003]
        self.assertEqual(len(trf003), 1)
        self.assertIn("old return_dict branching pattern", trf003[0].message)

    def test_trf003_allows_no_return_dict_arg(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF003})
        trf003 = [v for v in violations if v.rule_id == mlinter.TRF003]
        self.assertEqual(trf003, [])

    def test_trf003_allows_return_dict_without_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF003})
        trf003 = [v for v in violations if v.rule_id == mlinter.TRF003]
        self.assertEqual(trf003, [])

    # --- TRF004: tie_weights hard ban (reworked old TRF007) ---

    def test_trf004_flags_any_tie_weights_override(self):
        source = """
class FooModel:
    def tie_weights(self):
        super().tie_weights()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF004})
        trf004 = [v for v in violations if v.rule_id == mlinter.TRF004]
        self.assertEqual(len(trf004), 1)
        self.assertIn("overrides tie_weights", trf004[0].message)

    def test_trf004_allows_no_tie_weights(self):
        source = """
class FooModel:
    _tied_weights_keys = ["lm_head.weight"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF004})
        trf004 = [v for v in violations if v.rule_id == mlinter.TRF004]
        self.assertEqual(trf004, [])

    # --- TRF005: _no_split_modules (old TRF008) ---

    def test_trf005_valid_no_split_modules(self):
        source = """
class FooModel:
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(trf005, [])

    def test_trf005_invalid_empty_string(self):
        source = """
class FooModel:
    _no_split_modules = [""]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(len(trf005), 1)

    def test_trf005_allows_attribute_error_sentinel_in_modular(self):
        source = """
class FooModel(BarModel):
    _no_split_modules = AttributeError()
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(trf005, [])

    def test_trf005_rejects_attribute_error_sentinel_in_modeling(self):
        source = """
class FooModel(BarModel):
    _no_split_modules = AttributeError()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(len(trf005), 1)

    # --- TRF006: cache args usage (old TRF010) ---

    def test_trf006_catches_unused_cache_args(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF006})
        trf006 = [v for v in violations if v.rule_id == mlinter.TRF006]
        self.assertEqual(len(trf006), 1)
        self.assertIn("past_key_values/use_cache", trf006[0].message)

    # --- TRF007: post_init order (old TRF011) ---

    def test_trf007_flags_assignment_after_post_init(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        self.proj = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF007})
        trf007 = [v for v in violations if v.rule_id == mlinter.TRF007]
        self.assertEqual(len(trf007), 1)
        self.assertIn("assigns self.* after self.post_init()", trf007[0].message)

    def test_trf007_allows_post_init_at_end(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
        self.post_init()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF007})
        trf007 = [v for v in violations if v.rule_id == mlinter.TRF007]
        self.assertEqual(trf007, [])

    # --- TRF008: add_start_docstrings usage ---

    def test_trf008_flags_empty_add_start_docstrings(self):
        source = """
@add_start_docstrings("")
class FooPreTrainedModel(PreTrainedModel):
    pass
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF008})
        trf008 = [v for v in violations if v.rule_id == mlinter.TRF008]
        self.assertEqual(len(trf008), 1)
        self.assertIn("without non-empty docstring arguments", trf008[0].message)

    def test_trf008_allows_non_empty_add_start_docstrings(self):
        source = """
@add_start_docstrings("Foo model.")
class FooPreTrainedModel(PreTrainedModel):
    pass
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF008})
        trf008 = [v for v in violations if v.rule_id == mlinter.TRF008]
        self.assertEqual(trf008, [])

    # --- TRF009: cross-model imports (old TRF013) ---

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_flags_cross_model_import_in_modeling_file(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `llama`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_allows_same_model_import_in_modeling_file(self, _mock):
        source = """
from .configuration_foo import FooConfig
from transformers.models.foo.configuration_foo import FooConfig as FooConfigAlias
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(trf009, [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_ignores_modular_files(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        trf009 = [v for v in violations if v.rule_id == mlinter.TRF009]
        self.assertEqual(trf009, [])

    # --- TRF010: strict config decorator ---

    def test_trf010_allows_direct_config_with_strict(self):
        source = """
from huggingface_hub.dataclasses import strict

@strict
class FooConfig(PretrainedConfig):
    pass
"""
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF010})
        trf010 = [v for v in violations if v.rule_id == mlinter.TRF010]
        self.assertEqual(trf010, [])

    def test_trf010_flags_missing_strict_on_direct_config(self):
        source = """
class FooConfig(PretrainedConfig):
    pass
"""
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF010})
        trf010 = [v for v in violations if v.rule_id == mlinter.TRF010]
        self.assertEqual(len(trf010), 1)
        self.assertIn("missing @strict", trf010[0].message)

    def test_trf010_ignores_non_direct_config_alias_wrappers(self):
        source = """
from huggingface_hub.dataclasses import strict

@strict
class FooConfig(PretrainedConfig):
    pass

class FooCompatConfig(FooConfig):
    pass
"""
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF010})
        trf010 = [v for v in violations if v.rule_id == mlinter.TRF010]
        self.assertEqual(trf010, [])

    # --- TRF011: PP-safe forward (no submodule attribute access) ---

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_flags_layer_attr_access_in_forward_loop(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_map[decoder_layer.attention_type],
            )
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("decoder_layer.attention_type", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_flags_enumerate_loop_variant(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for i, layer in enumerate(self.layers):
            mask = mask_map[layer.layer_type]
            hidden_states = layer(hidden_states, attention_mask=mask)
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("layer.layer_type", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_flags_sliced_layers_loop(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = layer(hidden_states, mask=layer.is_sliding)
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("layer.is_sliding", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", {"foo": {"blocks"}})
    def test_trf011_flags_non_layers_pp_loop(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states, mask=block.layer_type)
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("block.layer_type", trf011[0].message)
        self.assertIn("self.blocks", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_flags_embedding_attr_access(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        padding_idx = self.embed_tokens.padding_idx
        return self.embed_tokens(input_ids.masked_fill(input_ids == padding_idx, 0))
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("self.embed_tokens.padding_idx", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_flags_final_norm_attr_access(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        return self.final_layer_norm(hidden_states.to(dtype=self.final_layer_norm.weight.dtype))
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(len(trf011), 1)
        self.assertIn("self.final_layer_norm.weight", trf011[0].message)

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_allows_config_based_lookup(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_map[self.config.layer_types[i]],
            )
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(trf011, [])

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_allows_nn_module_attrs(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for layer in self.layers:
            if layer.training:
                hidden_states = layer(hidden_states)
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(trf011, [])

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_allows_nn_module_attrs_on_direct_pp_submodule(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, input_ids):
        if self.embed_tokens.training:
            return self.embed_tokens(input_ids)
        return input_ids
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(trf011, [])

    def test_trf011_skips_models_without_pp_plan(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=layer.attention_type)
        return hidden_states
"""
        file_path = Path("src/transformers/models/no_pp_model/modeling_no_pp_model.py")
        with patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", {}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(trf011, [])

    @patch.object(_trf011_mod, "_PP_PLAN_MODULES_BY_MODEL_DIR", TEST_PP_PLAN_MODULES)
    def test_trf011_suppression_works(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states):
        for layer in self.layers:
            # trf-ignore: TRF011
            hidden_states = layer(hidden_states, mask=layer.attention_type)
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF011})
        trf011 = [v for v in violations if v.rule_id == mlinter.TRF011]
        self.assertEqual(trf011, [])

    # --- TRF012: _init_weights should use init primitives ---

    def test_trf012_flags_inplace_module_weight_ops(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        module.weight.normal_(mean=0.0, std=0.02)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF012})
        trf012 = [v for v in violations if v.rule_id == mlinter.TRF012]
        self.assertEqual(len(trf012), 1)
        self.assertIn("in-place operation on a module's weight", trf012[0].message)

    def test_trf012_allows_init_primitives(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        init.normal_(module.weight, mean=0.0, std=0.02)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF012})
        trf012 = [v for v in violations if v.rule_id == mlinter.TRF012]
        self.assertEqual(trf012, [])

    # --- TRF013: __init__ should call self.post_init ---

    def test_trf013_flags_missing_post_init(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF013})
        trf013 = [v for v in violations if v.rule_id == mlinter.TRF013]
        self.assertEqual(len(trf013), 1)
        self.assertIn("does not call `self.post_init`", trf013[0].message)

    def test_trf013_allows_post_init(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
        self.post_init()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF013})
        trf013 = [v for v in violations if v.rule_id == mlinter.TRF013]
        self.assertEqual(trf013, [])

    # --- Utility tests ---

    def test_package_root_reexports_supported_api(self):
        self.assertIs(public_api.analyze_file, mlinter.analyze_file)
        self.assertIs(public_api.format_rule_details, mlinter.format_rule_details)
        self.assertIs(public_api.render_rules_reference, mlinter.render_rules_reference)
        self.assertIs(public_api.Violation, _helpers_mod.Violation)
        self.assertEqual(public_api.__version__, mlinter.__version__)
        self.assertIs(public_api.collect_class_bases, _helpers_mod._collect_class_bases)
        self.assertIs(public_api.has_rule_suppression, _helpers_mod._has_rule_suppression)
        self.assertIs(public_api.inherits_pretrained_model, _helpers_mod._inherits_pretrained_model)
        self.assertIs(public_api.model_dir_name, _helpers_mod._model_dir_name)
        self.assertIs(public_api.is_rule_allowlisted_for_file, mlinter._is_rule_allowlisted_for_file)
        self.assertEqual(public_api.TRF001, "TRF001")
        self.assertEqual(public_api.TRF015, "TRF015")
        self.assertEqual(public_api.TRF016, "TRF016")

    def test_package_root_all_lists_supported_api(self):
        self.assertIn("__version__", public_api.__all__)
        self.assertIn("analyze_file", public_api.__all__)
        self.assertIn("collect_class_bases", public_api.__all__)
        self.assertIn("model_dir_name", public_api.__all__)
        self.assertIn("render_rules_reference", public_api.__all__)
        self.assertIn("TRF001", public_api.__all__)
        self.assertIn("TRF015", public_api.__all__)
        self.assertIn("TRF016", public_api.__all__)
        self.assertNotIn("_collect_class_bases", public_api.__all__)
        self.assertNotIn("_rule_id", public_api.__all__)

    def test_mlinter_module_does_not_leak_rule_loop_variable(self):
        self.assertFalse(hasattr(mlinter, "_rule_id"))

    def test_version_helper_reads_git_hash_from_direct_url(self):
        dist = SimpleNamespace(
            read_text=lambda name: json.dumps(
                {
                    "url": "https://github.com/huggingface/transformers-mlinter",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "abcdef1234567890",
                    },
                }
            )
        )

        self.assertEqual(_version_mod._read_git_hash_from_direct_url(dist), "abcdef1")

    def test_version_helper_reads_git_hash_from_checkout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            (project_root / ".git").write_text("gitdir: /tmp/fake\n", encoding="utf-8")

            with (
                patch.object(_version_mod, "PROJECT_ROOT", project_root),
                patch.object(
                    _version_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        args=["git", "rev-parse", "--short", "HEAD"],
                        returncode=0,
                        stdout="deadbee\n",
                        stderr="",
                    ),
                ),
            ):
                self.assertEqual(_version_mod._read_git_hash_from_checkout(), "deadbee")

    def test_version_helper_resolve_version_prefers_direct_url_hash(self):
        dist = SimpleNamespace(
            version="0.1.1",
            read_text=lambda name: json.dumps(
                {
                    "url": "https://github.com/huggingface/transformers-mlinter",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "abcdef1234567890",
                    },
                }
            ),
        )

        with (
            patch.object(_version_mod, "_installed_distribution", return_value=dist),
            patch.object(_version_mod, "_read_git_hash_from_checkout", return_value="deadbee"),
        ):
            self.assertEqual(_version_mod._resolve_version(), "0.1.1+gabcdef1")

    def test_version_helper_resolve_version_falls_back_without_metadata_or_pyproject(self):
        with (
            patch.object(_version_mod, "_installed_distribution", return_value=None),
            patch.object(_version_mod, "_read_version_from_pyproject", return_value=None),
            patch.object(_version_mod, "_read_git_hash_from_checkout", return_value=None),
        ):
            self.assertEqual(_version_mod._resolve_version(), _version_mod.DEFAULT_BASE_VERSION)

    def test_parse_args_version_prints_version_and_exits(self):
        stdout = StringIO()
        with patch.object(mlinter.sys, "argv", ["mlinter", "--version"]), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as exc:
                mlinter.parse_args()

        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), f"mlinter {mlinter.__version__}\n")

    def test_parse_args_accepts_custom_rules_toml(self):
        custom_rules_path = Path("/tmp/custom_rules.toml")
        with patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path)]):
            args = mlinter.parse_args()

        self.assertEqual(args.rules_toml, custom_rules_path)

    def test_render_rules_reference_matches_rule_specs(self):
        rendered = public_api.render_rules_reference()
        self.assertEqual(rendered.count("### TRF"), len(public_api.TRF_RULE_SPECS))
        self.assertTrue(rendered.endswith("\n"))

    def test_main_uses_custom_rules_toml_for_rule_listing_and_restores_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(
                Path(tmp_dir),
                trf001_description="Custom config_class guidance.",
                trf001_default_enabled=False,
            )
            stdout = StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stdout(stdout),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("TRF001: Custom config_class guidance. (default: disabled)", rendered)
        self.assertIn(
            "Class-level config_class on <Model>PreTrainedModel should match <Model>Config naming.",
            mlinter.format_rule_summary("TRF001"),
        )

    def test_content_hash_changes_with_custom_rule_specs(self):
        source = "class FooPreTrainedModel(PreTrainedModel):\n    pass\n"
        default_digest = mlinter._content_hash(source, {mlinter.TRF001})

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(
                Path(tmp_dir),
                trf001_description="Custom config_class guidance.",
                trf001_default_enabled=False,
            )
            with mlinter._using_rule_specs(custom_rules_path):
                custom_digest = mlinter._content_hash(source, {mlinter.TRF001})

        self.assertNotEqual(default_digest, custom_digest)
        self.assertEqual(mlinter.ACTIVE_RULE_SPECS_PATH, mlinter.DEFAULT_RULE_SPECS_PATH)

    def test_main_rejects_custom_rules_toml_with_unsupported_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(Path(tmp_dir))
            custom_rules_path.write_text(
                custom_rules_path.read_text(encoding="utf-8").replace("version = 1", "version = 2", 1),
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("expected version 1", stderr.getvalue())

    def test_analyze_file_allows_subscripted_class_bases(self):
        source = """
from collections import OrderedDict

class _LazyConfigMapping(OrderedDict[str, str]):
    pass
"""
        file_path = Path("src/transformers/models/auto/configuration_auto.py")
        violations = mlinter.analyze_file(file_path, source)
        self.assertEqual(violations, [])

    def test_cache_path_uses_xdg_cache_home_on_linux(self):
        with (
            patch.object(mlinter.sys, "platform", "linux"),
            patch.dict(mlinter.os.environ, {"XDG_CACHE_HOME": "/tmp/mlinter-xdg-cache"}, clear=True),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/tmp/mlinter-xdg-cache") / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_cache_path_uses_library_caches_on_macos(self):
        with (
            patch.object(mlinter.sys, "platform", "darwin"),
            patch.object(mlinter.Path, "home", return_value=Path("/Users/tester")),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/Users/tester") / "Library" / "Caches" / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_cache_path_uses_localappdata_on_windows(self):
        with (
            patch.object(mlinter.sys, "platform", "win32"),
            patch.dict(mlinter.os.environ, {"LOCALAPPDATA": "/tmp/localappdata"}, clear=True),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/tmp/localappdata") / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_save_cache_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "nested" / "mlinter" / mlinter.CACHE_FILENAME

            with patch("mlinter.mlinter._cache_path", return_value=cache_path):
                mlinter._save_cache({"foo.py": "digest"})

            self.assertTrue(cache_path.exists())
            self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8")), {"foo.py": "digest"})

    @patch("mlinter.mlinter.subprocess.run")
    def test_get_changed_modeling_files_includes_configuration_files(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout=(
                    "src/transformers/models/foo/modeling_foo.py\n"
                    "src/transformers/models/foo/modular_foo.py\n"
                    "src/transformers/models/foo/configuration_foo.py\n"
                    "docs/source/en/index.md\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff", "--cached"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "ls-files"], returncode=0, stdout="", stderr=""),
        ]
        changed_files = mlinter.get_changed_modeling_files("origin/main")
        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/foo/modeling_foo.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
                Path("src/transformers/models/foo/configuration_foo.py"),
            },
        )

    @patch("mlinter.mlinter.subprocess.run")
    def test_get_changed_modeling_files_includes_uncommitted_worktree_changes(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout="src/transformers/models/helium/modeling_helium.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "diff", "--cached"],
                returncode=0,
                stdout="src/transformers/models/foo/modular_foo.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "ls-files"],
                returncode=0,
                stdout=("src/transformers/models/bar/modeling_bar.py\ndocs/source/en/index.md\n"),
                stderr="",
            ),
        ]

        changed_files = mlinter.get_changed_modeling_files("origin/main")

        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/helium/modeling_helium.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
                Path("src/transformers/models/bar/modeling_bar.py"),
            },
        )

    # --- TRF015: _tied_weights_keys requires tie_word_embeddings in config ---

    def test_trf015_valid_config_has_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_missing_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)

    def test_trf015_empty_tied_weights_keys_no_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_inherited_config_no_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(LlamaConfig):
    model_type = "foo"
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_main_composite_requires_top_level_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True

class FooConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForConditionalGeneration(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)

    def test_trf015_config_file_suffix_matching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo_audio.py").write_text(
                """
class FooAudioConfig(PreTrainedConfig):
    sample_rate: int = 16000
""",
                encoding="utf-8",
            )
            (model_dir / "configuration_foo_text.py").write_text(
                """
class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooTextPreTrainedModel(PreTrainedModel):
    pass

class FooTextForCausalLM(FooTextPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo_text.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_only_checks_target_config_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooVisionConfig(FooConfig):
    model_type = "foo_vision"

class FooConfig(PreTrainedConfig):
    model_type = "foo"
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForConditionalGeneration(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)
            self.assertNotIn("FooVisionConfig", trf015[0].message)

    def test_trf015_resolves_inherited_config_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}
    hidden_size: int = 768

class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooTextConfig

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_resolves_inherited_config_annotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class CompositeConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}

class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class WrapperPreTrainedModel(PreTrainedModel):
    config: CompositeConfig

class FooMainModel(WrapperPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("CompositeConfig", trf015[0].message)

    def test_trf015_cache_invalidated_by_config_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
"""
            modeling_path = model_dir / "modeling_foo.py"
            modeling_path.write_text(modeling_source, encoding="utf-8")

            config_path = model_dir / "configuration_foo.py"
            config_path.write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )
            digest_v1 = mlinter._content_hash(
                modeling_source,
                {mlinter.TRF015},
                mlinter._find_companion_files(modeling_path),
            )

            config_path.write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )
            digest_v2 = mlinter._content_hash(
                modeling_source,
                {mlinter.TRF015},
                mlinter._find_companion_files(modeling_path),
            )

            self.assertNotEqual(digest_v1, digest_v2)

    # --- TRF016: do_* flags must be referenced by overridden preprocess/_preprocess ---

    def test_trf016_flags_dead_do_resize(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, size, **kwargs):
        for image in images:
            image = self.resize(image, size=size)
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_resize", trf016[0].message)
        self.assertIn("FooImageProcessor", trf016[0].message)

    def test_trf016_allows_referenced_flag_in_signature(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, do_resize, size, **kwargs):
        for image in images:
            if do_resize:
                image = self.resize(image, size=size)
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_allows_referenced_flag_in_body_only(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, **kwargs):
        do_resize = kwargs.get("do_resize", True)
        for image in images:
            if do_resize:
                image = self.resize(image)
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_allows_super_kwargs_forwarding(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True
    do_normalize = True

    def _preprocess(self, images, **kwargs):
        return super()._preprocess(images, **kwargs)
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_skips_class_without_preprocess_override(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True
    do_normalize = True
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_skips_non_processor_files(self):
        source = """
class FooModel(PreTrainedModel):
    do_resize = True

    def _preprocess(self, images):
        return images
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_allowlists_do_sample_frames(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_sample_frames = True

    def _preprocess(self, videos, **kwargs):
        return videos
"""
        file_path = Path("src/transformers/models/foo/video_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_flags_multiple_dead_flags(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True

    def _preprocess(self, videos, size, image_mean, image_std, **kwargs):
        for video in videos:
            video = self.resize(video, size=size)
            video = video / 255.0
            video = self.normalize(video, image_mean, image_std)
        return videos
"""
        file_path = Path("src/transformers/models/foo/video_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = sorted(v.message for v in violations if v.rule_id == mlinter.TRF016)
        self.assertEqual(len(trf016), 4)
        self.assertTrue(all("FooVideoProcessor" in m for m in trf016))
        flag_names = {
            flag
            for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb")
            if any(flag in m for m in trf016)
        }
        self.assertEqual(flag_names, {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"})

    def test_trf016_skips_non_bool_do_attribute(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = some_callable()

    def _preprocess(self, images):
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_respects_inline_suppression(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True  # trf-ignore: TRF016

    def _preprocess(self, images, size, **kwargs):
        for image in images:
            image = self.resize(image, size=size)
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])


if __name__ == "__main__":
    unittest.main()
