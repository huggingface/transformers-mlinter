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
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlinter as public_api
from mlinter import _helpers as _helpers_mod
from mlinter import _version as _version_mod
from mlinter import mlinter
from mlinter import trf011 as _trf011_mod
from mlinter import trf019 as _trf019_mod
from mlinter import trf020 as _trf020_mod
from mlinter import trf022 as _trf022_mod
from mlinter import trf023 as _trf023_mod
from mlinter import trf038 as _trf038_mod


TEST_PP_PLAN_MODULES = {"foo": {"embed_tokens", "final_layer_norm", "layers", "norm"}}

# The header every model file is expected to carry, verbatim from the library.
LICENSE_HEADER = """# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""


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
        self.assertEqual(public_api.TRF017, "TRF017")
        self.assertEqual(public_api.TRF018, "TRF018")
        self.assertEqual(public_api.TRF019, "TRF019")

    def test_package_root_all_lists_supported_api(self):
        self.assertIn("__version__", public_api.__all__)
        self.assertIn("analyze_file", public_api.__all__)
        self.assertIn("collect_class_bases", public_api.__all__)
        self.assertIn("model_dir_name", public_api.__all__)
        self.assertIn("render_rules_reference", public_api.__all__)
        self.assertIn("TRF001", public_api.__all__)
        self.assertIn("TRF015", public_api.__all__)
        self.assertIn("TRF016", public_api.__all__)
        self.assertIn("TRF017", public_api.__all__)
        self.assertIn("TRF018", public_api.__all__)
        self.assertIn("TRF019", public_api.__all__)
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
            version="9.9.9",
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
            self.assertEqual(_version_mod._resolve_version(), "9.9.9+gabcdef1")

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
        source = (
            LICENSE_HEADER
            + """
from collections import OrderedDict

class _LazyConfigMapping(OrderedDict[str, str]):
    pass
"""
        )
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

    def test_trf016_allows_image_do_convert_rgb_handled_by_base_prepare_pipeline(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def _preprocess(self, images, size, **kwargs):
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_allows_image_do_convert_rgb_in_custom_prepare_override(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def _preprocess_image_like_inputs(self, images, do_convert_rgb, **kwargs):
        images = self._prepare_image_like_inputs(images=images, do_convert_rgb=do_convert_rgb)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(trf016, [])

    def test_trf016_flags_image_do_convert_rgb_when_custom_preprocess_drops_flag(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def preprocess(self, images, **kwargs):
        images = self._prepare_image_like_inputs(images=images)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)
        self.assertIn("preprocess()", trf016[0].message)

    def test_trf016_flags_image_do_convert_rgb_when_custom_prepare_override_drops_flag(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def preprocess(self, images, **kwargs):
        return super().preprocess(images, **kwargs)

    def _preprocess_image_like_inputs(self, images, **kwargs):
        images = self._prepare_image_like_inputs(images=images)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)
        self.assertIn("_preprocess_image_like_inputs()", trf016[0].message)

    def test_trf016_still_flags_video_do_convert_rgb_without_reference(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_convert_rgb = True

    def _preprocess(self, videos, do_resize, size, **kwargs):
        return videos
"""
        file_path = Path("src/transformers/models/foo/video_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = [v for v in violations if v.rule_id == mlinter.TRF016]
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)

    def test_trf016_allows_delegating_flag_handling_to_super(self):
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

    # --- TRF017: @auto_docstring must be placed above @dataclass ---

    def test_trf017_flags_dataclass_above_auto_docstring(self):
        source = """
@dataclass
@auto_docstring
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(len(trf017), 1)
        self.assertIn("FooOutput", trf017[0].message)
        self.assertIn("@dataclass listed above @auto_docstring", trf017[0].message)

    def test_trf017_flags_dataclass_above_called_auto_docstring(self):
        source = '''
@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`FooForPreTraining`].
    """
)
class FooForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor = None
'''
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(len(trf017), 1)
        self.assertIn("FooForPreTrainingOutput", trf017[0].message)

    def test_trf017_allows_auto_docstring_above_dataclass(self):
        source = """
@auto_docstring
@dataclass
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(trf017, [])

    def test_trf017_allows_dataclass_only(self):
        source = """
@dataclass
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(trf017, [])

    def test_trf017_allows_auto_docstring_only(self):
        source = """
@auto_docstring
class FooModel(PreTrainedModel):
    pass
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(trf017, [])

    def test_trf017_respects_inline_suppression(self):
        source = """
@dataclass  # trf-ignore: TRF017
@auto_docstring
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF017})
        trf017 = [v for v in violations if v.rule_id == mlinter.TRF017]
        self.assertEqual(trf017, [])

    # --- Generated-file skipping ---

    def test_iter_modeling_files_skips_generated_files(self):
        banner = "# This file was automatically generated from src/transformers/models/foo/modular_foo.py.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            generated = model_dir / "modeling_foo.py"
            generated.write_text(banner + "class FooModel: ...\n", encoding="utf-8")
            handwritten = model_dir / "modeling_bar.py"
            handwritten.write_text("class BarModel: ...\n", encoding="utf-8")
            modular = model_dir / "modular_foo.py"
            modular.write_text("class FooModel: ...\n", encoding="utf-8")

            self.assertTrue(mlinter._is_generated_file(generated))
            self.assertFalse(mlinter._is_generated_file(handwritten))
            self.assertFalse(mlinter._is_generated_file(modular))

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                found = set(mlinter.iter_modeling_files())
            self.assertNotIn(generated, found)
            self.assertIn(handwritten, found)
            self.assertIn(modular, found)

    # --- TRF018: _init_weights overrides should call super ---

    def test_trf018_flags_missing_super_call(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(len(trf018), 1)
        self.assertIn("does not call `super()._init_weights", trf018[0].message)

    def test_trf018_allows_super_call(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    def test_trf018_allows_unbound_pretrained_model_call_in_modular(self):
        source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    def test_trf018_does_not_skip_unbound_pretrained_model_call_in_non_modular(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(len(trf018), 1)

    def test_trf018_allows_attribute_error_sentinel_in_modular(self):
        source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        raise AttributeError("Not needed")
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    def test_trf018_does_not_skip_attribute_error_in_non_modular(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        raise AttributeError("Not needed")
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(len(trf018), 1)

    def test_trf018_respects_inline_suppression(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    # trf-ignore: TRF018
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    def test_trf018_suppression_above_decorator(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    # trf-ignore: TRF018
    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    def test_trf018_skips_non_pretrained_classes(self):
        source = """
class FooHelper:
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF018})
        trf018 = [v for v in violations if v.rule_id == mlinter.TRF018]
        self.assertEqual(trf018, [])

    # --- generated-file filtering ---

    def test_is_generated_file_detects_banner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modeling_foo.py"
            path.write_text(
                "#                This file was automatically generated from foo/modular_foo.py.\n"
                "class FooModel:\n    pass\n",
                encoding="utf-8",
            )
            self.assertTrue(mlinter._is_generated_file(path))

    def test_is_generated_file_false_for_handwritten_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modular_foo.py"
            path.write_text("class FooModel:\n    pass\n", encoding="utf-8")
            self.assertFalse(mlinter._is_generated_file(path))

    def test_is_generated_file_false_for_missing_file(self):
        missing = Path("/nonexistent/modeling_foo.py")
        self.assertFalse(mlinter._is_generated_file(missing))

    def test_is_generated_file_only_reads_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modeling_foo.py"
            # Banner buried past the 1KB head is not treated as a generation marker.
            path.write_text(
                "x = 0\n" * 400 + f"# {mlinter._GENERATED_FILE_MARKER} foo/modular_foo.py\n",
                encoding="utf-8",
            )
            self.assertFalse(mlinter._is_generated_file(path))

    def test_iter_modeling_files_skips_generated_in_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            generated = model_dir / "modeling_foo.py"
            generated.write_text(
                f"# {mlinter._GENERATED_FILE_MARKER} foo/modular_foo.py\nclass FooModel:\n    pass\n",
                encoding="utf-8",
            )
            source = model_dir / "modular_foo.py"
            source.write_text("class FooModel:\n    pass\n", encoding="utf-8")

            yielded = list(mlinter.iter_modeling_files({generated, source}))

            self.assertEqual(yielded, [source])

    def test_iter_modeling_files_skips_generated_when_walking_models_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            generated = model_dir / "modeling_foo.py"
            generated.write_text(
                f"# {mlinter._GENERATED_FILE_MARKER} foo/modular_foo.py\nclass FooModel:\n    pass\n",
                encoding="utf-8",
            )
            source = model_dir / "modular_foo.py"
            source.write_text("class FooModel:\n    pass\n", encoding="utf-8")

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                yielded = set(mlinter.iter_modeling_files())

            self.assertEqual(yielded, {source})

    def test_iter_modeling_files_returns_processing_files(self):
        expected = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            filenames = ["modeling_foo.py", "processing_foo.py", "image_processing_foo.py", "video_processing_foo.py"]
            for name in filenames:
                path = model_dir / name
                path.write_text("import torch", encoding="utf-8")
                expected.add(path)

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                yielded = set(mlinter.iter_modeling_files())

            self.assertEqual(yielded, expected)

    # --- TRF019: ModelNameProcessorKwargs must not define _defaults ---

    def test_trf019_flags_non_empty_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {"return_tensors": "pt"},
    }
    text_kwargs: FooTokenizerKwargs
    images_kwargs: FooImageProcessorKwargs
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_trf019_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)
        self.assertIn("_defaults", trf019[0].message)
        self.assertIn("processor_config.json", trf019[0].message)

    def test_trf019_no_violation_without_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: FooTokenizerKwargs
    images_kwargs: FooImageProcessorKwargs
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_no_violation_with_empty_defaults(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}
    text_kwargs: FooTokenizerKwargs
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_ignores_non_processing_files(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        for file_name in ("image_processing_foo.py", "modeling_foo.py", "configuration_foo.py"):
            file_path = Path(f"src/transformers/models/foo/{file_name}")
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
            trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
            self.assertEqual(len(trf019), 0, f"Expected no violation in {file_name}")

    def test_trf019_ignores_non_processor_kwargs_classes(self):
        source = """
class FooConfig:
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_allowlisted_model_skipped(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.dict(mlinter.TRF_MODEL_DIR_ALLOWLISTS, {mlinter.TRF019: {"foo"}}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_flags_multiple_kwargs_classes(self):
        source = """
class FooTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"truncation": True}}

class FooVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"images_kwargs": {"do_resize": True}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_trf019_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 2)

    def test_trf019_cutoff_exempts_file_committed_before_cutoff(self):
        from datetime import date

        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-10"),
            patch.object(_trf019_mod, "model_contribution_date", return_value=date(2025, 1, 1)),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 0)

    def test_trf019_cutoff_flags_file_committed_on_or_after_cutoff(self):
        from datetime import date

        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-09"),
            patch.object(_trf019_mod, "model_contribution_date", return_value=date(2026, 6, 10)),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)

    def test_trf019_cutoff_flags_file_not_in_git(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with (
            patch.object(_trf019_mod, "CUTOFF_DATE", "2026-06-10"),
            patch.object(_trf019_mod, "model_contribution_date", return_value=None),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)

    def test_trf019_no_cutoff_always_flags(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        with patch.object(_trf019_mod, "CUTOFF_DATE", ""):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF019})
        trf019 = [v for v in violations if v.rule_id == mlinter.TRF019]
        self.assertEqual(len(trf019), 1)

    # --- TRF020: MLA models must isolate the KV LoRA expansion in a dedicated method ---

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_flags_kv_b_proj_applied_in_forward(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        key_states, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return key_states, value_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(len(trf020), 1)
        self.assertIn("self.kv_b_proj", trf020[0].message)

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_allows_expand_kv_method_called_by_forward(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def expand_kv(self, k_nope, k_pe):
        k_nope = self.kv_b_proj(k_nope).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)
        return key_states, value_states

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)
        return key_states, value_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(trf020, [])

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_flags_expansion_method_not_called_by_forward(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def expand_kv(self, k_nope, k_pe):
        k_nope = self.kv_b_proj(k_nope)
        return k_nope, k_pe

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        return compressed_kv
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(len(trf020), 1)
        self.assertIn("dedicated expansion method", trf020[0].message)

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_flags_generic_projection_name(self):
        # The expansion projection is identified by its kv_lora_rank input dim, not only by the
        # conventional `kv_b_proj` name.
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_up_proj = nn.Linear(self.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        key_states = self.latent_up_proj(compressed_kv)
        return key_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(len(trf020), 1)
        self.assertIn("self.latent_up_proj", trf020[0].message)

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_does_not_flag_kv_a_proj_with_mqa_compression(self):
        # kv_a_proj_with_mqa maps hidden_size -> kv_lora_rank + rope; it is the *compression* (input
        # dim is hidden_size), not the expansion, so applying it in forward must not be flagged.
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def expand_kv(self, k_nope, k_pe):
        k_nope = self.kv_b_proj(k_nope)
        return k_nope, k_pe

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)
        return key_states, value_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(trf020, [])

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"bar"})
    def test_trf020_skips_non_mla_models(self):
        # Same anti-pattern, but the model directory's config does not declare kv_lora_rank.
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        key_states = self.kv_b_proj(hidden_states)
        return key_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(trf020, [])

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_respects_suppression_comment(self):
        source = """
# trf-ignore: TRF020
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        key_states = self.kv_b_proj(hidden_states)
        return key_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF020})
        trf020 = [v for v in violations if v.rule_id == mlinter.TRF020]
        self.assertEqual(trf020, [])

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

    # --- TRF022: _no_split_modules entries must name existing classes ---

    def _trf022_violations(self, file_path, source):
        with patch.object(_trf022_mod, "_MODEL_DIR_CLASS_NAMES", {}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF022})
        return [v for v in violations if v.rule_id == mlinter.TRF022]

    def test_trf022_accepts_locally_defined_class(self):
        source = """
class FooDecoderLayer(nn.Module):
    pass


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_flags_unknown_module_name(self):
        source = """
class FooDecoderLayer(nn.Module):
    pass


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer", "FooVisionAttention"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        trf022 = self._trf022_violations(file_path, source)
        self.assertEqual(len(trf022), 1)
        self.assertIn("FooVisionAttention", trf022[0].message)
        self.assertIn("FooPreTrainedModel", trf022[0].message)
        self.assertEqual(trf022[0].line_number, 7)

    def test_trf022_accepts_imported_class(self):
        source = """
from ..bar.modeling_bar import BarResidualUnit


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["BarResidualUnit"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_accepts_class_defined_in_sibling_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "vision.py").write_text(
                "class FooVisionEncoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooVisionEncoderLayer"]
"""
            modeling_path = model_dir / "modeling_foo.py"
            modeling_path.write_text(source, encoding="utf-8")
            self.assertEqual(self._trf022_violations(modeling_path, source), [])

    def test_trf022_model_dir_index_is_shared_across_modeling_files(self):
        # The per-directory class index is cached by directory, so it must stay correct no matter
        # which modeling file of that directory populated it first.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            first_source = """
class FooTextDecoderLayer(nn.Module):
    pass


class FooTextPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooTextDecoderLayer"]
"""
            second_source = """
class FooAudioPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooTextDecoderLayer"]
"""
            first_path = model_dir / "modeling_foo_text.py"
            second_path = model_dir / "modeling_foo_audio.py"
            first_path.write_text(first_source, encoding="utf-8")
            second_path.write_text(second_source, encoding="utf-8")

            with patch.object(_trf022_mod, "_MODEL_DIR_CLASS_NAMES", {}):
                first = mlinter.analyze_file(first_path, first_source, enabled_rules={mlinter.TRF022})
                second = mlinter.analyze_file(second_path, second_source, enabled_rules={mlinter.TRF022})
            self.assertEqual([v for v in first if v.rule_id == mlinter.TRF022], [])
            self.assertEqual([v for v in second if v.rule_id == mlinter.TRF022], [])

    def test_trf022_resolves_modular_names_against_generated_modeling_file(self):
        # A modular file inherits `FooDecoderLayer` implicitly, so the name only appears in the
        # generated modeling file. That file is a sibling, so the name must still resolve.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_foo.py").write_text(
                "class FooDecoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
            modular_path = model_dir / "modular_foo.py"
            modular_path.write_text(source, encoding="utf-8")
            self.assertEqual(self._trf022_violations(modular_path, source), [])

    def test_trf022_flags_unknown_module_name_in_modular_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_foo.py").write_text(
                "class FooDecoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    _no_split_modules = ["BarDecoderLayer"]
"""
            modular_path = model_dir / "modular_foo.py"
            modular_path.write_text(source, encoding="utf-8")
            violations = self._trf022_violations(modular_path, source)
            self.assertEqual(len(violations), 1)
            self.assertIn("BarDecoderLayer", violations[0].message)
            self.assertEqual(violations[0].line_number, 3)

    def test_trf022_accepts_parametrized_class_created_at_runtime(self):
        # `torch.nn.utils.parametrize` names its runtime subclasses `Parametrized<cls>`, so no
        # source file defines them.
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["ParametrizedConv1d"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_flags_bare_parametrized_entry(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["Parametrized"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = self._trf022_violations(file_path, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("Parametrized", violations[0].message)

    def test_trf022_accepts_timm_wrapper_class_from_another_model_directory(self):
        # A timm backbone is built from third-party classes, so the wrapper is the smallest unit a
        # timm-backed model can name -- even though it lives in the `timm_wrapper` directory.
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooEncoderLayer", "TimmWrapperForImageClassification"]


class FooEncoderLayer(nn.Module):
    pass
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_skips_non_model_files(self):
        source = """
class FooConfig(PreTrainedConfig):
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_ignores_none_and_malformed_values(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = None


class BarPreTrainedModel(PreTrainedModel):
    _no_split_modules = []


class BazPreTrainedModel(PreTrainedModel):
    _no_split_modules = SOME_CONSTANT
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_respects_suppression_comment(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = [
        # trf-ignore: TRF022
        "FooVisionAttention",
    ]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

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

    # --- TRF024: layer dimensions must come from the config ---

    def _trf024(self, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF024})
        return [v for v in violations if v.rule_id == mlinter.TRF024]

    def test_trf024_flags_hardcoded_dimensions(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(768, 3072, bias=False)
        self.norm = nn.LayerNorm(3072)
        self.embed = nn.Embedding(32000, config.hidden_size)
"""
        violations = self._trf024(source)
        self.assertEqual(len(violations), 3)
        self.assertIn("768", violations[0].message)
        self.assertIn("nn.Linear", violations[0].message)

    def test_trf024_flags_keyword_dimensions_and_sequence_shapes(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(in_features=config.hidden_size, out_features=4096)
        self.norm = nn.LayerNorm((1024,))
"""
        violations = self._trf024(source)
        self.assertEqual(len(violations), 2)

    def test_trf024_allows_config_values_and_small_literals(self):
        source = """
class FooHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.binary = nn.Linear(config.hidden_size, 2)
        self.patch = nn.Conv2d(3, config.hidden_size, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.group = nn.GroupNorm(32, config.hidden_size)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_ignores_operator_shape_arguments(self):
        source = """
class FooConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=128, padding=64)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_ignores_unrelated_linear_attribute(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = self.registry.Linear(768, 768)
"""
        self.assertEqual(self._trf024(source), [])

    def test_trf024_respects_suppression_and_file_type(self):
        source = """
class FooEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # trf-ignore: TRF024
        self.proj = nn.Linear(768, 3072)
"""
        self.assertEqual(self._trf024(source), [])
        plain = "class FooConfig(PreTrainedConfig):\n    proj = nn.Linear(768, 3072)\n"
        self.assertEqual(self._trf024(plain, file_name="configuration_foo.py"), [])

    # --- TRF025: masks must be built once in the model, not per layer ---

    def _trf025(self, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF025})
        return [v for v in violations if v.rule_id == mlinter.TRF025]

    def test_trf025_flags_mask_creation_inside_a_layer(self):
        source = """
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attention_mask = create_causal_mask(config=self.config, attention_mask=attention_mask)
        return self.self_attn(hidden_states, attention_mask, **kwargs)
"""
        violations = self._trf025(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("create_causal_mask", violations[0].message)
        self.assertIn("FooDecoderLayer", violations[0].message)

    def test_trf025_flags_custom_mask_factory_in_attention(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        mask = create_local_causal_valid_mask(hidden_states)
        return hidden_states + mask
"""
        self.assertEqual(len(self._trf025(source)), 1)

    def test_trf025_allows_mask_creation_in_the_model(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        causal_mask = create_causal_mask(config=self.config, attention_mask=attention_mask)
        for layer in self.layers:
            inputs_embeds = layer(inputs_embeds, causal_mask, **kwargs)
        return inputs_embeds


class FooEncoder(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        attention_mask = create_bidirectional_mask(config=self.config, attention_mask=attention_mask)
        return hidden_states
"""
        self.assertEqual(self._trf025(source), [])

    def test_trf025_allows_layer_consuming_a_prepared_mask(self):
        source = """
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.self_attn(hidden_states, attention_mask, **kwargs)
"""
        self.assertEqual(self._trf025(source), [])

    def test_trf025_respects_suppression_and_file_type(self):
        source = """
# trf-ignore: TRF025
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return create_causal_mask(config=self.config, attention_mask=attention_mask)
"""
        self.assertEqual(self._trf025(source), [])
        plain = """
class FooLayer(nn.Module):
    def forward(self, x):
        return create_causal_mask(x)
"""
        self.assertEqual(self._trf025(plain, file_name="processing_foo.py"), [])

    # --- TRF026: a module that only forwards to its single submodule ---

    def _trf026(self, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF026})
        return [v for v in violations if v.rule_id == mlinter.TRF026]

    def test_trf026_flags_pass_through_wrapper(self):
        source = """
class FooAtomTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
        violations = self._trf026(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("FooAtomTransformer", violations[0].message)
        self.assertIn("self.encoder", violations[0].message)

    def test_trf026_flags_wrapper_with_docstring(self):
        source = '''
class FooValueEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.value_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        """Project the inputs."""
        return self.value_projection(hidden_states)
'''
        self.assertEqual(len(self._trf026(source)), 1)

    def test_trf026_allows_module_doing_extra_work(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.norm(self.encoder(hidden_states, **kwargs))
"""
        self.assertEqual(self._trf026(source), [])

    def test_trf026_allows_extra_statement_or_method(self):
        residual = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        return residual + self.encoder(hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(residual), [])
        extra_method = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def reset(self):
        self.encoder.reset()

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(extra_method), [])

    def test_trf026_exempts_pretrained_model_subclasses(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(source), [])

    def test_trf026_exempts_modular_class_inheriting_an_imported_model(self):
        # `LlamaModel` is a PreTrainedModel, but it is imported, so the base cannot be resolved from
        # this file. Flagging it would report a public model class as a pass-through wrapper.
        source = """
from ..llama.modeling_llama import LlamaModel

class FooModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = FooTextModel(config)

    def forward(self, hidden_states, **kwargs):
        return self.language_model(hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(source, file_name="modular_foo.py"), [])
        # The same holds one level down, through a locally defined subclass of the imported base.
        indirect = (
            source
            + """
class FooDecoder(FooModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
        )
        self.assertEqual(self._trf026(indirect, file_name="modular_foo.py"), [])

    def test_trf026_still_flags_plain_module_bases_in_modular(self):
        # GradientCheckpointingLayer and anything under `torch.nn` are known not to be models, so an
        # unresolvable-base exemption must not swallow these.
        for base in ("nn.Module", "torch.nn.Module", "GradientCheckpointingLayer"):
            source = f"""
class FooAtomTransformer({base}):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
            with self.subTest(base=base):
                self.assertEqual(len(self._trf026(source, file_name="modular_foo.py")), 1)

    def test_trf026_allows_delegating_to_a_different_attribute(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder.layers[0](hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(source), [])

    def test_trf026_respects_suppression(self):
        source = """
# trf-ignore: TRF026
class FooAtomTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FooEncoder(config)

    def forward(self, hidden_states, **kwargs):
        return self.encoder(hidden_states, **kwargs)
"""
        self.assertEqual(self._trf026(source), [])

    # --- TRF027: no bare assert in model files ---

    def _run(self, rule, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={rule})
        return [v for v in violations if v.rule_id == rule]

    def test_trf027_flags_assert(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        assert hidden_states.dim() == 3
        return hidden_states
"""
        violations = self._run(mlinter.TRF027, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("assert", violations[0].message)

    def test_trf027_accepts_raise_and_skips_other_files(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        if hidden_states.dim() != 3:
            raise ValueError("expected 3D")
        return hidden_states
"""
        self.assertEqual(self._run(mlinter.TRF027, source), [])
        assert_source = "def f(x):\n    assert x\n"
        self.assertEqual(self._run(mlinter.TRF027, assert_source, file_name="processing_foo.py"), [])

    def test_trf027_respects_suppression(self):
        source = """
def f(x):
    # trf-ignore: TRF027
    assert x
"""
        self.assertEqual(self._run(mlinter.TRF027, source), [])

    # --- TRF028: Apache license header ---

    def test_trf028_flags_missing_header(self):
        violations = self._run(mlinter.TRF028, '"""PyTorch Foo model."""\n\nimport torch\n')
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 1)
        self.assertIn("missing the license header", violations[0].message)

    def test_trf028_accepts_header(self):
        self.assertEqual(self._run(mlinter.TRF028, LICENSE_HEADER + "\nimport torch\n"), [])

    def test_trf028_accepts_header_below_the_generated_file_banner(self):
        banner = "#  🚨 This file was automatically generated from modular_foo.py.\n#  Do NOT edit it manually.\n"
        self.assertEqual(self._run(mlinter.TRF028, banner + LICENSE_HEADER + "\nimport torch\n"), [])

    def test_trf028_flags_truncated_header(self):
        # bitnet ships this: everything but the closing `limitations under the License.`
        truncated = LICENSE_HEADER.rsplit("\n", 2)[0] + "\n"
        violations = self._run(mlinter.TRF028, truncated + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("limitations under the license.", violations[0].message)

    def test_trf028_flags_header_mangled_by_search_and_replace(self):
        # tvp and bridgetower ship this: a stray `=` inserted before every comma.
        mangled = LICENSE_HEADER.replace(",", "=,")
        violations = self._run(mlinter.TRF028, mangled + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("incomplete license header", violations[0].message)

    def test_trf028_accepts_a_non_apache_license(self):
        # Not every model is Apache 2.0: blip is BSD-3-clause and sapiens2 carries Meta's own
        # license. Both spell out the same warranty paragraph, which is what the rule checks.
        bsd3 = LICENSE_HEADER.replace(
            'Apache License, Version 2.0 (the "License")', 'BSD-3-clause license (the "License")'
        ).replace("http://www.apache.org/licenses/LICENSE-2.0", "https://opensource.org/licenses/BSD-3-Clause")
        sapiens = """# Copyright 2026 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Sapiens2 License. You may obtain a copy of the License at
#
#     https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
        for name, header in (("bsd-3-clause", bsd3), ("sapiens2", sapiens)):
            with self.subTest(license=name):
                self.assertEqual(self._run(mlinter.TRF028, header + "\nimport torch\n"), [])

    def test_trf028_flags_a_header_that_never_names_a_license(self):
        # The warranty paragraph alone, with no "Licensed under the ..." line above it.
        headless = "\n".join(LICENSE_HEADER.splitlines()[4:]) + "\n"
        violations = self._run(mlinter.TRF028, headless + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("does not state what license the file is under", violations[0].message)

    def test_trf028_accepts_any_copyright_attribution(self):
        # The year and the attributed team legitimately vary; only the boilerplate is checked.
        for line in (
            "# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.",
            "# Copyright 2019 The Google AI Language Team Authors.",
        ):
            source = line + "\n" + LICENSE_HEADER.split("\n", 1)[1] + "\nimport torch\n"
            with self.subTest(copyright=line):
                self.assertEqual(self._run(mlinter.TRF028, source), [])

    def test_trf028_respects_suppression(self):
        source = "# trf-ignore: TRF028\n\nimport torch\n"
        self.assertEqual(self._run(mlinter.TRF028, source), [])

    def test_trf028_ignores_unrelated_files(self):
        self.assertEqual(self._run(mlinter.TRF028, "import torch\n", file_name="tokenization_foo.py"), [])

    # --- TRF029: config plus a redundant config field ---

    def test_trf029_flags_redundant_config_arguments(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
"""
        violations = self._run(mlinter.TRF029, source)
        self.assertEqual(len(violations), 1)
        for name in ("embed_dim", "num_heads", "dropout"):
            self.assertIn(name, violations[0].message)

    def test_trf029_accepts_config_only_and_layer_idx(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, layer_idx=None, device=None, **kwargs):
        super().__init__()
        self.embed_dim = config.hidden_size
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])

    def test_trf029_ignores_modules_without_config(self):
        source = """
class FooRotary(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        self.head_dim = head_dim
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])

    # --- TRF030: config attribute chain depth ---

    def test_trf030_flags_three_level_config_chain(self):
        source = """
class FooAtomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = FooLayerNorm(config.diffusion_config.atom_encoder_config.hidden_size)
"""
        violations = self._run(mlinter.TRF030, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("3 levels", violations[0].message)

    def test_trf030_accepts_one_and_two_hops(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.a = config.hidden_size
        self.b = config.text_config.hidden_size
        self.c = self.config.vision_config.num_attention_heads
"""
        self.assertEqual(self._run(mlinter.TRF030, source), [])

    def test_trf030_reports_once_per_line(self):
        source = """
class FooBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.a = config.x.y.hidden_size + config.x.y.intermediate_size
"""
        self.assertEqual(len(self._run(mlinter.TRF030, source)), 1)

    # --- TRF031: dataclass must inherit ModelOutput ---

    def test_trf031_flags_plain_dataclass(self):
        source = """
@dataclass
class FooStructureOutput:
    positions: torch.Tensor
"""
        violations = self._run(mlinter.TRF031, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("ModelOutput", violations[0].message)

    def test_trf031_accepts_model_output_bases(self):
        source = """
@auto_docstring
@dataclass
class FooOutput(ModelOutput):
    logits: torch.Tensor


@dataclass
class FooModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.Tensor


@dataclass
class FooProjectionAttentions(BaseModelOutputWithPooling):
    projection_attentions: torch.Tensor
"""
        self.assertEqual(self._run(mlinter.TRF031, source), [])

    def test_trf031_ignores_non_dataclasses(self):
        self.assertEqual(self._run(mlinter.TRF031, "class FooConfigHolder:\n    x: int\n"), [])

    # --- TRF032: masked fill must use torch.finfo(dtype).min ---

    def test_trf032_flags_magic_negative(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, scores, mask):
        return scores.masked_fill(~mask, -1e9)
"""
        violations = self._run(mlinter.TRF032, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("finfo", violations[0].message)

    def test_trf032_accepts_finfo_and_small_values(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, scores, mask):
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        pad = torch.full_like(scores, -1.0)
        return scores + pad
"""
        self.assertEqual(self._run(mlinter.TRF032, source), [])

    def test_trf032_reports_once_per_call(self):
        source = """
def f(scores, mask):
    return scores.masked_fill(~mask, -1e9).masked_fill(~mask, -1e4)
"""
        self.assertEqual(len(self._run(mlinter.TRF032, source)), 2)

    # --- TRF033: no set_<hyperparameter> mutators ---

    def test_trf033_flags_hyperparameter_setter(self):
        source = """
class FooTriangleAttention(nn.Module):
    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size
"""
        violations = self._run(mlinter.TRF033, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("set_chunk_size", violations[0].message)

    def test_trf033_accepts_sanctioned_setters(self):
        source = """
class FooModel(FooPreTrainedModel):
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.decoder = decoder
"""
        self.assertEqual(self._run(mlinter.TRF033, source), [])

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

    # --- TRF035: no # noqa in model files ---

    def test_trf035_flags_noqa(self):
        source = "from ...modeling_utils import PreTrainedModel  # noqa: F401\n"
        violations = self._run(mlinter.TRF035, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("F401", violations[0].message)

    def test_trf035_flags_bare_noqa_and_skips_other_files(self):
        self.assertEqual(len(self._run(mlinter.TRF035, "import torch  # noqa\n")), 1)
        self.assertEqual(self._run(mlinter.TRF035, "import torch  # noqa\n", file_name="processing_foo.py"), [])

    def test_trf035_respects_suppression(self):
        source = "# trf-ignore: TRF035\nimport torch  # noqa: F401\n"
        self.assertEqual(self._run(mlinter.TRF035, source), [])

    # --- TRF036: no nn.Sequential in modeling ---

    def test_trf036_flags_sequential(self):
        source = """
class FooMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU())
"""
        violations = self._run(mlinter.TRF036, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("nn.Sequential", violations[0].message)

    def test_trf036_accepts_explicit_submodules(self):
        source = """
class FooMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
"""
        self.assertEqual(self._run(mlinter.TRF036, source), [])

    # --- TRF037: no torch.einsum in modeling (opt-in) ---

    def test_trf037_flags_einsum_with_equation(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, q, k):
        return torch.einsum("bqhc,bkhc->bhqk", q, k)
"""
        violations = self._run(mlinter.TRF037, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("bqhc,bkhc->bhqk", violations[0].message)

    def test_trf037_is_disabled_by_default(self):
        self.assertNotIn(mlinter.TRF037, mlinter.DEFAULT_ENABLED_TRF_RULES)

    def test_trf037_accepts_explicit_matmul(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, q, k):
        return q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
"""
        self.assertEqual(self._run(mlinter.TRF037, source), [])

    # --- TRF038: every modeling-family file needs a matching test file ---

    def _trf038(self, file_name: str, tests_root: Path | None = None):
        file_path = Path("src/transformers/models/foo") / file_name
        source = "class FooModel: ...\n"
        with patch.object(_trf038_mod, "TESTS_ROOT", tests_root or Path("/nonexistent/tests/models")):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF038})
        return [v for v in violations if v.rule_id == mlinter.TRF038]

    def test_trf038_flags_missing_test_file(self):
        violations = self._trf038("modeling_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("tests/models/foo/test_modeling_foo.py", violations[0].message)

    def test_trf038_allows_existing_test_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tests_root = Path(tmp_dir) / "tests" / "models"
            (tests_root / "foo").mkdir(parents=True)
            (tests_root / "foo" / "test_modeling_foo.py").write_text("class FooModelTest: ...\n", encoding="utf-8")
            self.assertEqual(self._trf038("modeling_foo.py", tests_root=tests_root), [])

    def test_trf038_maps_modular_files_to_test_modeling(self):
        violations = self._trf038("modular_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_modeling_foo.py", violations[0].message)

    def test_trf038_maps_image_processing_files(self):
        violations = self._trf038("image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_image_processing_foo.py", violations[0].message)

    def test_trf038_ignores_configuration_files(self):
        # Config classes are conventionally covered by ConfigTester inside test_modeling_*.py,
        # so configuration_*.py does not need a standalone test file.
        self.assertEqual(self._trf038("configuration_foo.py"), [])

    def test_trf038_preserves_multi_config_directory_suffix(self):
        # modeling_foo_text.py -> test_modeling_foo_text.py, not test_modeling_text.py.
        violations = self._trf038("modeling_foo_text.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_modeling_foo_text.py", violations[0].message)

    def test_trf038_has_no_suppression_escape_hatch(self):
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        source = "class FooModel: ...  # trf-ignore: TRF038\n"
        with patch.object(_trf038_mod, "TESTS_ROOT", Path("/nonexistent/tests/models")):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF038})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF038]), 1)

    # --- TRF039: imports guarded by is_*_available() must actually be used ---

    def test_trf039_flags_unused_guarded_import(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo():
    return 1
"""
        violations = self._run(mlinter.TRF039, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`Image`", violations[0].message)

    def test_trf039_allows_used_guarded_import(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo(x):
    return Image.open(x)
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_allows_usage_in_string_type_hint(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo(x: "Image.Image"):
    return x
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_respects_suppression_comment(self):
        source = """
if is_vision_available():
    from PIL import Image  # trf-ignore: TRF039

def foo():
    return 1
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_handles_combined_availability_guard(self):
        source = """
if is_vision_available() and is_torch_available():
    import torch
    from PIL import Image

def foo():
    return torch.zeros(1)
"""
        violations = self._run(mlinter.TRF039, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`Image`", violations[0].message)

    def test_trf039_ignores_imports_outside_availability_guard(self):
        source = """
if some_other_condition():
    from PIL import Image

def foo():
    return 1
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_handles_aliased_and_dotted_imports(self):
        source = """
if is_torch_available():
    import torch.nn as nn

def foo():
    return 1
"""
        violations = self._run(mlinter.TRF039, source, file_name="modeling_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`nn`", violations[0].message)


if __name__ == "__main__":
    unittest.main()
