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


from tests.rule_test_utils import Path, RuleTestCase, mlinter


class TRF058Test(RuleTestCase):
    # --- TRF058: buffers must be nn.Buffer attributes, not register_buffer() calls ---

    def _trf058(self, source: str, file_name: str = "modeling_foo.py") -> list:
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF058})
        return [v for v in violations if v.rule_id == mlinter.TRF058]

    def test_trf058_flags_self_register_buffer(self):
        source = """
class FooRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
"""
        violations = self._trf058(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("self.inv_freq = nn.Buffer(..., persistent=False)", violations[0].message)

    def test_trf058_flags_register_buffer_on_another_module(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        mup_vector = compute_mup_vector(config)
        for layer in self.layers:
            layer.mamba.register_buffer("mup_vector", mup_vector.clone(), persistent=False)
"""
        violations = self._trf058(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("layer.mamba.mup_vector = nn.Buffer(..., persistent=False)", violations[0].message)

    def test_trf058_omits_persistent_when_the_call_does_not_pass_it(self):
        source = """
class FooEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("beta", torch.tensor(config.beta))
"""
        violations = self._trf058(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("self.beta = nn.Buffer(...)", violations[0].message)

    def test_trf058_allows_nn_buffer_assignment(self):
        source = """
class FooRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
"""
        self.assertEqual(self._trf058(source), [])

    def test_trf058_exempts_computed_buffer_names(self):
        # A name built at runtime has no attribute-assignment equivalent.
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        for i, mask in enumerate(config.masks):
            self.register_buffer(f"mask_{i}", mask, persistent=False)
            self.register_buffer(buffer_name, mask, persistent=False)
"""
        self.assertEqual(self._trf058(source), [])

    def test_trf058_ignores_unrelated_register_calls(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.register_parameter("weight", nn.Parameter(torch.zeros(4)))
        self._register_load_state_dict_pre_hook(self.load_hook)
"""
        self.assertEqual(self._trf058(source), [])

    def test_trf058_skips_non_modeling_files(self):
        source = """
class FooEmbeddings(nn.Module):
    def __init__(self, config):
        self.register_buffer("position_ids", torch.arange(4), persistent=False)
"""
        self.assertEqual(self._trf058(source, file_name="configuration_foo.py"), [])

    def test_trf058_checks_modular_files(self):
        source = """
class FooRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.register_buffer("inv_freq", config.inv_freq, persistent=False)
"""
        self.assertEqual(len(self._trf058(source, file_name="modular_foo.py")), 1)

    def test_trf058_respects_suppression_comment(self):
        source = """
class FooEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # trf-ignore: TRF058
        self.register_buffer("position_ids", torch.arange(4), persistent=False)
"""
        self.assertEqual(self._trf058(source), [])
