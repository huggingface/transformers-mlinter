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


from tests.rule_test_utils import RuleTestCase, _trf020_mod, mlinter, patch


class TRF020Test(RuleTestCase):
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
        trf020 = self._run(mlinter.TRF020, source)
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
        trf020 = self._run(mlinter.TRF020, source)
        self.assertEqual(trf020, [])

    @patch.object(_trf020_mod, "_MLA_MODEL_DIRS", {"foo"})
    def test_trf020_allows_inherited_expansion_method_in_modular(self):
        source = """
class FooAttention(DeepseekV32Attention):
    def __init__(self, config):
        super().__init__(config)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.num_heads * config.v_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)
        return key_states, value_states
"""
        trf020 = self._run(mlinter.TRF020, source, file_name="modular_foo.py")
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
        trf020 = self._run(mlinter.TRF020, source)
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
        trf020 = self._run(mlinter.TRF020, source)
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
        trf020 = self._run(mlinter.TRF020, source)
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
        trf020 = self._run(mlinter.TRF020, source)
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
        trf020 = self._run(mlinter.TRF020, source)
        self.assertEqual(trf020, [])
