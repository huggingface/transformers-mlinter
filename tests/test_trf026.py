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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, mlinter, patch


class TRF026Test(RuleTestCase):
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

    def test_trf026_allows_wrapper_around_super_forward(self):
        source = """
class FooNormedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, norm_eps=1e-6):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_norm = FooRMSNorm(eps=norm_eps, with_scale=False)

    def forward(self, input_ids):
        return self.embed_norm(super().forward(input_ids))
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
