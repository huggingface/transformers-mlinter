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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf061_mod, date, mlinter, patch


class TRF061Test(RuleTestCase):
    # --- TRF061: no local inheritance in modular files ---

    def _run_trf061(self, source):
        return self._run(mlinter.TRF061, source, file_name="modular_foo.py")

    def test_trf061_flags_subclass_of_local_class(self):
        source = """
from ..llama.modeling_llama import LlamaDecoderLayer

class FooDecoderLayer(LlamaDecoderLayer):
    pass

class FooSparseDecoderLayer(FooDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = FooSparseMoeBlock(config)
"""
        violations = self._run_trf061(source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 7)
        self.assertIn("FooSparseDecoderLayer subclasses FooDecoderLayer", violations[0].message)

    def test_trf061_allows_imported_bases(self):
        source = """
from torch import nn
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaMLP

class FooMLP(LlamaMLP):
    pass

class FooDecoderLayer(LlamaDecoderLayer):
    pass

class FooNorm(nn.Module):
    pass
"""
        self.assertEqual(self._run_trf061(source), [])

    def test_trf061_flags_empty_subclass_and_generic_base(self):
        source = """
from ..llama.modeling_llama import LlamaRotaryEmbedding

class FooThinkerRotaryEmbedding(LlamaRotaryEmbedding):
    pass

class FooTalkerRotaryEmbedding(FooThinkerRotaryEmbedding):
    pass

class FooOtherRotaryEmbedding(FooThinkerRotaryEmbedding[int]):
    pass
"""
        violations = self._run_trf061(source)
        self.assertEqual([v.line_number for v in violations], [7, 10])

    def test_trf061_allows_pretrained_model_and_its_subclasses(self):
        source = """
from ..llama.modeling_llama import LlamaPreTrainedModel

class FooPreTrainedModel(LlamaPreTrainedModel):
    pass

class FooPreTrainedModelForConditionalGeneration(FooPreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    pass

class FooThinkerForConditionalGeneration(FooPreTrainedModelForConditionalGeneration):
    pass

class FooFrameLevelModel(FooModel):
    pass
"""
        self.assertEqual(self._run_trf061(source), [])

    def test_trf061_allows_backward_compat_alias(self):
        source = """
import warnings
from ..llama.modeling_llama import LlamaMLP

class FooMLP(LlamaMLP):
    pass

class FooOldMLP(FooMLP):
    \"\"\"Deprecated alias for `FooMLP`.\"\"\"

    def __init__(self, *args, **kwargs):
        logger.warning_once("`FooOldMLP` is deprecated; please use `FooMLP` instead.")
        super().__init__(*args, **kwargs)

class FooOlderMLP(FooMLP):
    def __init__(self, *args, **kwargs):
        warnings.warn("deprecated", FutureWarning)
        super().__init__(*args, **kwargs)

class FooOldestMLP(FooMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
"""
        self.assertEqual(self._run_trf061(source), [])

    def test_trf061_flags_alias_with_extra_members(self):
        source = """
from ..llama.modeling_llama import LlamaMLP

class FooMLP(LlamaMLP):
    pass

class FooOldMLP(FooMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = None

class FooOlderMLP(FooMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
"""
        violations = self._run_trf061(source)
        self.assertEqual([v.line_number for v in violations], [7, 12])

    def test_trf061_ignores_non_modular_files(self):
        source = """
class FooMLP(nn.Module):
    pass

class FooOtherMLP(FooMLP):
    pass
"""
        self.assertEqual(self._run(mlinter.TRF061, source, file_name="modeling_foo.py"), [])

    def test_trf061_respects_suppression(self):
        source = """
class FooMLP(nn.Module):
    pass

# trf-ignore: TRF061
class FooOtherMLP(FooMLP):
    pass
"""
        self.assertEqual(self._run_trf061(source), [])

    def test_trf061_exempts_models_before_cutoff(self):
        source = """
class FooMLP(nn.Module):
    pass

class FooOtherMLP(FooMLP):
    pass
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2023, 1, 1)):
            with patch.object(_trf061_mod, "CUTOFF_DATE", "2026-06-20"):
                violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF061})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF061], [])
