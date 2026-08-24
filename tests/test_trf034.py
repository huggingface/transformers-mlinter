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


class TRF034Test(RuleTestCase):
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

    def test_trf034_allows_imported_modular_layer_base(self):
        source = """
from ..cohere2.modeling_cohere2 import Cohere2DecoderLayer


class FooDecoderLayer(Cohere2DecoderLayer):
    pass


class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([FooDecoderLayer(config) for _ in range(2)])
"""
        self.assertEqual(self._run(mlinter.TRF034, source, file_name="modular_foo.py"), [])

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
