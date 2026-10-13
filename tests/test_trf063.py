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


class TRF063Test(RuleTestCase):
    # --- TRF063: _init_weights must use the transformers initialization module ---

    def test_trf063_flags_nn_init(self):
        source = """
from torch import nn

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        nn.init.constant_(module.gamma, 1.0)
"""
        trf063 = self._run(mlinter.TRF063, source)
        self.assertEqual(len(trf063), 1)
        self.assertEqual(trf063[0].line_number, 7)
        self.assertIn("`torch.nn.init.constant_`", trf063[0].message)
        self.assertIn("`init.constant_`", trf063[0].message)

    def test_trf063_flags_aliased_torch_nn(self):
        source = """
import torch
import torch.nn as nn

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        torch.nn.init.zeros_(module.bias)
        nn.init.ones_(module.scale)
"""
        trf063 = self._run(mlinter.TRF063, source)
        self.assertEqual([v.line_number for v in trf063], [7, 8])

    def test_trf063_flags_init_imported_from_torch(self):
        # `from torch.nn import init` binds the same name as the transformers module.
        source = """
from torch.nn import init

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        init.normal_(module.weight, mean=0.0, std=0.02)
"""
        trf063 = self._run(mlinter.TRF063, source, file_name="modular_foo.py")
        self.assertEqual(len(trf063), 1)
        self.assertIn("`torch.nn.init.normal_`", trf063[0].message)

    def test_trf063_flags_function_imported_from_torch_init(self):
        source = """
from torch.nn.init import trunc_normal_ as tn

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        tn(module.weight, std=0.02)
"""
        trf063 = self._run(mlinter.TRF063, source)
        self.assertEqual(len(trf063), 1)
        self.assertIn("`torch.nn.init.trunc_normal_`", trf063[0].message)

    def test_trf063_allows_transformers_init(self):
        source = """
import torch
from torch import nn

from ... import initialization as init

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            init.zeros_(module.bias)
"""
        self.assertEqual(self._run(mlinter.TRF063, source), [])

    def test_trf063_allows_private_torch_init_helpers(self):
        source = """
import math
import torch

from ... import initialization as init

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(module.bias, -bound, bound)
"""
        self.assertEqual(self._run(mlinter.TRF063, source), [])

    def test_trf063_ignores_torch_init_outside_init_weights(self):
        source = """
from torch import nn

class FooModel(FooPreTrainedModel):
    def _init_pair_state(self, reference):
        state = torch.empty_like(reference)
        nn.init.trunc_normal_(state, std=0.02)
        return state
"""
        self.assertEqual(self._run(mlinter.TRF063, source), [])

    def test_trf063_respects_suppression(self):
        source = """
from torch import nn

class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        # trf-ignore: TRF063
        nn.init.constant_(module.gamma, 1.0)
"""
        self.assertEqual(self._run(mlinter.TRF063, source), [])
