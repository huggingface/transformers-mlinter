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


class TRF018Test(RuleTestCase):
    # --- TRF018: _init_weights overrides should call super ---

    def test_trf018_flags_missing_super_call(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source)
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
        trf018 = self._run(mlinter.TRF018, source)
        self.assertEqual(trf018, [])

    def test_trf018_allows_unbound_pretrained_model_call_in_modular(self):
        source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source, file_name="modular_foo.py")
        self.assertEqual(trf018, [])

    def test_trf018_allows_unbound_pretrained_model_module_arg_in_modular(self):
        source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source, file_name="modular_foo.py")
        self.assertEqual(trf018, [])

    def test_trf018_does_not_skip_unbound_pretrained_model_call_in_non_modular(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source)
        self.assertEqual(len(trf018), 1)

    def test_trf018_allows_attribute_error_sentinel_in_modular(self):
        source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        raise AttributeError("Not needed")
"""
        trf018 = self._run(mlinter.TRF018, source, file_name="modular_foo.py")
        self.assertEqual(trf018, [])

    def test_trf018_does_not_skip_attribute_error_in_non_modular(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        raise AttributeError("Not needed")
"""
        trf018 = self._run(mlinter.TRF018, source)
        self.assertEqual(len(trf018), 1)

    def test_trf018_respects_inline_suppression(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    # trf-ignore: TRF018
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source)
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
        trf018 = self._run(mlinter.TRF018, source)
        self.assertEqual(trf018, [])

    def test_trf018_skips_non_pretrained_classes(self):
        source = """
class FooHelper:
    def _init_weights(self, module):
        if isinstance(module, FooCustomLayer):
            module.gate.data.zero_()
"""
        trf018 = self._run(mlinter.TRF018, source)
        self.assertEqual(trf018, [])
