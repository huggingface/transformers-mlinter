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


class TRF031Test(RuleTestCase):
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
