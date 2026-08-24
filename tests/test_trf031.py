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

    def test_trf031_flags_plain_dataclass_with_one_required_field_and_optional_fields(self):
        source = """
@dataclass
class FooStructureOutput:
    positions: torch.Tensor
    confidence: Optional[torch.Tensor] = None
"""
        violations = self._run(mlinter.TRF031, source)
        self.assertEqual(len(violations), 1)

    def test_trf031_allows_internal_argument_bundle_dataclass(self):
        source = """
@dataclass
class FooAtomInputs:
    input_tokens: torch.Tensor
    input_mask: torch.Tensor
    residue_index: torch.Tensor
    atom_positions: torch.Tensor
    atom_mask: torch.Tensor
    trunk_embeddings: torch.Tensor
    pair_embeddings: torch.Tensor
"""
        self.assertEqual(self._run(mlinter.TRF031, source, file_name="modular_foo.py"), [])

    def test_trf031_allows_dimension_info_dataclass(self):
        source = """
@dataclass
class DimensionInfo:
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_blocks: int
    block_size: int
    padding_length: int
    padded_sequence_length: int
    num_heads: int
    head_dim: int
"""
        self.assertEqual(self._run(mlinter.TRF031, source), [])

    def test_trf031_allows_exactly_two_required_fields(self):
        source = """
@dataclass
class FooInputs:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
"""
        self.assertEqual(self._run(mlinter.TRF031, source), [])

    def test_trf031_ignores_class_vars_when_counting_required_fields(self):
        source = """
@dataclass
class FooStructureOutput:
    registry: ClassVar[dict]
    positions: torch.Tensor
"""
        violations = self._run(mlinter.TRF031, source)
        self.assertEqual(len(violations), 1)

    def test_trf031_treats_field_without_default_as_required(self):
        source = """
@dataclass
class FooInputs:
    hidden_states: torch.Tensor = field()
    attention_mask: torch.Tensor = field()
"""
        self.assertEqual(self._run(mlinter.TRF031, source), [])

    def test_trf031_treats_field_with_default_as_optional(self):
        source = """
@dataclass
class FooStructureOutput:
    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor] = field(default=None)
    past_key_values: list[torch.Tensor] = field(default_factory=list)
"""
        violations = self._run(mlinter.TRF031, source)
        self.assertEqual(len(violations), 1)

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
