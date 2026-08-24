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


class TRF033Test(RuleTestCase):
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
