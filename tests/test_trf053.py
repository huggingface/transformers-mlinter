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


class TRF053Test(RuleTestCase):
    # --- TRF053: no manual label shifting ---

    def test_trf053_flags_manual_shift(self):
        source = """
class FooForCausalLM(FooPreTrainedModel):
    def forward(self, input_ids, labels=None):
        logits = self.lm_head(self.model(input_ids))
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        return logits
"""
        trf053 = self._run(mlinter.TRF053, source)
        self.assertEqual(len(trf053), 2)
        self.assertIn("self.loss_function owns shifting", trf053[0].message)

    def test_trf053_allows_loss_function_call(self):
        source = """
class FooForCausalLM(FooPreTrainedModel):
    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.lm_head(self.model(input_ids))
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return loss
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF053})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF053], [])

    def test_trf053_allows_received_shift_labels(self):
        source = """
class FooForConditionalGeneration(FooPreTrainedModel):
    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.lm_head(self.model(input_ids))
        loss = None
        if labels is not None:
            shift_labels = kwargs.pop("shift_labels", labels)
            loss = self.loss_function(
                logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=self.config.vocab_size
            )
        return loss
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF053})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF053], [])
