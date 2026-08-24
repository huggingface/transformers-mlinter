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


class TRF008Test(RuleTestCase):
    # --- TRF008: add_start_docstrings usage ---

    def test_trf008_flags_empty_add_start_docstrings(self):
        source = """
@add_start_docstrings("")
class FooPreTrainedModel(PreTrainedModel):
    pass
"""
        trf008 = self._run(mlinter.TRF008, source)
        self.assertEqual(len(trf008), 1)
        self.assertIn("without non-empty docstring arguments", trf008[0].message)

    def test_trf008_allows_non_empty_add_start_docstrings(self):
        source = """
@add_start_docstrings("Foo model.")
class FooPreTrainedModel(PreTrainedModel):
    pass
"""
        trf008 = self._run(mlinter.TRF008, source)
        self.assertEqual(trf008, [])
