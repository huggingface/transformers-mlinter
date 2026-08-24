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


class TRF003Test(RuleTestCase):
    # --- TRF003: capture_output enforcement (reworked old TRF005) ---

    def test_trf003_flags_old_return_dict_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        if not return_dict:
            return (x,)
        return x
"""
        trf003 = self._run(mlinter.TRF003, source)
        self.assertEqual(len(trf003), 1)
        self.assertIn("old return_dict branching pattern", trf003[0].message)

    def test_trf003_allows_no_return_dict_arg(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x):
        return x
"""
        trf003 = self._run(mlinter.TRF003, source)
        self.assertEqual(trf003, [])

    def test_trf003_allows_return_dict_without_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        return x
"""
        trf003 = self._run(mlinter.TRF003, source)
        self.assertEqual(trf003, [])
