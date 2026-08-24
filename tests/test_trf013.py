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


class TRF013Test(RuleTestCase):
    # --- TRF013: __init__ should call self.post_init ---

    def test_trf013_flags_missing_post_init(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
"""
        trf013 = self._run(mlinter.TRF013, source)
        self.assertEqual(len(trf013), 1)
        self.assertIn("does not call `self.post_init`", trf013[0].message)

    def test_trf013_allows_post_init(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
        self.post_init()
"""
        trf013 = self._run(mlinter.TRF013, source)
        self.assertEqual(trf013, [])
