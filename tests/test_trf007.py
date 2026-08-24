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


class TRF007Test(RuleTestCase):
    # --- TRF007: post_init order (old TRF011) ---

    def test_trf007_flags_assignment_after_post_init(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        self.proj = None
"""
        trf007 = self._run(mlinter.TRF007, source)
        self.assertEqual(len(trf007), 1)
        self.assertIn("assigns self.* after self.post_init()", trf007[0].message)

    def test_trf007_allows_post_init_at_end(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
        self.post_init()
"""
        trf007 = self._run(mlinter.TRF007, source)
        self.assertEqual(trf007, [])
