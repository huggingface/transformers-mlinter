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


class TRF014Test(RuleTestCase):
    # --- TRF014: trust_remote_code should never be used in native model integrations ---

    def test_trf014_flags_keyword_argument(self):
        source = """
model = AutoModel.from_pretrained("foo", trust_remote_code=True)
"""
        trf014 = self._run(mlinter.TRF014, source)
        self.assertEqual(len(trf014), 1)
        self.assertIn("trust_remote_code", trf014[0].message)

    def test_trf014_allows_calls_without_trust_remote_code(self):
        source = """
model = AutoModel.from_pretrained("foo", torch_dtype="auto")
"""
        trf014 = self._run(mlinter.TRF014, source)
        self.assertEqual(trf014, [])
