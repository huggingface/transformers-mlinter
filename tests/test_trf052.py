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


class TRF052Test(RuleTestCase):
    # --- TRF052: no *_ATTENTION_CLASSES dispatch dicts ---

    def test_trf052_flags_attention_classes_dict(self):
        source = """
FOO_ATTENTION_CLASSES = {
    "eager": FooAttention,
    "sdpa": FooSdpaAttention,
}
"""
        trf052 = self._run(mlinter.TRF052, source)
        self.assertEqual(len(trf052), 1)
        self.assertIn("FOO_ATTENTION_CLASSES", trf052[0].message)

    def test_trf052_ignores_other_module_constants(self):
        source = """
FOO_PRETRAINED_MODEL_ARCHIVE_LIST = ["foo-base"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF052})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF052], [])
