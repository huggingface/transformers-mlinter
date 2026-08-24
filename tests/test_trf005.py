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


class TRF005Test(RuleTestCase):
    # --- TRF005: _no_split_modules (old TRF008) ---

    def test_trf005_valid_no_split_modules(self):
        source = """
class FooModel:
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(trf005, [])

    def test_trf005_invalid_empty_string(self):
        source = """
class FooModel:
    _no_split_modules = [""]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(len(trf005), 1)

    def test_trf005_allows_attribute_error_sentinel_in_modular(self):
        source = """
class FooModel(BarModel):
    _no_split_modules = AttributeError()
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(trf005, [])

    def test_trf005_rejects_attribute_error_sentinel_in_modeling(self):
        source = """
class FooModel(BarModel):
    _no_split_modules = AttributeError()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF005})
        trf005 = [v for v in violations if v.rule_id == mlinter.TRF005]
        self.assertEqual(len(trf005), 1)
