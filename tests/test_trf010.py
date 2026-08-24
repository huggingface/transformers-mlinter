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


class TRF010Test(RuleTestCase):
    # --- TRF010: strict config decorator ---

    def test_trf010_allows_direct_config_with_strict(self):
        source = """
from huggingface_hub.dataclasses import strict

@strict
class FooConfig(PretrainedConfig):
    pass
"""
        trf010 = self._run(mlinter.TRF010, source, file_name="configuration_foo.py")
        self.assertEqual(trf010, [])

    def test_trf010_flags_missing_strict_on_direct_config(self):
        source = """
class FooConfig(PretrainedConfig):
    pass
"""
        trf010 = self._run(mlinter.TRF010, source, file_name="configuration_foo.py")
        self.assertEqual(len(trf010), 1)
        self.assertIn("missing @strict", trf010[0].message)

    def test_trf010_ignores_non_direct_config_alias_wrappers(self):
        source = """
from huggingface_hub.dataclasses import strict

@strict
class FooConfig(PretrainedConfig):
    pass

class FooCompatConfig(FooConfig):
    pass
"""
        trf010 = self._run(mlinter.TRF010, source, file_name="configuration_foo.py")
        self.assertEqual(trf010, [])
