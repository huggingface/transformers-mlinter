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


class TRF055Test(RuleTestCase):
    # --- TRF055: `config` must be annotation, not assignment ---

    def _trf055(self, source: str, file_name: str = "modeling_foo.py") -> list:
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF055})
        return [v for v in violations if v.rule_id == mlinter.TRF055]

    def test_trf055_flags_config_assignment(self):
        source = """
class Gemma4PreTrainedModel(PreTrainedModel):
    config_class = Gemma4Config

class Gemma4VisionModel(Gemma4PreTrainedModel):
    config = Gemma4VisionConfig
"""
        trf055 = self._trf055(source)
        self.assertEqual(len(trf055), 1)
        self.assertIn("config: Gemma4VisionConfig", trf055[0].message)
        self.assertIn("Gemma4VisionModel", trf055[0].message)

    def test_trf055_allows_config_annotation(self):
        source = """
class Gemma4PreTrainedModel(PreTrainedModel):
    config_class = Gemma4Config

class Gemma4VisionModel(Gemma4PreTrainedModel):
    config: Gemma4VisionConfig
"""
        self.assertEqual(self._trf055(source), [])

    def test_trf055_allows_no_config_declaration(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig

class FooModel(FooPreTrainedModel):
    pass
"""
        self.assertEqual(self._trf055(source), [])

    def test_trf055_skips_non_pretrained_classes(self):
        # A plain class (not inheriting PreTrainedModel) with config = X should not be flagged.
        source = """
class FooHelper:
    config = FooConfig
"""
        self.assertEqual(self._trf055(source), [])

    def test_trf055_skips_non_modeling_files(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig

class FooModel(FooPreTrainedModel):
    config = FooConfig
"""
        self.assertEqual(self._trf055(source, file_name="configuration_foo.py"), [])

    def test_trf055_respects_suppression_comment(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig

class FooModel(FooPreTrainedModel):
    # trf-ignore: TRF055
    config = FooConfig
"""
        self.assertEqual(self._trf055(source), [])

    def test_trf055_flags_dotted_config_name(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig

class FooModel(FooPreTrainedModel):
    config = configuration_foo.FooConfig
"""
        trf055 = self._trf055(source)
        self.assertEqual(len(trf055), 1)
        self.assertIn("FooConfig", trf055[0].message)
