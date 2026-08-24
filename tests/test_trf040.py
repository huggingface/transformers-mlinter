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


class TRF040Test(RuleTestCase):
    # --- TRF040: @can_return_tuple must not be combined with @capture_outputs ---

    def _trf040_violations(self, file_path: Path, source: str) -> list[str]:
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF040})
        return [v.message for v in violations if v.rule_id == mlinter.TRF040]

    def test_trf040_flags_can_return_tuple_above_capture_outputs(self):
        source = """
class FooPreTrainedModel:
    pass


class FooModel(FooPreTrainedModel):
    @can_return_tuple
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        messages = self._trf040_violations(file_path, source)
        self.assertEqual(len(messages), 1)
        self.assertIn("combines @can_return_tuple with @capture_outputs", messages[0])

    def test_trf040_flags_decorators_in_either_order(self):
        source = """
class FooModel(FooPreTrainedModel):
    @capture_outputs
    @can_return_tuple
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(len(self._trf040_violations(file_path, source)), 1)

    def test_trf040_flags_class_whose_base_lives_in_another_model_file(self):
        # Modular files stack these on encoder classes that do not inherit PreTrainedModel locally.
        source = """
class FooEncoder(BarEncoder):
    @can_return_tuple
    @capture_outputs
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        self.assertEqual(len(self._trf040_violations(file_path, source)), 1)

    def test_trf040_flags_dotted_decorator_names(self):
        source = """
class FooModel(FooPreTrainedModel):
    @utils.can_return_tuple
    @utils.capture_outputs
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(len(self._trf040_violations(file_path, source)), 1)

    def test_trf040_allows_either_decorator_alone(self):
        source = """
class FooModel(FooPreTrainedModel):
    @capture_outputs
    def forward(self, x):
        return x

    @can_return_tuple
    def other(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf040_violations(file_path, source), [])

    def test_trf040_reports_each_offending_method_once(self):
        source = """
class FooModel(FooPreTrainedModel):
    @can_return_tuple
    @capture_outputs
    def forward(self, x):
        return x

    @can_return_tuple
    @capture_outputs
    def get_image_features(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(len(self._trf040_violations(file_path, source)), 2)

    def test_trf040_respects_suppression_comment(self):
        source = """
class FooModel(FooPreTrainedModel):
    # trf-ignore: TRF040
    @can_return_tuple
    @capture_outputs
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf040_violations(file_path, source), [])

    def test_trf040_skips_non_modeling_files(self):
        source = """
class FooProcessor(ProcessorMixin):
    @can_return_tuple
    @capture_outputs
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/processing_foo.py")
        self.assertEqual(self._trf040_violations(file_path, source), [])
