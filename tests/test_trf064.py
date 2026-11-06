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


class TRF064Test(RuleTestCase):
    # --- TRF064: <Model>ProcessorKwargs must not override inherited fields ---

    def _trf064(self, source: str, file_name: str = "processing_foo.py") -> list:
        return self._run(mlinter.TRF064, source, file_name=file_name)

    def test_flags_defaults_in_class_body(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        violations = self._trf064(source)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 3)
        self.assertIn("`FooProcessorKwargs` overrides `_defaults`", violations[0].message)

    def test_flags_modality_default_values(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs = {"padding": False}
    images_kwargs: FooImagesKwargs = {"do_resize": True}
"""
        violations = self._trf064(source)
        self.assertEqual([v.line_number for v in violations], [3, 4])
        self.assertIn("class attribute of the processor", violations[0].message)

    def test_flags_type_annotation_overrides(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults: dict
    images_kwargs: FooImageProcessorKwargs
    text_kwargs: TextKwargs | None
    videos_kwargs: VideosKwargs
    num_crops: int
"""
        violations = self._trf064(source)
        self.assertEqual([v.line_number for v in violations], [3, 4, 5, 6])
        self.assertIn("Keep the type inherited from `ProcessingKwargs`", violations[1].message)

    def test_allows_deprecated_shim_annotation(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: DeprecatedFooTextKwargs
    add_eos_token: bool
"""
        self.assertEqual(self._trf064(source), [])

    def test_flags_subclass_of_other_model_processor_kwargs_in_modular(self):
        source = """
from ..bar.processing_bar import BarProcessorKwargs

class FooProcessorKwargs(BarProcessorKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}
"""
        violations = self._trf064(source, file_name="modular_foo.py")
        self.assertEqual(len(violations), 1)

    def test_allows_defaults_on_processor_class(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    num_crops: int


class FooProcessor(ProcessorMixin):
    valid_processor_kwargs = FooProcessorKwargs
    text_kwargs = {"padding": False}
    images_kwargs = {"do_resize": True}
"""
        self.assertEqual(self._trf064(source), [])

    def test_ignores_sub_processor_kwargs(self):
        source = """
class FooImageProcessorKwargs(BarImageProcessorKwargs, total=False):
    images_kwargs = {}

class FooAudioProcessorKwargs(BarAudioProcessorKwargs, total=False):
    audio_kwargs = {}
"""
        self.assertEqual(self._trf064(source, file_name="modular_foo.py"), [])

    def test_ignores_other_file_types(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}
"""
        self.assertEqual(self._trf064(source, file_name="modeling_foo.py"), [])

    def test_suppression(self):
        source = """
class FooProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}  # trf-ignore: TRF064
"""
        self.assertEqual(self._trf064(source), [])
