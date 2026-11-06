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


class TRF065Test(RuleTestCase):
    # --- TRF065: processors must not subclass the shared modality kwargs ---

    def _trf065(self, source: str, file_name: str = "processing_foo.py") -> list:
        return self._run(mlinter.TRF065, source, file_name=file_name)

    def test_flags_modality_kwargs_subclasses_in_processing(self):
        source = """
class FooImagesKwargs(ImagesKwargs, total=False):
    num_crops: int

class FooTextKwargs(TextKwargs, total=False):
    pass

class FooAudioKwargs(AudioKwargs, total=False):
    pass

class FooVideosKwargs(processing_utils.VideosKwargs, total=False):
    pass
"""
        violations = self._trf065(source)
        self.assertEqual([v.line_number for v in violations], [2, 5, 8, 11])
        self.assertIn("`FooImagesKwargs` subclasses `ImagesKwargs`", violations[0].message)

    def test_flags_image_processor_naming_in_processing(self):
        # Processing files hold processors only, so the image processor naming is no escape hatch.
        source = """
class FooImageProcessorKwargs(ImagesKwargs, total=False):
    pass
"""
        self.assertEqual(len(self._trf065(source)), 1)

    def test_allows_deprecated_shims(self):
        source = """
class DeprecatedFooTextKwargs(TextKwargs, total=False):
    add_eos_token: bool
"""
        self.assertEqual(self._trf065(source), [])

    def test_flags_subclass_of_other_model_modality_kwargs(self):
        source = """
from ..bar.processing_bar import BarImagesKwargs

class FooImagesKwargs(BarImagesKwargs, total=False):
    pass
"""
        self.assertEqual(len(self._trf065(source, file_name="modular_foo.py")), 1)

    def test_modular_allows_image_and_video_processor_kwargs(self):
        source = """
class FooImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int

class FooVideoProcessorKwargs(VideosKwargs, total=False):
    pass

class FooVideoProcessorInitKwargs(VideosKwargs, total=False):
    pass

class FooAudioProcessorKwargs(AudioKwargs, total=False):
    pass

class DeprecatedFooImagesKwargs(ImagesKwargs, total=False):
    pass
"""
        self.assertEqual(self._trf065(source, file_name="modular_foo.py"), [])

    def test_modular_flags_processor_level_names(self):
        source = """
class FooImagesKwargs(ImagesKwargs, total=False):
    pass

class FooTextKwargs(TextKwargs, total=False):
    pass

class FooImageProcessorKwargs(VideosKwargs, total=False):
    pass
"""
        violations = self._trf065(source, file_name="modular_foo.py")
        self.assertEqual([v.line_number for v in violations], [2, 5, 8])
        self.assertIn("name it `<Model>ImageProcessorKwargs`", violations[0].message)
        # `TextKwargs` has no image/video processor counterpart to suggest.
        self.assertNotIn("name it", violations[1].message)

    def test_ignores_image_processing_files(self):
        source = """
class FooImagesKwargs(ImagesKwargs, total=False):
    pass
"""
        self.assertEqual(self._trf065(source, file_name="image_processing_foo.py"), [])

    def test_suppression(self):
        source = """
class FooAudioKwargs(AudioKwargs, total=False):  # trf-ignore: TRF065
    pass
"""
        self.assertEqual(self._trf065(source), [])
