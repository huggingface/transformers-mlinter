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

from mlinter import trf061
from tests.rule_test_utils import RuleTestCase, _helpers_mod, date, mlinter, patch


class TRF061Test(RuleTestCase):
    def test_rejects_model_specific_types_for_each_modality(self):
        for field, expected in (
            ("text_kwargs", "TextKwargs"),
            ("images_kwargs", "ImagesKwargs"),
            ("audio_kwargs", "AudioKwargs"),
            ("videos_kwargs", "VideosKwargs"),
        ):
            for file_name in ("processing_foo.py", "modular_foo.py"):
                with self.subTest(field=field, file_name=file_name):
                    source = f"class Foo(ProcessingKwargs, total=False):\n    {field}: Molmo2{expected}\n"
                    violations = self._run(mlinter.TRF061, source, file_name)
                    self.assertEqual(len(violations), 1)
                    self.assertEqual(violations[0].line_number, 2)
                    self.assertIn(f"Foo.{field}", violations[0].message)
                    self.assertIn(f"must be typed as `{expected}`", violations[0].message)

    def test_accepts_shared_types_and_omitted_fields(self):
        source = """
class Foo(ProcessingKwargs, total=False):
    text_kwargs: TextKwargs
    images_kwargs: ImagesKwargs
    audio_kwargs: AudioKwargs
    videos_kwargs: VideosKwargs
    custom_option: bool

class Bar(ProcessingKwargs):
    pass
"""
        self.assertEqual(self._run(mlinter.TRF061, source, "processing_foo.py"), [])

    def test_qualified_names_aliases_and_forward_annotations(self):
        source = """
from transformers.processing_utils import ProcessingKwargs as Base, ImagesKwargs as Images

class Foo(Base):
    images_kwargs: "Images"
    text_kwargs: "processing_utils.TextKwargs"
    audio_kwargs: processing_utils.AudioKwargs

class Bar(processing_utils.ProcessingKwargs):
    images_kwargs: "custom.Molmo2ImagesKwargs"
"""
        violations = self._run(mlinter.TRF061, source, "processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("Bar.images_kwargs", violations[0].message)

    def test_model_specific_import_alias_is_not_shared_type(self):
        source = """
from .image_processing_foo import FooImagesKwargs as ImagesKwargs
class Foo(ProcessingKwargs):
    images_kwargs: ImagesKwargs
"""
        self.assertEqual(len(self._run(mlinter.TRF061, source)), 1)

    def test_follows_local_inheritance_and_reports_declaration_once(self):
        source = """
class Base(ProcessingKwargs):
    images_kwargs: FooImagesKwargs
class Middle(Base):
    pass
class Child(Mixin, Middle):
    text_kwargs: FooTextKwargs
"""
        violations = self._run(mlinter.TRF061, source, "modular_foo.py")
        self.assertEqual([v.line_number for v in violations], [3, 7])

    def test_skips_unrelated_classes_and_unresolved_imported_base(self):
        source = """
from ..other.processing_other import OtherProcessorKwargs
class Foo(OtherProcessorKwargs):
    images_kwargs: FooImagesKwargs
class Unrelated(TypedDict):
    text_kwargs: FooTextKwargs
class LooksLikeProcessorKwargs:
    audio_kwargs: FooAudioKwargs
"""
        self.assertEqual(self._run(mlinter.TRF061, source, "modular_foo.py"), [])

    def test_rejects_wrong_modality_and_noncanonical_annotations(self):
        for annotation in (
            "TextKwargs",
            "dict[str, object]",
            "ImagesKwargs | None",
            "'FooImagesKwargs'",
            "'invalid['",
        ):
            with self.subTest(annotation=annotation):
                source = f"class Foo(ProcessingKwargs):\n    images_kwargs: {annotation}\n"
                self.assertEqual(len(self._run(mlinter.TRF061, source)), 1)

    def test_does_not_inspect_method_local_annotations(self):
        source = """
class Foo(ProcessingKwargs):
    def helper(self):
        images_kwargs: FooImagesKwargs
"""
        self.assertEqual(self._run(mlinter.TRF061, source), [])

    def test_respects_class_and_field_suppressions(self):
        source = """
# trf-ignore: TRF061
class Foo(ProcessingKwargs):
    images_kwargs: FooImagesKwargs
    text_kwargs: FooTextKwargs
class Bar(ProcessingKwargs):
    images_kwargs: FooImagesKwargs  # trf-ignore: TRF061
    # trf-ignore: TRF061
    audio_kwargs: FooAudioKwargs
    videos_kwargs: FooVideosKwargs
"""
        violations = self._run(mlinter.TRF061, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("Bar.videos_kwargs", violations[0].message)

    def test_inheritance_cycle_terminates(self):
        source = """
class Foo(Bar):
    images_kwargs: FooImagesKwargs
class Bar(Foo):
    pass
"""
        self.assertEqual(self._run(mlinter.TRF061, source), [])

    def test_contribution_date_cutoff(self):
        source = "class Foo(ProcessingKwargs):\n    images_kwargs: FooImagesKwargs\n"
        for contributed, expected_count in (
            (date(2026, 6, 19), 0),
            (date(2026, 6, 20), 1),
            (date(2026, 6, 21), 1),
            (None, 1),
        ):
            with (
                self.subTest(contributed=contributed),
                patch.object(trf061, "CUTOFF_DATE", "2026-06-20"),
                patch.object(_helpers_mod, "model_contribution_date", return_value=contributed),
            ):
                self.assertEqual(len(self._run(mlinter.TRF061, source, "processing_foo.py")), expected_count)
