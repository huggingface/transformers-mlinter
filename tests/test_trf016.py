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


class TRF016Test(RuleTestCase):
    # --- TRF016: do_* flags must be referenced by overridden preprocess/_preprocess ---

    def test_trf016_flags_dead_do_resize(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, size, **kwargs):
        for image in images:
            image = self.resize(image, size=size)
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_resize", trf016[0].message)
        self.assertIn("FooImageProcessor", trf016[0].message)

    def test_trf016_allows_referenced_flag_in_signature(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, do_resize, size, **kwargs):
        for image in images:
            if do_resize:
                image = self.resize(image, size=size)
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_allows_referenced_flag_in_body_only(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True

    def _preprocess(self, images, **kwargs):
        do_resize = kwargs.get("do_resize", True)
        for image in images:
            if do_resize:
                image = self.resize(image)
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_allows_image_do_convert_rgb_handled_by_base_prepare_pipeline(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def _preprocess(self, images, size, **kwargs):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_allows_image_do_convert_rgb_in_custom_prepare_override(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def _preprocess_image_like_inputs(self, images, do_convert_rgb, **kwargs):
        images = self._prepare_image_like_inputs(images=images, do_convert_rgb=do_convert_rgb)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_flags_image_do_convert_rgb_when_custom_preprocess_drops_flag(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def preprocess(self, images, **kwargs):
        images = self._prepare_image_like_inputs(images=images)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)
        self.assertIn("preprocess()", trf016[0].message)

    def test_trf016_flags_image_do_convert_rgb_when_custom_prepare_override_drops_flag(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_convert_rgb = True

    def preprocess(self, images, **kwargs):
        return super().preprocess(images, **kwargs)

    def _preprocess_image_like_inputs(self, images, **kwargs):
        images = self._prepare_image_like_inputs(images=images)
        return self._preprocess(images, **kwargs)

    def _preprocess(self, images, **kwargs):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)
        self.assertIn("_preprocess_image_like_inputs()", trf016[0].message)

    def test_trf016_still_flags_video_do_convert_rgb_without_reference(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_convert_rgb = True

    def _preprocess(self, videos, do_resize, size, **kwargs):
        return videos
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="video_processing_foo.py")
        self.assertEqual(len(trf016), 1)
        self.assertIn("do_convert_rgb", trf016[0].message)

    def test_trf016_allows_delegating_flag_handling_to_super(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True
    do_normalize = True

    def _preprocess(self, images, **kwargs):
        return super()._preprocess(images, **kwargs)
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_skips_class_without_preprocess_override(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True
    do_normalize = True
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_skips_non_processor_files(self):
        source = """
class FooModel(PreTrainedModel):
    do_resize = True

    def _preprocess(self, images):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source)
        self.assertEqual(trf016, [])

    def test_trf016_allowlists_do_sample_frames(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_sample_frames = True

    def _preprocess(self, videos, **kwargs):
        return videos
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="video_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_flags_multiple_dead_flags(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True

    def _preprocess(self, videos, size, image_mean, image_std, **kwargs):
        for video in videos:
            video = self.resize(video, size=size)
            video = video / 255.0
            video = self.normalize(video, image_mean, image_std)
        return videos
"""
        file_path = Path("src/transformers/models/foo/video_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF016})
        trf016 = sorted(v.message for v in violations if v.rule_id == mlinter.TRF016)
        self.assertEqual(len(trf016), 4)
        self.assertTrue(all("FooVideoProcessor" in m for m in trf016))
        flag_names = {
            flag
            for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb")
            if any(flag in m for m in trf016)
        }
        self.assertEqual(flag_names, {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"})

    def test_trf016_skips_non_bool_do_attribute(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = some_callable()

    def _preprocess(self, images):
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])

    def test_trf016_respects_inline_suppression(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    do_resize = True  # trf-ignore: TRF016

    def _preprocess(self, images, size, **kwargs):
        for image in images:
            image = self.resize(image, size=size)
        return images
"""
        trf016 = self._run(mlinter.TRF016, source, file_name="image_processing_foo.py")
        self.assertEqual(trf016, [])
