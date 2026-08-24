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


class TRF047Test(RuleTestCase):
    # --- TRF047: image/video processors are stateless ---

    def test_trf047_flags_self_assignment_in_preprocess(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    def _preprocess(self, images, **kwargs):
        self.original_sizes = [image.shape for image in images]
        return images
"""
        trf047 = self._run(mlinter.TRF047, source, file_name="image_processing_foo.py")
        self.assertEqual(len(trf047), 1)
        self.assertIn("FooImageProcessor._preprocess writes self.original_sizes", trf047[0].message)

    def test_trf047_flags_post_process_methods(self):
        source = """
class FooVideoProcessor(BaseVideoProcessor):
    def post_process_detection(self, outputs):
        self.pred = outputs
        return outputs
"""
        file_path = Path("src/transformers/models/foo/video_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF047})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF047]), 1)

    def test_trf047_allows_init_assignments(self):
        source = """
class FooImageProcessor(BaseImageProcessor):
    def __init__(self, do_resize=True, **kwargs):
        super().__init__(**kwargs)
        self.do_resize = do_resize

    def _preprocess(self, images, **kwargs):
        original_sizes = [image.shape for image in images]
        return images
"""
        file_path = Path("src/transformers/models/foo/image_processing_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF047})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF047], [])

    def test_trf047_ignores_modeling_files(self):
        source = """
class FooModel(FooPreTrainedModel):
    def preprocess(self, images):
        self.original_sizes = images
        return images
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF047})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF047], [])
