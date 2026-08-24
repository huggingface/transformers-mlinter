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


class TRF039Test(RuleTestCase):
    # --- TRF039: imports guarded by is_*_available() must actually be used ---

    def test_trf039_flags_unused_guarded_import(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo():
    return 1
"""
        violations = self._run(mlinter.TRF039, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`Image`", violations[0].message)

    def test_trf039_flags_unused_guarded_import_after_ruff(self):
        source = """
if is_vision_available():
    pass

def foo():
    return 1
"""
        violations = self._run(mlinter.TRF039, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("Availability guard has an empty body", violations[0].message)

    def test_trf039_allows_used_guarded_import(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo(x):
    return Image.open(x)
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_allows_usage_in_string_type_hint(self):
        source = """
if is_vision_available():
    from PIL import Image

def foo(x: "Image.Image"):
    return x
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_respects_suppression_comment(self):
        source = """
if is_vision_available():
    from PIL import Image  # trf-ignore: TRF039

def foo():
    return 1
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_handles_combined_availability_guard(self):
        source = """
if is_torch_available():
    if is_torchvision_available():
        import torchvision
        from PIL import Image

def foo():
    return torchvision.nn.functional.resize(image)
"""
        violations = self._run(mlinter.TRF039, source, file_name="image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`Image`", violations[0].message)

    def test_trf039_ignores_imports_outside_availability_guard(self):
        source = """
if some_other_condition():
    from PIL import Image

def foo():
    return 1
"""
        self.assertEqual(self._run(mlinter.TRF039, source, file_name="image_processing_foo.py"), [])

    def test_trf039_handles_aliased_and_dotted_imports(self):
        source = """
if is_torch_available():
    import torch.nn as nn

def foo():
    return 1
"""
        violations = self._run(mlinter.TRF039, source, file_name="modeling_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("`nn`", violations[0].message)
