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


from tests.rule_test_utils import Path, RuleTestCase, mlinter, tempfile


TORCHVISION_IMAGE_PROCESSOR = """
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend


class AcmeImageProcessorKwargs(ImagesKwargs, total=False):
    max_soft_tokens: int


class AcmeImageProcessor(TorchvisionBackend):
    pass


__all__ = ["AcmeImageProcessor"]
"""

PIL_IMAGE_PROCESSOR = """
from PIL import Image

from ...image_processing_backends import PilBackend


class AcmeImageProcessorKwargs(ImagesKwargs, total=False):
    max_soft_tokens: int


class AcmeImageProcessorPil(PilBackend):
    pass


__all__ = ["AcmeImageProcessorPil"]
"""

VISION_PROCESSOR = """
from ...processing_utils import ProcessorMixin
from ...utils import is_vision_available
from ...utils.import_utils import requires


if is_vision_available():
    from .{module} import AcmeImageProcessorKwargs


@requires(backends=("vision",))
class AcmeProcessor(ProcessorMixin):
    pass


__all__ = ["AcmeProcessor"]
"""


class TRF064Test(RuleTestCase):
    # --- TRF064: model-local imports must not need undeclared backends ---

    def _run_in_model_dir(self, files: dict[str, str], target: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "src" / "transformers" / "models" / "acme"
            model_dir.mkdir(parents=True)
            for name, source in files.items():
                (model_dir / name).write_text(source, encoding="utf-8")
            target_path = model_dir / target
            violations = mlinter.analyze_file(
                target_path, target_path.read_text(encoding="utf-8"), enabled_rules={mlinter.TRF064}
            )
            return [v for v in violations if v.rule_id == mlinter.TRF064]

    def test_flags_vision_processor_importing_torchvision_image_processor(self):
        violations = self._run_in_model_dir(
            {
                "image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR,
                "processing_acme.py": VISION_PROCESSOR.format(module="image_processing_acme"),
            },
            "processing_acme.py",
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 8)
        self.assertIn("needs torch, torchvision", violations[0].message)
        self.assertIn("image_processing_acme.py:2 `import torch`", violations[0].message)

    def test_accepts_vision_processor_importing_pil_image_processor(self):
        violations = self._run_in_model_dir(
            {
                "image_processing_pil_acme.py": PIL_IMAGE_PROCESSOR,
                "processing_acme.py": VISION_PROCESSOR.format(module="image_processing_pil_acme"),
            },
            "processing_acme.py",
        )
        self.assertEqual(violations, [])

    def test_accepts_import_under_matching_guard(self):
        processor = """
from ...utils import is_torchvision_available, is_vision_available


if is_vision_available() and is_torchvision_available():
    from .image_processing_acme import AcmeImageProcessorKwargs


class AcmeProcessor(ProcessorMixin):
    pass


__all__ = ["AcmeProcessor"]
"""
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR, "processing_acme.py": processor},
            "processing_acme.py",
        )
        self.assertEqual(violations, [])

    def test_accepts_import_covered_by_requires(self):
        processor = VISION_PROCESSOR.format(module="image_processing_acme").replace(
            '@requires(backends=("vision",))', '@requires(backends=("vision", "torch", "torchvision"))'
        )
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR, "processing_acme.py": processor},
            "processing_acme.py",
        )
        self.assertEqual(violations, [])

    def test_weakest_exported_object_decides(self):
        # The lazy module imports the file as soon as `acme_helper`'s (empty) requirements are met.
        processor = VISION_PROCESSOR.format(module="image_processing_acme").replace(
            '@requires(backends=("vision",))', '@requires(backends=("vision", "torch", "torchvision"))'
        )
        processor = processor.replace(
            '__all__ = ["AcmeProcessor"]',
            'def acme_helper():\n    pass\n\n\n__all__ = ["AcmeProcessor", "acme_helper"]',
        )
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR, "processing_acme.py": processor},
            "processing_acme.py",
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("needs torch, torchvision at", violations[0].message)
        self.assertIn("with only vision installed", violations[0].message)

    def test_follows_transitive_imports(self):
        utils = "from .image_processing_acme import AcmeImageProcessorKwargs\n"
        processor = VISION_PROCESSOR.format(module="acme_utils")
        violations = self._run_in_model_dir(
            {
                "image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR,
                "acme_utils.py": utils,
                "processing_acme.py": processor,
            },
            "processing_acme.py",
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("acme_utils.py:1", violations[0].message)
        self.assertIn("-> image_processing_acme.py:2 `import torch`", violations[0].message)

    def test_ignores_guarded_and_optional_imports_in_the_imported_module(self):
        image_processor = """
from ...utils import is_torch_available


if is_torch_available():
    import torch

try:
    import torchvision
except ImportError:
    torchvision = None


class AcmeImageProcessorKwargs(ImagesKwargs, total=False):
    pass
"""
        violations = self._run_in_model_dir(
            {
                "image_processing_acme.py": image_processor,
                "processing_acme.py": VISION_PROCESSOR.format(module="image_processing_acme"),
            },
            "processing_acme.py",
        )
        self.assertEqual(violations, [])

    def test_ignores_type_checking_and_function_imports(self):
        processor = """
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .image_processing_acme import AcmeImageProcessorKwargs


class AcmeProcessor(ProcessorMixin):
    def _image_kwargs(self):
        from .image_processing_acme import AcmeImageProcessorKwargs

        return AcmeImageProcessorKwargs


__all__ = ["AcmeProcessor"]
"""
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR, "processing_acme.py": processor},
            "processing_acme.py",
        )
        self.assertEqual(violations, [])

    def test_modeling_file_may_import_torch_only_siblings(self):
        modeling = "from .image_processing_acme import AcmeImageProcessorKwargs\n"
        image_processor = "import torch\n\n\nclass AcmeImageProcessorKwargs:\n    pass\n"
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": image_processor, "modeling_acme.py": modeling},
            "modeling_acme.py",
        )
        self.assertEqual(violations, [])

    def test_modular_file_reports_generated_processor(self):
        generated = (
            "#           This file was automatically generated from src/transformers/models/acme/modular_acme.py.\n"
            + VISION_PROCESSOR.format(module="image_processing_acme")
        )
        modular = """
from ..base.processing_base import BaseProcessor


class AcmeProcessor(BaseProcessor):
    pass
"""
        violations = self._run_in_model_dir(
            {
                "image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR,
                "processing_acme.py": generated,
                "modular_acme.py": modular,
            },
            "modular_acme.py",
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 5)
        self.assertIn("generated processing_acme.py:9", violations[0].message)

    def test_suppression(self):
        processor = VISION_PROCESSOR.format(module="image_processing_acme").replace(
            "    from .image_processing_acme", "    # trf-ignore: TRF064\n    from .image_processing_acme"
        )
        violations = self._run_in_model_dir(
            {"image_processing_acme.py": TORCHVISION_IMAGE_PROCESSOR, "processing_acme.py": processor},
            "processing_acme.py",
        )
        self.assertEqual(violations, [])
