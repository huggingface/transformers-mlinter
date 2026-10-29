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

from pathlib import Path

from mlinter import trf066
from tests.rule_test_utils import RuleTestCase, _helpers_mod, date, mlinter, patch, tempfile


class TRF066Test(RuleTestCase):
    def check(self, source, file_name="processing_foo.py"):
        return self._run(mlinter.TRF066, source, file_name=file_name)

    def test_class_and_method_overlap(self):
        source = """
@requires(backends=("torch",))
class Processor:
    @requires(backends=("torch", "scipy"))
    def process(self):
        pass
    @requires(backends=["torch"])
    def other(self):
        pass
"""
        violations = self.check(source)
        self.assertEqual([v.line_number for v in violations], [4, 7])
        self.assertIn("Remove `torch` from `@requires(backends=...)`.", violations[0].message)
        self.assertIn("Remove this `@requires(backends=...)` decorator.", violations[1].message)

    def test_additional_method_requirements(self):
        self.assertEqual(
            self.check("""
@requires(backends=("torch",))
class Processor:
    @requires(backends=("scipy",))
    def process(self):
        pass
"""),
            [],
        )

    def test_file_requirements(self):
        for file_name, backend, base in (
            ("modeling_foo.py", "torch", "object"),
            ("generation_foo.py", "torch", "object"),
            ("image_processing_pil_foo.py", "vision", "PilBackend"),
            ("image_processing_foo.py", "torchvision", "TorchvisionBackend"),
            ("video_processing_foo.py", "torch", "BaseVideoProcessor"),
            ("tokenization_foo_fast.py", "tokenizers", "object"),
        ):
            with self.subTest(file_name=file_name):
                self.assertEqual(
                    len(self.check(f'@requires(backends=("{backend}",))\nclass Foo({base}): pass', file_name)), 1
                )

    def test_no_inferred_modular_or_pil_torch_requirements(self):
        source = '@requires(backends=("torch",))\nclass Foo(ImportedProcessor): pass'
        for filename in ("modular_foo.py", "image_processing_pil_foo.py", "processing_foo.py"):
            with self.subTest(filename=filename):
                self.assertEqual(self.check(source, filename), [])

    def test_leading_call_after_docstring(self):
        violations = self.check('''
def process(outputs):
    """Process outputs."""
    # A comment is not an operation.
    requires_backends(process, ["torch"])
    return outputs
''')
        self.assertEqual([v.line_number for v in violations], [5])
        self.assertIn("Remove it and use a `@requires(backends=...)` decorator instead.", violations[0].message)

    def test_leading_call_after_empty_docstring(self):
        source = 'def process():\n    ""\n    requires_backends(process, "torch")'
        self.assertEqual(len(self.check(source)), 1)

    def test_leading_calls(self):
        for call in (
            'requires_backends(self, "torch")',
            'requires_backends(self, ("torch", "scipy"))',
            'requires_backends(self, backends=["torch"])',
            'requires_backends(obj=self, backends=["torch"])',
        ):
            with self.subTest(call=call):
                self.assertEqual(len(self.check(f"class Foo:\n    async def process(self):\n        {call}")), 1)

    def test_conditional_later_and_dynamic_calls(self):
        for body in (
            'if outputs is None:\n        requires_backends(process, "torch")',
            'outputs = None\n    requires_backends(process, "torch")',
            "requires_backends(process, backends)",
            "requires_backends(process, [backend])",
            "requires_backends(process, outputs.backends)",
            'other.requires_backends(process, "torch")',
        ):
            with self.subTest(body=body):
                self.assertEqual(self.check(f"def process(outputs):\n    {body}"), [])

    def test_version_constraints_are_not_discarded(self):
        self.assertEqual(
            self.check(
                """
@requires(backends=("torch>=2.5",))
class Processor:
    @requires(backends=("torch>=2.6",))
    def process(self): pass
""",
                "modeling_foo.py",
            ),
            [],
        )
        self.assertEqual(
            len(
                self.check("""
@requires(backends=("torch>=2.5",))
class Processor:
    @requires(backends=("torch>=2.5",))
    def process(self): pass
""")
            ),
            1,
        )

    def test_dynamic_decorators(self):
        self.assertEqual(
            self.check("""
@requires(backends=BACKENDS)
class Foo:
    @requires(backends=("torch",))
    def process(self): pass
@requires(backends=("torch", *EXTRA))
def process(): pass
"""),
            [],
        )

    def test_nested_scopes_do_not_inherit_class_requirements(self):
        self.assertEqual(
            self.check("""
@requires(backends=("torch",))
class Outer:
    @requires(backends=("torch",))
    class Inner:
        pass
    def process(self):
        @requires(backends=("torch",))
        def nested(): pass
@requires(backends=("torch",))
def unrelated(): pass
"""),
            [],
        )

    def test_import_aliases(self):
        self.assertEqual(
            len(
                self.check("""
from ...utils.import_utils import requires as needs, requires_backends as check_backends
@needs(backends=("torch",))
class Foo:
    @needs(backends=("torch",))
    def process(self):
        check_backends(self, "scipy")
""")
            ),
            2,
        )
        self.assertEqual(
            len(
                self.check(
                    """
import transformers.utils.import_utils as iu
@iu.requires(backends=("torch",))
def process():
    iu.requires_backends(process, "scipy")
""",
                    "modeling_foo.py",
                )
            ),
            2,
        )

    def test_suppressions(self):
        self.assertEqual(
            self.check("""
@requires(backends=("torch",))
class Foo:
    # trf-ignore: TRF066
    @requires(backends=("torch",))
    def process(self):
        requires_backends(self, "scipy")  # trf-ignore: TRF066
"""),
            [],
        )

    def test_checks_any_supplied_file(self):
        source = 'def process():\n    requires_backends(process, "torch")'
        for filename in ("convert_foo.py", "test_modeling_foo.py", "utils.py"):
            with self.subTest(filename=filename):
                self.assertEqual(len(self.check(source, filename)), 1)

    def test_empty_backend_list(self):
        self.assertEqual(len(self.check("@requires(backends=())\ndef process(): pass")), 1)

    def test_cutoff_exempts_older_models(self):
        source = '@requires(backends=("torch",))\ndef process():\n    requires_backends(process, "scipy")'
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2026, 10, 28)):
            with patch.object(trf066, "CUTOFF_DATE", "2026-10-29"):
                self.assertEqual(self.check(source, "modeling_foo.py"), [])

    def test_cutoff_checks_new_and_undated_models(self):
        source = '@requires(backends=("torch",))\ndef process():\n    requires_backends(process, "scipy")'
        for contribution in (date(2026, 10, 29), date(2026, 10, 30), None):
            with self.subTest(contribution=contribution):
                with patch.object(_helpers_mod, "model_contribution_date", return_value=contribution):
                    with patch.object(trf066, "CUTOFF_DATE", "2026-10-29"):
                        self.assertEqual(len(self.check(source, "modeling_foo.py")), 2)

    def test_empty_cutoff_checks_older_models(self):
        source = '@requires(backends=("torch",))\ndef process(): pass'
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2020, 1, 1)):
            with patch.object(trf066, "CUTOFF_DATE", ""):
                self.assertEqual(len(self.check(source, "modeling_foo.py")), 1)

    def test_modular_uses_each_generated_files_requirements(self):
        source = """from ..detr.image_processing_pil_detr import DetrImageProcessorPil
@requires(backends=("vision",))
class FooImageProcessorPil(DetrImageProcessorPil):
    @requires(backends=("vision",))
    def process(self): pass
@requires(backends=("vision",))
class FooModel(ImportedModel): pass
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modular = root / "modular_foo.py"
            header = f"# {_helpers_mod.GENERATED_FILE_MARKER} {modular.name}.\n"
            (root / "image_processing_pil_foo.py").write_text(header + "class FooImageProcessorPil(PilBackend): pass")
            (root / "modeling_foo.py").write_text(header + "class FooModel(PreTrainedModel): pass")
            findings = mlinter.analyze_file(modular, source, enabled_rules={mlinter.TRF066})
        self.assertEqual([v.line_number for v in findings], [2, 4])

    def test_modular_ignores_unrelated_or_invalid_companions(self):
        source = '@requires(backends=("vision",))\nclass Foo(ImportedProcessor): pass'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modular = root / "modular_foo.py"
            header = f"# {_helpers_mod.GENERATED_FILE_MARKER} {modular.name}.\n"
            (root / "image_processing_foo.py").write_text("class Foo(TorchvisionBackend): pass")
            (root / "image_processing_pil_foo.py").write_text(header + "class Foo(")
            self.assertEqual(mlinter.analyze_file(modular, source, enabled_rules={mlinter.TRF066}), [])

    def test_modular_shared_helper_uses_common_requirements(self):
        source = '@requires(backends=("vision",))\ndef helper(): pass'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modular = root / "modular_foo.py"
            header = f"# {_helpers_mod.GENERATED_FILE_MARKER} {modular.name}.\n"
            for name in ("image_processing_pil_foo.py", "modeling_foo.py"):
                (root / name).write_text(header + "def helper(): pass")
            self.assertEqual(mlinter.analyze_file(modular, source, enabled_rules={mlinter.TRF066}), [])
