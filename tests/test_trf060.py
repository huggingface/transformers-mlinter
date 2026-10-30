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

import tempfile

from tests.rule_test_utils import Path, RuleTestCase, mlinter


class TRF060Test(RuleTestCase):
    # --- trf060: `_no_split_modules` must only list classes that exist ---

    def _trf060(self, source: str, file_name: str = "modeling_foo.py") -> list:
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        return [v for v in violations if v.rule_id == mlinter.TRF060]

    def _trf060_modular(self, modular_source: str, generated_source: str | None, generated_name="modeling_foo.py"):
        # Modular files are checked through the modeling file they generate, which is read from disk.
        with tempfile.TemporaryDirectory() as tmp:
            if generated_source is not None:
                (Path(tmp) / generated_name).write_text(generated_source, encoding="utf-8")
            violations = mlinter.analyze_file(
                Path(tmp) / "modular_foo.py", modular_source, enabled_rules={mlinter.TRF060}
            )
        return [v for v in violations if v.rule_id == mlinter.TRF060]

    GENERATED_HEADER = "# This file was automatically generated from src/transformers/models/foo/modular_foo.py.\n"

    def test_trf060_flags_missing_class(self):
        source = """
class FooTextDecoderLayer(nn.Module):
    pass

class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
        trf060 = self._trf060(source)
        self.assertEqual(len(trf060), 1)
        self.assertIn("FooPreTrainedModel", trf060[0].message)
        self.assertIn("`FooDecoderLayer`", trf060[0].message)
        self.assertIn("`FooTextDecoderLayer`", trf060[0].message)  # suggestion

    def test_trf060_flags_each_missing_class(self):
        source = """
class FooLayer(nn.Module):
    pass

class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooLayer", "MissingA", "MissingB"]
"""
        self.assertEqual(len(self._trf060(source)), 2)

    def test_trf060_flags_annotated_assignment(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules: list[str] = ["Missing"]
"""
        self.assertEqual(len(self._trf060(source)), 1)

    def test_trf060_allows_existing_classes(self):
        source = """
class FooDecoderLayer(nn.Module):
    pass

class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_allows_imported_classes(self):
        source = """
from ..clip.modeling_clip import CLIPEncoderLayer

class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["CLIPEncoderLayer"]
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_allows_torch_parametrized_classes(self):
        # torch.nn.utils.parametrize creates `Parametrized<Name>` classes at runtime.
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["ParametrizedConv1d"]
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_skips_non_literal_values(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooLayer"]

class FooModel(FooPreTrainedModel):
    _no_split_modules = FooPreTrainedModel._no_split_modules
"""
        # FooLayer is missing in FooPreTrainedModel, but FooModel's non-literal value is not reported again.
        self.assertEqual(len(self._trf060(source)), 1)

    def test_trf060_allows_none_declaration(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = None
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_skips_non_modeling_files(self):
        source = """
class FooConfig(PreTrainedConfig):
    _no_split_modules = ["Missing"]
"""
        self.assertEqual(self._trf060(source, file_name="configuration_foo.py"), [])

    def test_trf060_respects_suppression_comment(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    # trf-ignore: TRF060
    _no_split_modules = ["TimmWrapperForImageClassification"]
"""
        self.assertEqual(self._trf060(source), [])

    def test_trf060_modular_checks_generated_file(self):
        # The modular class inherits `_no_split_modules`; the converter renames the prefix to a class that
        # does not exist in the generated file.
        modular = """
class FooTextDecoderLayer(BarDecoderLayer):
    pass

class FooModel(BarModel):
    pass
"""
        generated = (
            self.GENERATED_HEADER
            + """
class FooTextDecoderLayer(nn.Module):
    pass

class FooModel(FooPreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
        )
        trf060 = self._trf060_modular(modular, generated)
        self.assertEqual(len(trf060), 1)
        self.assertIn("FooModel", trf060[0].message)
        self.assertIn("`FooDecoderLayer`", trf060[0].message)
        self.assertIn("`FooTextDecoderLayer`", trf060[0].message)

    def test_trf060_modular_allows_valid_generated_file(self):
        modular = """
class FooModel(BarModel):
    pass
"""
        generated = (
            self.GENERATED_HEADER
            + """
class FooTextDecoderLayer(nn.Module):
    pass

class FooModel(FooPreTrainedModel):
    _no_split_modules = ["FooTextDecoderLayer"]
"""
        )
        self.assertEqual(self._trf060_modular(modular, generated), [])

    def test_trf060_modular_reports_on_modular_class_so_it_can_be_suppressed(self):
        modular = """
class FooModel(BarModel):  # trf-ignore: TRF060
    pass
"""
        generated = (
            self.GENERATED_HEADER
            + """
class FooModel(FooPreTrainedModel):  # trf-ignore: TRF060
    _no_split_modules = ["Missing"]
"""
        )
        self.assertEqual(self._trf060_modular(modular, generated), [])

    def test_trf060_modular_without_generated_file_is_skipped(self):
        modular = """
class FooModel(BarModel):
    pass
"""
        self.assertEqual(self._trf060_modular(modular, None), [])

    def test_trf060_modular_ignores_files_generated_from_another_modular(self):
        modular = """
class FooModel(BarModel):
    pass
"""
        generated = """# This file was automatically generated from src/transformers/models/foo/modular_other.py.
class FooModel(FooPreTrainedModel):
    _no_split_modules = ["Missing"]
"""
        self.assertEqual(self._trf060_modular(modular, generated), [])
