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


from tests.rule_test_utils import Path, RuleTestCase, _trf022_mod, mlinter, patch, tempfile


class TRF022Test(RuleTestCase):
    # --- TRF022: _no_split_modules entries must name existing classes ---

    def _trf022_violations(self, file_path, source):
        with patch.object(_trf022_mod, "_MODEL_DIR_CLASS_NAMES", {}):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF022})
        return [v for v in violations if v.rule_id == mlinter.TRF022]

    def test_trf022_accepts_locally_defined_class(self):
        source = """
class FooDecoderLayer(nn.Module):
    pass


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_flags_unknown_module_name(self):
        source = """
class FooDecoderLayer(nn.Module):
    pass


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooDecoderLayer", "FooVisionAttention"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        trf022 = self._trf022_violations(file_path, source)
        self.assertEqual(len(trf022), 1)
        self.assertIn("FooVisionAttention", trf022[0].message)
        self.assertIn("FooPreTrainedModel", trf022[0].message)
        self.assertEqual(trf022[0].line_number, 7)

    def test_trf022_accepts_imported_class(self):
        source = """
from ..bar.modeling_bar import BarResidualUnit


class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["BarResidualUnit"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_accepts_class_defined_in_sibling_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "vision.py").write_text(
                "class FooVisionEncoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooVisionEncoderLayer"]
"""
            modeling_path = model_dir / "modeling_foo.py"
            modeling_path.write_text(source, encoding="utf-8")
            self.assertEqual(self._trf022_violations(modeling_path, source), [])

    def test_trf022_model_dir_index_is_shared_across_modeling_files(self):
        # The per-directory class index is cached by directory, so it must stay correct no matter
        # which modeling file of that directory populated it first.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            first_source = """
class FooTextDecoderLayer(nn.Module):
    pass


class FooTextPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooTextDecoderLayer"]
"""
            second_source = """
class FooAudioPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooTextDecoderLayer"]
"""
            first_path = model_dir / "modeling_foo_text.py"
            second_path = model_dir / "modeling_foo_audio.py"
            first_path.write_text(first_source, encoding="utf-8")
            second_path.write_text(second_source, encoding="utf-8")

            with patch.object(_trf022_mod, "_MODEL_DIR_CLASS_NAMES", {}):
                first = mlinter.analyze_file(first_path, first_source, enabled_rules={mlinter.TRF022})
                second = mlinter.analyze_file(second_path, second_source, enabled_rules={mlinter.TRF022})
            self.assertEqual([v for v in first if v.rule_id == mlinter.TRF022], [])
            self.assertEqual([v for v in second if v.rule_id == mlinter.TRF022], [])

    def test_trf022_resolves_modular_names_against_generated_modeling_file(self):
        # A modular file inherits `FooDecoderLayer` implicitly, so the name only appears in the
        # generated modeling file. That file is a sibling, so the name must still resolve.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_foo.py").write_text(
                "class FooDecoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    _no_split_modules = ["FooDecoderLayer"]
"""
            modular_path = model_dir / "modular_foo.py"
            modular_path.write_text(source, encoding="utf-8")
            self.assertEqual(self._trf022_violations(modular_path, source), [])

    def test_trf022_flags_unknown_module_name_in_modular_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_foo.py").write_text(
                "class FooDecoderLayer(nn.Module):\n    pass\n", encoding="utf-8"
            )
            source = """
class FooPreTrainedModel(LlamaPreTrainedModel):
    _no_split_modules = ["BarDecoderLayer"]
"""
            modular_path = model_dir / "modular_foo.py"
            modular_path.write_text(source, encoding="utf-8")
            violations = self._trf022_violations(modular_path, source)
            self.assertEqual(len(violations), 1)
            self.assertIn("BarDecoderLayer", violations[0].message)
            self.assertEqual(violations[0].line_number, 3)

    def test_trf022_accepts_parametrized_class_created_at_runtime(self):
        # `torch.nn.utils.parametrize` names its runtime subclasses `Parametrized<cls>`, so no
        # source file defines them.
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["ParametrizedConv1d"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_flags_bare_parametrized_entry(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["Parametrized"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = self._trf022_violations(file_path, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("Parametrized", violations[0].message)

    def test_trf022_accepts_timm_wrapper_class_from_another_model_directory(self):
        # A timm backbone is built from third-party classes, so the wrapper is the smallest unit a
        # timm-backed model can name -- even though it lives in the `timm_wrapper` directory.
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = ["FooEncoderLayer", "TimmWrapperForImageClassification"]


class FooEncoderLayer(nn.Module):
    pass
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_skips_non_model_files(self):
        source = """
class FooConfig(PreTrainedConfig):
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/configuration_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_ignores_none_and_malformed_values(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = None


class BarPreTrainedModel(PreTrainedModel):
    _no_split_modules = []


class BazPreTrainedModel(PreTrainedModel):
    _no_split_modules = SOME_CONSTANT
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])

    def test_trf022_respects_suppression_comment(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    _no_split_modules = [
        # trf-ignore: TRF022
        "FooVisionAttention",
    ]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(self._trf022_violations(file_path, source), [])
