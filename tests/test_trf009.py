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


from mlinter import trf009 as _trf009_mod
from tests.rule_test_utils import Path, RuleTestCase, mlinter, patch, tempfile


class TRF009Test(RuleTestCase):
    # --- TRF009: cross-model imports (old TRF013) ---

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_flags_cross_model_import_in_modeling_file(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        trf009 = self._run(mlinter.TRF009, source)
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `llama`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_allows_same_model_import_in_modeling_file(self, _mock):
        source = """
from .configuration_foo import FooConfig
from transformers.models.foo.configuration_foo import FooConfig as FooConfigAlias
"""
        trf009 = self._run(mlinter.TRF009, source)
        self.assertEqual(trf009, [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_ignores_modular_files(self, _mock):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        trf009 = self._run(mlinter.TRF009, source, file_name="modular_foo.py")
        self.assertEqual(trf009, [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_flags_cross_model_import_in_configuration_file(self, _mock):
        # Issue #5: the single-file policy covers every file in a model directory, not just the
        # modeling one -- configuration_*.py was importing another model's config unchallenged.
        source = """
from ..llama.configuration_llama import LlamaConfig
"""
        trf009 = self._run(mlinter.TRF009, source, file_name="configuration_foo.py")
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `llama`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "llama", "auto"})
    def test_trf009_flags_cross_model_import_in_every_covered_file_kind(self, _mock):
        source = """
from ..llama.modeling_llama import LlamaAttention
"""
        for file_name in (
            "processing_foo.py",
            "image_processing_foo.py",
            "video_processing_foo.py",
            "feature_extraction_foo.py",
            "tokenization_foo.py",
        ):
            with self.subTest(file_name=file_name):
                self.assertEqual(len(self._run(mlinter.TRF009, source, file_name=file_name)), 1)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "clip", "auto"})
    def test_trf009_flags_import_through_the_public_api(self, _mock):
        # Issue #39: `from transformers import ClipModel` reaches another model's implementation
        # without ever naming its package.
        source = """
from transformers import ClipModel
"""
        with patch("mlinter.trf009._defined_class_names", return_value=frozenset({"ClipModel"})):
            trf009 = self._run(mlinter.TRF009, source)
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `clip`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "clip", "auto"})
    def test_trf009_allows_shared_library_imports_through_the_public_api(self, _mock):
        # `PreTrainedModel` and friends live outside src/transformers/models, so no directory claims
        # them and the import is left alone.
        source = """
from transformers import AutoModel, PreTrainedModel, logging
"""
        with patch("mlinter.trf009._defined_class_names", return_value=frozenset({"ClipModel"})):
            self.assertEqual(self._run(mlinter.TRF009, source), [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "bit", "auto"})
    def test_trf009_public_api_prefix_match_is_confirmed_against_the_directory(self, _mock):
        # `BitsAndBytesConfig` reads like the `bit` model directory but is not defined there, so a
        # prefix match on its own must not be reported.
        source = """
from transformers import BitsAndBytesConfig
"""
        with patch("mlinter.trf009._defined_class_names", return_value=frozenset({"BitModel", "BitConfig"})):
            self.assertEqual(self._run(mlinter.TRF009, source), [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "bit", "bitnet", "auto"})
    def test_trf009_public_api_reports_the_longest_matching_directory(self, _mock):
        source = """
from transformers import BitNetForCausalLM
"""
        defined = {"bitnet": frozenset({"BitNetForCausalLM"}), "bit": frozenset({"BitModel"})}
        with patch("mlinter.trf009._defined_class_names", side_effect=lambda model_dir: defined[model_dir]):
            trf009 = self._run(mlinter.TRF009, source)
        self.assertEqual(len(trf009), 1)
        self.assertIn("`bitnet`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "auto"})
    def test_trf009_allows_own_class_imported_through_the_public_api(self, _mock):
        source = """
from transformers import FooConfig
"""
        with patch("mlinter.trf009._defined_class_names", return_value=frozenset({"FooConfig"})):
            self.assertEqual(self._run(mlinter.TRF009, source), [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "clip", "auto"})
    def test_trf009_public_api_import_is_suppressible(self, _mock):
        source = """
from transformers import ClipModel  # trf-ignore: TRF009
"""
        with patch("mlinter.trf009._defined_class_names", return_value=frozenset({"ClipModel"})):
            self.assertEqual(self._run(mlinter.TRF009, source), [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "clip", "auto"})
    def test_trf009_flags_relative_import_that_names_the_models_package(self, _mock):
        source = """
from ...models.clip.modeling_clip import CLIPAttention
"""
        trf009 = self._run(mlinter.TRF009, source)
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `clip`", trf009[0].message)

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "auto", "timm_wrapper"})
    def test_trf009_allows_shared_entry_point_packages(self, _mock):
        # `auto` holds the mappings a composite model resolves its sub-models with, and
        # `timm_wrapper` is the adapter that exposes any timm backbone as a transformers model.
        # Both are the shared way in, not another model's implementation.
        source = """
from ..auto.configuration_auto import AutoConfig
from ..timm_wrapper import TimmWrapperConfig
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig as Alias
"""
        self.assertEqual(self._run(mlinter.TRF009, source, file_name="configuration_foo.py"), [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"auto", "encoder_decoder"})
    def test_trf009_skips_files_in_the_auto_package(self, _mock):
        # Naming every model's classes is what the auto package is for, so its own files are not
        # one model reaching into another.
        source = """
from ..encoder_decoder import EncoderDecoderConfig
"""
        file_path = Path("src/transformers/models/auto/tokenization_auto.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF009})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF009], [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "clip"})
    def test_trf009_directory_scan_is_cached_per_path_not_per_name(self, _mock):
        # The scan of a model directory is memoized for the life of the process, and MODELS_ROOT is
        # not a constant: tests point it at temporary trees. Two roots that both hold a `clip`
        # directory must not answer each other's lookups.
        self.addCleanup(_trf009_mod._reset_defined_class_names)
        source = "from transformers import ClipModel\n"
        results = []
        with tempfile.TemporaryDirectory() as defines_it, tempfile.TemporaryDirectory() as does_not:
            for root, body in ((defines_it, "class ClipModel: ...\n"), (does_not, "class Unrelated: ...\n")):
                (Path(root) / "clip").mkdir()
                (Path(root) / "clip" / "modeling_clip.py").write_text(body, encoding="utf-8")
            for root in (defines_it, does_not):
                with patch.object(_trf009_mod, "MODELS_ROOT", Path(root)):
                    results.append(self._run(mlinter.TRF009, source))

        self.assertEqual(len(results[0]), 1)
        self.assertIn("`clip`", results[0][0].message)
        # The second root's `clip` does not define ClipModel, so nothing is reported there.
        self.assertEqual(results[1], [])

    @patch("mlinter.trf009._known_model_dirs", return_value={"foo", "parakeet", "auto"})
    def test_trf009_flags_cross_model_import_in_generation_file(self, _mock):
        # A generation_*.py in a model directory is implementation code like any other: importing
        # another model's generation mixin couples the two exactly as tightly.
        source = """
from ...models.parakeet.generation_parakeet import ParakeetGenerationMixin
"""
        trf009 = self._run(mlinter.TRF009, source, file_name="generation_foo.py")
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `parakeet`", trf009[0].message)
