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


from tests.rule_test_utils import Path, RuleTestCase, _trf038_mod, mlinter, patch, tempfile


class TRF038Test(RuleTestCase):
    # --- TRF038: every modeling-family file needs a matching test file ---

    def check_trf038(self, file_name: str, tests_root: Path | None = None, source: str | None = None):
        file_path = Path("src/transformers/models/foo") / file_name
        source = "class FooModel: ...\n" if source is None else source
        with patch.object(_trf038_mod, "TESTS_ROOT", tests_root or Path("/nonexistent/tests/models")):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF038})
        return [v for v in violations if v.rule_id == mlinter.TRF038]

    def test_trf038_flags_missing_test_file(self):
        violations = self.check_trf038("modeling_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("tests/models/foo/test_modeling_foo.py", violations[0].message)

    def test_trf038_allows_existing_test_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tests_root = Path(tmp_dir) / "tests" / "models"
            (tests_root / "foo").mkdir(parents=True)
            (tests_root / "foo" / "test_modeling_foo.py").write_text("class FooModelTest: ...\n", encoding="utf-8")
            self.assertEqual(self.check_trf038("modeling_foo.py", tests_root=tests_root), [])

    def test_trf038_maps_image_processing_files(self):
        violations = self.check_trf038("image_processing_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_image_processing_foo.py", violations[0].message)

    def test_trf038_ignores_configuration_files(self):
        # Config classes are conventionally covered by ConfigTester inside test_modeling_*.py,
        # so configuration_*.py does not need a standalone test file.
        self.assertEqual(self.check_trf038("configuration_foo.py"), [])

    def test_trf038_preserves_composite_name_directory_suffix(self):
        # modeling_foo_text.py -> test_modeling_foo_text.py, not test_modeling_text.py.
        violations = self.check_trf038("modeling_foo_text.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_modeling_foo_text.py", violations[0].message)

    def test_trf038_has_no_suppression_escape_hatch(self):
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        source = "class FooModel: ...  # trf-ignore: TRF038\n"
        with patch.object(_trf038_mod, "TESTS_ROOT", Path("/nonexistent/tests/models")):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF038})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF038]), 1)

    def test_trf038_modular_infers_single_category_from_class_names(self):
        source = """
class FooConfig(PretrainedConfig):
    pass

class FooModel(FooPreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    pass
"""
        violations = self.check_trf038("modular_foo.py", source=source)
        self.assertEqual(len(violations), 1)
        self.assertIn("test_modeling_foo.py", violations[0].message)

    def test_trf038_modular_infers_multiple_categories_from_class_names(self):
        source = """
class FooConfig(PretrainedConfig):
    pass

class FooModel(FooPreTrainedModel):
    pass

class FooImageProcessorFast(BaseImageProcessorFast):
    pass

class FooProcessor(ProcessorMixin):
    pass
"""
        violations = self.check_trf038("modular_foo.py", source=source)
        messages = {v.message for v in violations}
        self.assertEqual(len(violations), 3)
        self.assertTrue(any("test_modeling_foo.py" in m for m in messages))
        self.assertTrue(any("test_image_processing_foo.py" in m for m in messages))
        self.assertTrue(any("test_processing_foo.py" in m for m in messages))
        # ImageProcessorFast must resolve to the shared image-processing test file, not a
        # nonexistent `test_image_processing_fast_foo.py`.
        self.assertFalse(any("fast" in m.lower() for m in messages))

    def test_trf038_modular_reports_one_violation_per_missing_category(self):
        source = """
class FooModel(FooPreTrainedModel):
    pass

class FooVideoProcessor(BaseVideoProcessor):
    pass
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tests_root = Path(tmp_dir) / "tests" / "models"
            (tests_root / "foo").mkdir(parents=True)
            (tests_root / "foo" / "test_modeling_foo.py").write_text("class FooModelTest: ...\n", encoding="utf-8")
            violations = self.check_trf038("modular_foo.py", source=source, tests_root=tests_root)

        self.assertEqual(len(violations), 1)
        self.assertIn("test_video_processing_foo.py", violations[0].message)

    def test_trf038_maps_tokenization_files(self):
        violations = self.check_trf038("tokenization_foo.py", source="class FooTokenizer: ...\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_tokenization_foo.py", violations[0].message)

    def test_trf038_message_asks_for_the_test_the_missing_file_kind_needs(self):
        # A model is exercised on a dummy config; a tokenizer needs something to tokenize against,
        # so the advice has to follow the kind of test file that is missing.
        self.assertIn("dummy config", self.check_trf038("modeling_foo.py")[0].message)
        tokenizer_message = self.check_trf038("tokenization_foo.py", source="class FooTokenizer: ...\n")[0].message
        self.assertIn("hand-written vocabulary", tokenizer_message)
        self.assertNotIn("dummy config", tokenizer_message)

    def test_trf038_fast_tokenizer_shares_the_slow_test_file(self):
        # transformers ships no test_tokenization_*_fast.py: the fast tokenizer is exercised by the
        # same test file as its slow counterpart, so an existing one satisfies both files.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tests_root = Path(tmp_dir) / "tests" / "models"
            (tests_root / "foo").mkdir(parents=True)
            (tests_root / "foo" / "test_tokenization_foo.py").write_text(
                "class FooTokenizationTest: ...\n", encoding="utf-8"
            )
            self.assertEqual(
                self.check_trf038(
                    "tokenization_foo_fast.py",
                    tests_root=tests_root,
                    source="class FooTokenizerFast: ...\n",
                ),
                [],
            )

    def test_trf038_fast_tokenizer_reports_the_slow_test_file_when_missing(self):
        violations = self.check_trf038("tokenization_foo_fast.py", source="class FooTokenizerFast: ...\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("test_tokenization_foo.py", violations[0].message)
        # ...and not a nonexistent `test_tokenization_foo_fast.py`.
        self.assertNotIn("test_tokenization_foo_fast.py", violations[0].message)

    def test_trf038_ignores_tokenization_helper_modules(self):
        # roformer/tokenization_utils.py holds a pre-tokenizer helper, not a tokenizer of its own.
        self.assertEqual(self.check_trf038("tokenization_utils.py", source="class JiebaPreTokenizer: ...\n"), [])
        self.assertEqual(self.check_trf038("tokenization_utils_base.py", source="class Helper: ...\n"), [])

    def test_trf038_modular_infers_tokenization_from_class_names(self):
        source = """
class FooTokenizer(PreTrainedTokenizer):
    pass

class FooTokenizerFast(PreTrainedTokenizerFast):
    pass

class FooConfig(PretrainedConfig):
    pass
"""
        violations = self.check_trf038("modular_foo.py", source=source)
        # Both tokenizer classes map to the one shared test file, so it is reported once.
        self.assertEqual(len(violations), 1)
        self.assertIn("test_tokenization_foo.py", violations[0].message)
