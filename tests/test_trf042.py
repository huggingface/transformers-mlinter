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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, _trf042_mod, mlinter, patch, tempfile


class TRF042Test(RuleTestCase):
    # --- TRF042: tokenizer tests must use TokenizerTesterMixin ---

    def _trf042(self, source, file_name="test_tokenization_foo.py", model="foo"):
        file_path = Path(f"tests/models/{model}/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF042})
        return [v for v in violations if v.rule_id == mlinter.TRF042]

    def test_trf042_flags_test_without_mixin(self):
        source = """
class FooTokenizationTest(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(tokenizer("hi").input_ids, [1, 2])
"""
        violations = self._trf042(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("TokenizerTesterMixin", violations[0].message)
        self.assertIn("FooTokenizationTest", violations[0].message)

    def test_trf042_accepts_mixin(self):
        source = """
class FooTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FooTokenizer
"""
        self.assertEqual(self._trf042(source), [])

    def test_trf042_accepts_dotted_mixin_reference(self):
        source = """
class FooTokenizationTest(test_tokenization_common.TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FooTokenizer
"""
        self.assertEqual(self._trf042(source), [])

    def test_trf042_ignores_files_with_only_helper_classes(self):
        source = """
class FooTokenizerHelper:
    def build(self):
        return None
"""
        self.assertEqual(self._trf042(source), [])

    def test_trf042_ignores_non_tokenization_tests(self):
        source = """
class FooModelTest(unittest.TestCase):
    pass
"""
        self.assertEqual(self._trf042(source, file_name="test_modeling_foo.py"), [])

    def test_trf042_respects_suppression(self):
        source = """
# trf-ignore: TRF042
class FooTokenizationTest(unittest.TestCase):
    pass
"""
        self.assertEqual(self._trf042(source), [])

    def test_trf042_helper_carrying_the_mixin_does_not_satisfy_the_rule(self):
        """A mixin on a helper does not run under the test runner, so the real test class still owes it."""
        source = """
class FooTokenizerHelper(TokenizerTesterMixin):
    def build(self):
        return None


class FooTokenizationTest(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(tokenizer("hi").input_ids, [1, 2])
"""
        violations = self._trf042(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("FooTokenizationTest", violations[0].message)

    def test_trf042_accepts_a_local_base_carrying_the_mixin(self):
        source = """
class FooTokenizationTestBase(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FooTokenizer


class FooTokenizationTest(FooTokenizationTestBase, unittest.TestCase):
    pass
"""
        self.assertEqual(self._trf042(source), [])

    def test_trf042_follows_inheritance_into_another_models_tokenizer_test(self):
        """`class DistilBertTokenizationTest(test_tokenization_bert.BertTokenizationTest)` is satisfied."""
        with tempfile.TemporaryDirectory() as tmp:
            tests_root = Path(tmp) / "tests" / "models"
            (tests_root / "bert").mkdir(parents=True)
            bert = tests_root / "bert" / "test_tokenization_bert.py"

            derived = """
from ..bert import test_tokenization_bert


class FooTokenizationTest(test_tokenization_bert.BertTokenizationTest, unittest.TestCase):
    tokenizer_class = FooTokenizer
"""
            # Imported by class rather than by module: the other spelling the library uses.
            derived_by_class = """
from ..bert.test_tokenization_bert import BertTokenizationTest


class FooTokenizationTest(BertTokenizationTest, unittest.TestCase):
    tokenizer_class = FooTokenizer
"""
            with patch.object(_trf042_mod, "TESTS_ROOT", tests_root):
                # The base carries the mixin, so the deriving test inherits the whole suite.
                bert.write_text(
                    "class BertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):\n    pass\n",
                    encoding="utf-8",
                )
                self.assertEqual(self._trf042(derived), [])
                self.assertEqual(self._trf042(derived_by_class), [])

                # The base does not carry it either, so nothing in the chain runs the suite.
                bert.write_text("class BertTokenizationTest(unittest.TestCase):\n    pass\n", encoding="utf-8")
                self.assertEqual(len(self._trf042(derived)), 1)

                # A base the tests tree cannot resolve is not assumed to carry the mixin.
                bert.unlink()
                self.assertEqual(len(self._trf042(derived)), 1)

    def test_trf042_resolves_model_name_from_the_tests_tree(self):
        self.assertEqual(_helpers_mod._model_dir_name(Path("tests/models/esmc/test_tokenization_esmc.py")), "esmc")
        self.assertEqual(_helpers_mod._model_dir_name(Path("src/transformers/models/esmc/modeling_esmc.py")), "esmc")
        self.assertIsNone(_helpers_mod._model_dir_name(Path("utils/check_repo.py")))

    def test_discovery_includes_tokenization_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "src/transformers/models/foo"
            tests = root / "tests/models/foo"
            models.mkdir(parents=True)
            tests.mkdir(parents=True)
            (models / "modeling_foo.py").write_text("", encoding="utf-8")
            (tests / "test_tokenization_foo.py").write_text("", encoding="utf-8")
            (tests / "test_modeling_foo.py").write_text("", encoding="utf-8")
            with (
                patch.object(mlinter, "MODELS_ROOT", root / "src/transformers/models"),
                patch.object(mlinter, "TESTS_ROOT", root / "tests/models"),
            ):
                found = {path.name for path in mlinter.iter_modeling_files()}
        self.assertEqual(found, {"modeling_foo.py", "test_tokenization_foo.py"})

    def test_changed_only_candidate_accepts_tokenization_tests(self):
        self.assertTrue(mlinter._is_modeling_candidate(Path("tests/models/foo/test_tokenization_foo.py")))
        self.assertTrue(mlinter._is_modeling_candidate(Path("src/transformers/models/foo/modeling_foo.py")))
        self.assertFalse(mlinter._is_modeling_candidate(Path("tests/models/foo/test_modeling_foo.py")))
