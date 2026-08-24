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

"""TRF038: every modeling-family source file must have a matching test file under tests/models/."""

import ast
import re
from pathlib import Path

from ._helpers import TESTS_ROOT, Violation, _model_dir_name


RULE_ID = ""  # Set by discovery

# Maps a single-purpose source-file prefix to the prefix its test file is expected to use.
# `configuration_*.py` is intentionally absent: config classes are tested with `ConfigTester
# inside the `test_modeling_*.py` file
# `modular_*.py` is also absent: we need to check content of modular to map to necessary tests
_TEST_PREFIX_BY_SOURCE_PREFIX = {
    "modeling_": "test_modeling_",
    "processing_": "test_processing_",
    "image_processing_pil_": "test_image_processing_",
    "image_processing_": "test_image_processing_",
    "video_processing_": "test_video_processing_",
    "feature_extraction_": "test_feature_extraction_",
    "tokenization_": "test_tokenization_",
}

# What the cheapest test that satisfies the rule looks like, which differs by file kind: a model can
# be exercised on a dummy config, while a tokenizer needs something to tokenize against.
_MINIMAL_TEST_HINT = "a minimal test built on a dummy config and randomly initialized weights"
_MINIMAL_TOKENIZER_TEST_HINT = "a minimal test built on a small hand-written vocabulary"

# Tokenizer helper modules that sit in a model directory without defining a tokenizer of their own
# (e.g. `roformer/tokenization_utils.py`, which holds a Jieba pre-tokenizer). They are exercised
# through the tokenizer that uses them, so they own no test file.
_TOKENIZATION_HELPER_STEM = "tokenization_utils"

# Used to infer which tests are needed for modular's content
_CLASS_SUFFIX_TEST_PREFIX = (
    ("ImageProcessorFast", "test_image_processing_"),
    ("ImageProcessor", "test_image_processing_"),
    ("VideoProcessor", "test_video_processing_"),
    ("FeatureExtractor", "test_feature_extraction_"),
    ("TokenizerFast", "test_tokenization_"),
    ("Tokenizer", "test_tokenization_"),
    ("Processor", "test_processing_"),
    ("PreTrainedModel", "test_modeling_"),
    ("Model", "test_modeling_"),
)

# Task-head modeling classes (`XxxForCausalLM`, `XxxForConditionalGeneration`, ...) don't share a
# common suffix, but they do share the `For<Task>` infix, which is not used by any other category.
_MODELING_INFIX_RE = re.compile(r"For[A-Z]")


def _class_test_prefix(class_name: str) -> str | None:
    """Infer which test file a class inside a modular file should be covered by, from its name.

    Returns None for config classes (covered by `ConfigTester` inside `test_modeling_*.py`) and
    for any class whose name doesn't match a known convention (e.g. an internal `XxxAttention` or
    `XxxRotaryEmbedding` helper), since those aren't independently test-file-worthy.
    """
    if class_name.endswith("Config"):
        return None

    for suffix, test_prefix in _CLASS_SUFFIX_TEST_PREFIX:
        if class_name.endswith(suffix):
            return test_prefix

    if _MODELING_INFIX_RE.search(class_name):
        return "test_modeling_"

    return None


def _expected_test_file(file_path: Path) -> Path | None:
    """Return the test file *file_path* must be covered by, for single-purpose source files."""
    model_dir = _model_dir_name(file_path)
    if model_dir is None:
        return None

    stem = file_path.stem
    match = next(
        (
            (source_prefix, test_prefix)
            for source_prefix, test_prefix in _TEST_PREFIX_BY_SOURCE_PREFIX.items()
            if stem.startswith(source_prefix)
        ),
        None,
    )
    if match is None:
        return None
    source_prefix, test_prefix = match

    suffix = stem[len(source_prefix) :]
    if source_prefix == "tokenization_":
        if stem == _TOKENIZATION_HELPER_STEM or stem.startswith(f"{_TOKENIZATION_HELPER_STEM}_"):
            return None
        # A `tokenization_<name>_fast.py` file is covered by the same `test_tokenization_<name>.py`
        # as its slow counterpart -- transformers ships no `test_tokenization_*_fast.py` file -- so
        # the `_fast` marker is dropped before the expected path is built.
        suffix = suffix.removesuffix("_fast")
    return TESTS_ROOT / model_dir / f"{test_prefix}{suffix}.py"


def _expected_test_files_modular(tree: ast.Module, file_path: Path) -> list[Path]:
    """Return every test file a modular_*.py file must be covered by.

    A modular file can define modeling, processing, image/video-processor and config classes side
    by side, so the filename alone doesn't say which test files it needs. Instead, every class is
    classified by its name (see `_class_test_prefix`), and one expected test file is returned per
    distinct category found.
    """
    model_dir = _model_dir_name(file_path)
    if model_dir is None:
        return []

    suffix = file_path.stem.removeprefix("modular_")

    test_prefixes = {
        test_prefix
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and (test_prefix := _class_test_prefix(node.name)) is not None
    }

    return sorted(TESTS_ROOT / model_dir / f"{test_prefix}{suffix}.py" for test_prefix in test_prefixes)


def _minimal_test_hint(test_file: Path) -> str:
    """How to satisfy the rule cheaply, phrased for the kind of test file that is missing."""
    if test_file.name.startswith("test_tokenization_"):
        return _MINIMAL_TOKENIZER_TEST_HINT
    return _MINIMAL_TEST_HINT


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if file_path.name.startswith("modular_"):
        expected_test_files = _expected_test_files_modular(tree, file_path)
    else:
        expected_test_file = _expected_test_file(file_path)
        expected_test_files = [expected_test_file] if expected_test_file is not None else []

    missing_test_files = [test_file for test_file in expected_test_files if not test_file.exists()]
    if not missing_test_files:
        return []

    # Deliberately no `# trf-ignore: TRF038` support: every one of these files can be exercised on
    # something made up -- a dummy config for a model, a hand-written vocabulary for a tokenizer --
    # so there is no legitimate per-file exemption. Models that genuinely cannot add a test yet go in
    # `allowlist_models` in rules.toml, which is visible in review instead of buried in the diff.
    return [
        Violation(
            file_path=file_path,
            line_number=1,
            message=(
                f"{RULE_ID}: no test file found at `{test_file}` for `{file_path}`. "
                f"Add one, even {_minimal_test_hint(test_file)} if no real checkpoint exists yet."
            ),
        )
        for test_file in missing_test_files
    ]
