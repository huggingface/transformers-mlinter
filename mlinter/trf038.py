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
from pathlib import Path

from ._helpers import Violation, _model_dir_name


RULE_ID = ""  # Set by discovery

TESTS_ROOT = Path("tests/models")

# Maps a source-file prefix to the prefix its test file is expected to use
_TEST_PREFIX_BY_SOURCE_PREFIX = {
    "modeling_": "test_modeling_",
    "processing_": "test_processing_",
    # order matters, PIL should match first if exists so we don't look for `test_image_processing_pil_{ModelName}.py`
    "image_processing_pil_": "test_image_processing_",
    "image_processing_": "test_image_processing_",
    "video_processing_": "test_video_processing_",
    "feature_extraction_": "test_feature_extraction_",
}


def _expected_test_file(file_path: Path) -> Path | None:
    """Return the test file *file_path* must be covered by, or None if this rule doesn't apply."""
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

    model_name = stem[len(source_prefix) :]  # e.g. "llama"
    return TESTS_ROOT / model_dir / f"{test_prefix}{model_name}.py"


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    expected_test_file = _expected_test_file(file_path)
    if expected_test_file is None or expected_test_file.exists():
        return []

    # Deliberately no `# trf-ignore: TRF038` support: every modeling/processing file can be
    # exercised with a dummy config and randomly initialized weights, so there is no legitimate
    # per-file exemption. Models that genuinely cannot add a test yet go in
    # `allowlist_models` in rules.toml, which is visible in review instead of buried in the diff.
    return [
        Violation(
            file_path=file_path,
            line_number=1,
            message=(
                f"{RULE_ID}: no test file found at `{expected_test_file}` for `{file_path}`. "
                "Add one, even a minimal test built on a dummy config and randomly initialized "
                "weights if no real checkpoint exists yet."
            ),
        )
    ]
