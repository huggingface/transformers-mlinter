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

"""TRF042: A tokenizer test must exercise the shared TokenizerTesterMixin suite."""

import ast
from pathlib import Path

from ._helpers import (
    TESTS_ROOT,
    Violation,
    _collect_class_bases,
    _has_rule_suppression,
    full_name,
    is_exempt_by_cutoff,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

MIXIN = "TokenizerTesterMixin"

_TEST_MODULE_PREFIX = "test_tokenization_"
# A model's tokenizer test may inherit another model's, which inherits the mixin. Two hops covers the
# chains in the library (distilbert -> bert) with room to spare, and bounds the files a check reads.
_MAX_INHERITANCE_HOPS = 4


def _base_name(base: ast.expr) -> str:
    try:
        return full_name(base).split(".")[-1]
    except ValueError:
        return ""


def _is_test_class(class_node: ast.ClassDef) -> bool:
    """Whether the class is one the test runner collects, rather than a helper or a mixin.

    A `TestCase` base is the real marker, and the naming convention covers the case where the base is
    another model's test class instead (`FooTokenizationTest(BertTokenizationTest)`). Deriving from a
    mixin is deliberately not enough: `TokenizerTesterMixin` is the suite, not a test class, so a helper
    that mixes it in never satisfies the rule on a real test class's behalf.
    """
    if class_node.name.endswith(("Test", "TestCase")):
        return True
    return any(_base_name(base).endswith("TestCase") for base in class_node.bases)


def _dotted_bases(class_node: ast.ClassDef) -> list[str]:
    """The dotted name of every base, skipping the ones `full_name` cannot render (subscripts, calls)."""
    names = []
    for base in class_node.bases:
        try:
            names.append(full_name(base))
        except ValueError:
            continue
    return names


def _test_module_stems(tree: ast.Module) -> dict[str, str]:
    """Maps each imported name to the tokenizer-test module it came from.

    Covers both spellings the library uses: `from ..bert import test_tokenization_bert`, where the bound
    name is the module itself, and `from ..bert.test_tokenization_bert import BertTokenizationTest`,
    where it is the class. Either way the module stem names the file the class is defined in.
    """
    stems: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_stem = (node.module or "").split(".")[-1]
        for alias in node.names:
            bound = alias.asname or alias.name
            if alias.name.startswith(_TEST_MODULE_PREFIX):
                stems[bound] = alias.name
            elif module_stem.startswith(_TEST_MODULE_PREFIX):
                stems[bound] = module_stem
    return stems


def _test_file_for_stem(stem: str) -> Path | None:
    """`test_tokenization_bert` -> `tests/models/bert/test_tokenization_bert.py`, if it exists."""
    model = stem.removeprefix(_TEST_MODULE_PREFIX)
    if not model:
        return None
    candidate = TESTS_ROOT / model / f"{stem}.py"
    return candidate if candidate.is_file() else None


def _parse_test_file(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return None


def _inherits_mixin(bases: list[str], tree: ast.Module, seen: set[str], hops: int) -> bool:
    """Whether any of `bases` is the mixin, or reaches it through a class this file can resolve.

    A base resolves either locally, when the class is defined in the same file, or across files, when it
    is another model's tokenizer test imported by name. Anything else — a third-party base, a class from
    a module outside the tests tree — is left alone, so an unresolvable base never counts as satisfied.
    """
    if any(base.split(".")[-1] == MIXIN for base in bases):
        return True
    if hops >= _MAX_INHERITANCE_HOPS:
        return False

    local_bases = _collect_class_bases(tree)
    stems = _test_module_stems(tree)

    for base in bases:
        class_name = base.split(".")[-1]
        if class_name in local_bases:
            key = f"local:{class_name}"
            if key in seen:
                continue
            seen.add(key)
            if _inherits_mixin(local_bases[class_name], tree, seen, hops + 1):
                return True
            continue

        stem = stems.get(base.split(".")[0])
        path = _test_file_for_stem(stem) if stem else None
        if path is None or str(path) in seen:
            continue
        seen.add(str(path))
        other = _parse_test_file(path)
        if other is None:
            continue
        other_bases = {node.name: _dotted_bases(node) for node in other.body if isinstance(node, ast.ClassDef)}
        if class_name in other_bases and _inherits_mixin(other_bases[class_name], other, seen, hops + 1):
            return True
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(_TEST_MODULE_PREFIX):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    test_classes: list[ast.ClassDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_test_class(node):
            continue
        # Only a test class can satisfy the rule. A helper that carries the mixin does not run under the
        # test runner, so it would otherwise silence the file while the real test class skips the suite.
        if _inherits_mixin(_dotted_bases(node), tree, set(), 0):
            return []
        test_classes.append(node)

    if not test_classes:
        return []

    target = test_classes[0]
    if _has_rule_suppression(source_lines, RULE_ID, target.lineno):
        return []

    return [
        Violation(
            file_path=file_path,
            line_number=target.lineno,
            message=(
                f"{RULE_ID}: `{target.name}` does not inherit `{MIXIN}`. "
                "Add it so the tokenizer runs the shared round-trip, padding, truncation and "
                "special-token suite instead of only its own assertions."
            ),
        )
    ]
