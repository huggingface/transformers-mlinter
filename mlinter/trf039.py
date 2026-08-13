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

"""TRF039: imports guarded by `is_*_available()` must be removed once nothing in the file uses them."""

import ast
import re
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression


RULE_ID = ""  # Set by discovery

_AVAILABILITY_CHECK_RE = re.compile(r"^is_[a-z0-9_]+_available$")


def _is_availability_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and bool(_AVAILABILITY_CHECK_RE.match(node.func.id))
    )


def _guards_on_availability(test: ast.AST) -> bool:
    """Whether *test* is, or is built out of, an `is_*_available()` call.

    Handles the common `is_vision_available() and is_torch_available()` / `not is_vision_available()`
    variants so the guard is still recognized when combined with other conditions.
    """
    if _is_availability_call(test):
        return True
    if isinstance(test, ast.BoolOp):
        return any(_guards_on_availability(value) for value in test.values)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _guards_on_availability(test.operand)
    return False


def _imported_bindings(stmt: ast.Import | ast.ImportFrom) -> list[tuple[str, ast.alias]]:
    """Return `(bound_name, alias_node)` pairs a single Import/ImportFrom statement introduces."""
    if isinstance(stmt, ast.ImportFrom):
        return [(alias.asname or alias.name, alias) for alias in stmt.names if alias.name != "*"]
    return [(alias.asname or alias.name.split(".")[0], alias) for alias in stmt.names]


def _guarded_imports(tree: ast.Module) -> list[tuple[str, ast.stmt, ast.alias | None]]:
    """Collect `(bound_name, import_stmt, alias)` for every import sitting inside an availability guard.

    An empty guard body is reported as the sentinel name `"pass"` with no alias.
    """
    found: list[tuple[str, ast.stmt, ast.alias | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _guards_on_availability(node.test):
            continue
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                found.extend((name, stmt, alias) for name, alias in _imported_bindings(stmt))
            elif isinstance(stmt, ast.Pass):
                found.append(("pass", stmt, None))
    return found


def _is_referenced_elsewhere(tree: ast.Module, name: str, import_stmt: ast.stmt) -> bool:
    """Whether *name* is used anywhere in *tree* outside of *import_stmt* itself.

    Also matches the name inside string literals so forward-reference type hints
    (`def f(x: "Image.Image")`) and `__all__` entries count as usage.
    """
    word = re.compile(rf"\b{re.escape(name)}\b")
    for node in ast.walk(tree):
        if node is import_stmt:
            continue
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and word.search(node.value):
            return True
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for name, stmt, alias in _guarded_imports(tree):
        if name == "pass":
            # dummy pass statement so we know it is not used anywhere in code
            line_number = stmt.lineno
            message = f"{RULE_ID}: Availability guard has an empty body — ruff removed the import it protected, remove the guard too."
        else:
            if _is_referenced_elsewhere(tree, name, stmt):
                continue

            line_number = getattr(alias, "lineno", stmt.lineno)
            if _has_rule_suppression(source_lines, RULE_ID, line_number):
                continue

            message = (
                f"{RULE_ID}: `{name}` is imported behind an availability guard but is never used in "
                f"this file. ruff does not flag or clean up imports inside `if is_*_available():` "
                "blocks, so a leftover import from a refactor stays behind silently. Remove it."
            )

        violations.append(
            Violation(
                file_path=file_path,
                line_number=line_number,
                message=message,
            )
        )
    return violations
