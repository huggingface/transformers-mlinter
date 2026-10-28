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

"""TRF061: classes in a modular file must not subclass another class defined in the same file."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

_PRETRAINED_MODEL_SUFFIX = "PreTrainedModel"
_WARNING_CALLS = {"warn", "warning", "warning_once", "warning_advice"}


def _base_name(base: ast.expr) -> str | None:
    # `Foo[T]` subclasses `Foo`.
    if isinstance(base, ast.Subscript):
        base = base.value
    try:
        return full_name(base)
    except ValueError:
        return None


def _pretrained_model_classes(local_bases: dict[str, list[str]]) -> set[str]:
    """Local classes that are `PreTrainedModel` subclasses, as far as the file and the names tell."""

    def is_pretrained(name: str, visiting: set[str]) -> bool:
        if name.split(".")[-1].endswith(_PRETRAINED_MODEL_SUFFIX):
            return True
        if name not in local_bases or name in visiting:
            return False
        visiting.add(name)
        return any(is_pretrained(base, visiting) for base in local_bases[name])

    return {name for name in local_bases if is_pretrained(name, set())}


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_warning_call(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr in _WARNING_CALLS
    )


def _is_super_init_call(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "__init__"
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "super"
    )


def _is_backward_compat_alias(class_node: ast.ClassDef) -> bool:
    """For BC we have some aliases that inherit from the new class name.

    This is allowed as long as the alias only calls `super().__init__()` and optionally warns.
    """
    body = [stmt for stmt in class_node.body if not _is_docstring(stmt)]
    if len(body) != 1:
        return False
    init = body[0]
    if not isinstance(init, ast.FunctionDef) or init.name != "__init__":
        return False
    statements = [stmt for stmt in init.body if not _is_docstring(stmt)]
    if not statements or not _is_super_init_call(statements[-1]):
        return False
    return all(_is_warning_call(stmt) for stmt in statements[:-1])


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith("modular_"):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    # Only classes defined *before* a subclass can be its base at import time, but every local class
    # is collected up front so the PreTrainedModel resolution can follow the whole local chain.
    local_bases: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            local_bases[node.name] = [name for base in node.bases if (name := _base_name(base)) is not None]
    pretrained_classes = _pretrained_model_classes(local_bases)

    defined: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        local_parents = [base for base in local_bases[node.name] if base in defined and base not in pretrained_classes]
        defined.add(node.name)
        if not local_parents or _is_backward_compat_alias(node):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        parents = ", ".join(local_parents)
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {node.name} subclasses {parents}, defined earlier in this modular file. The "
                    "generated modeling file keeps that inheritance, so the class cannot be read on its own there. "
                    "Subclass the class imported from another model instead, or write the class out in full."
                ),
            )
        )

    return violations
