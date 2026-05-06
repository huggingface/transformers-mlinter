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

"""TRF018: _init_weights overrides should call super()._init_weights(module)."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_super_method_call, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def _is_unbound_init_weights_call(node: ast.AST) -> bool:
    """Return True for `<SomeClass>._init_weights(self, ...)` — the modular-file equivalent of super()."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != "_init_weights":
        return False
    if not node.args:
        return False
    first = node.args[0]
    return isinstance(first, ast.Name) and first.id == "self"


def _is_modular_delete_sentinel(function_node: ast.FunctionDef) -> bool:
    """In modular files, `def _init_weights(self, module): raise AttributeError(...)` means "drop this method"."""
    body = function_node.body
    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "AttributeError"


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []

    for class_node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        for sub_node in class_node.body:
            if not (isinstance(sub_node, ast.FunctionDef) and sub_node.name == "_init_weights"):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, sub_node.lineno):
                continue

            args = sub_node.args.args
            if len(args) < 2 or getattr(args[0], "arg", None) != "self":
                continue

            if "modular_" in file_path.name and _is_modular_delete_sentinel(sub_node):
                continue

            calls_super = any(
                is_super_method_call(node, method="_init_weights") or _is_unbound_init_weights_call(node)
                for node in ast.walk(sub_node)
            )
            if calls_super:
                continue

            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=sub_node.lineno,
                    message=(
                        f"{RULE_ID}: `_init_weights` of {class_node.name} does not call "
                        f"`super()._init_weights(...)`. If this is intentional, suppress with "
                        f"`# trf-ignore: {RULE_ID}` on the line above the method."
                    ),
                )
            )

    return violations
