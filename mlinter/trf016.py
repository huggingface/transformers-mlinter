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

"""TRF016: do_* flags on a processor class must be referenced by overridden preprocess/_preprocess."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _class_methods,
    _function_uses_name,
    _get_class_assignments,
    _has_rule_suppression,
    is_super_method_call,
)


RULE_ID = ""  # Set by discovery

# Flags consumed by the base preprocess() before _preprocess() runs, so an
# override is not required to reference them. Keep this list tight: the whole
# point of the rule is to catch flags that silently do nothing.
_BASE_HANDLED_FLAGS = frozenset({"do_sample_frames"})

_PROCESSOR_FILE_PREFIXES = ("image_processing_", "video_processing_")

_OVERRIDABLE_METHODS = ("_preprocess", "preprocess")


def _is_processor_file(file_path: Path) -> bool:
    return file_path.suffix == ".py" and file_path.name.startswith(_PROCESSOR_FILE_PREFIXES)


def _is_bool_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, bool)


def _collect_do_flags(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    flags: dict[str, ast.AST] = {}
    for name, value in _get_class_assignments(class_node).items():
        if not name.startswith("do_"):
            continue
        if not _is_bool_constant(value):
            continue
        flags[name] = value
    return flags


def _function_forwards_to_super(function_node: ast.FunctionDef) -> bool:
    """Return True if the function calls super().preprocess(...) or super()._preprocess(...)
    with **kwargs forwarding, i.e. the override delegates flag handling back to the base."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if not any(is_super_method_call(node, method) for method in _OVERRIDABLE_METHODS):
            continue
        for keyword in node.keywords:
            if keyword.arg is None:  # **kwargs
                return True
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []

    if not _is_processor_file(file_path):
        return violations

    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, class_node.lineno):
            continue

        flags = _collect_do_flags(class_node)
        if not flags:
            continue

        methods = _class_methods(class_node)
        override = next((methods[name] for name in _OVERRIDABLE_METHODS if name in methods), None)
        if override is None:
            # Class doesn't override preprocess; the base applies the flags.
            continue

        if _function_forwards_to_super(override):
            # Override forwards **kwargs to super(); flags reach the base implementation.
            continue

        for flag_name, flag_value in flags.items():
            if flag_name in _BASE_HANDLED_FLAGS:
                continue
            if _function_uses_name(override, flag_name):
                continue
            line = getattr(flag_value, "lineno", class_node.lineno)
            if _has_rule_suppression(source_lines, RULE_ID, line):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=line,
                    message=(
                        f"{RULE_ID}: {class_node.name}.{flag_name} is declared but never referenced "
                        f"by the overridden {override.name}() — the flag is dead. Either gate the "
                        f"corresponding operation behind `if {flag_name}:` or remove the attribute."
                    ),
                )
            )

    return violations
