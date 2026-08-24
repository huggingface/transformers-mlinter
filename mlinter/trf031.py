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

"""TRF031: Output-like dataclasses in modeling files must inherit ModelOutput."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption


def _is_dataclass(class_node: ast.ClassDef) -> bool:
    for decorator in class_node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        try:
            if full_name(target).split(".")[-1] == "dataclass":
                return True
        except ValueError:
            continue
    return False


def _annotation_name(annotation: ast.expr) -> str | None:
    target = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    try:
        return full_name(target).split(".")[-1]
    except ValueError:
        return None


def _has_dataclass_default(value: ast.expr | None) -> bool:
    if value is None:
        return False
    if not isinstance(value, ast.Call):
        return True
    try:
        func_name = full_name(value.func).split(".")[-1]
    except ValueError:
        return True
    if func_name != "field":
        return True
    return any(keyword.arg in {"default", "default_factory"} for keyword in value.keywords)


def _required_dataclass_field_count(class_node: ast.ClassDef) -> int:
    count = 0
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if _annotation_name(item.annotation) == "ClassVar":
            continue
        if not _has_dataclass_default(item.value):
            count += 1
    return count


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or not _is_dataclass(class_node):
            continue
        base_names = []
        for base in class_node.bases:
            try:
                base_names.append(full_name(base).split(".")[-1])
            except ValueError:
                continue
        # Any base carrying `Output` in its name is a ModelOutput subclass: ModelOutput itself, one of
        # the BaseModelOutputWith* variants, or another model's output class.
        if any("Output" in name for name in base_names):
            continue
        if _required_dataclass_field_count(class_node) >= 2:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, class_node.lineno):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=class_node.lineno,
                message=(
                    f"{RULE_ID}: `{class_node.name}` is a plain dataclass. "
                    "Inherit `ModelOutput` so it indexes like a tuple and picks up @auto_docstring."
                ),
            )
        )
    return violations
