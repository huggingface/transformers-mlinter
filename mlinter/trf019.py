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

"""TRF019: `ProcessorKwargs` must not define non-empty `_defaults`; move them in `processor_config.json` in the hub."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption


def _is_processing_file(file_path: Path) -> bool:
    return file_path.suffix == ".py" and file_path.name.startswith("processing_")


def _defaults_assignment(class_node: ast.ClassDef) -> ast.stmt | None:
    """Return the AST statement for `_defaults = ...` inside the class body, or None."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_defaults" for t in item.targets):
                return item
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and item.target.id == "_defaults" and item.value is not None:
                return item
    return None


def _is_non_empty_dict(node: ast.AST) -> bool:
    return isinstance(node, ast.Dict) and len(node.keys) > 0


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not _is_processing_file(file_path):
        return []

    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        if not any(base.id == "ProcessingKwargs" for base in node.bases if isinstance(base, ast.Name)):
            continue

        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        stmt = _defaults_assignment(node)
        if stmt is None:
            continue

        value = stmt.value
        if not _is_non_empty_dict(value):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=stmt.lineno,
                message=(
                    f"{RULE_ID}: `{node.name}` sets `_defaults` in code. "
                    "Move processor defaults to `processor_config.json` instead."
                ),
            )
        )

    return violations
