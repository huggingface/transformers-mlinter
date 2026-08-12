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

"""TRF040: @can_return_tuple must not be combined with @capture_outputs."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _simple_name, full_name


RULE_ID = ""  # Set by discovery

_CAPTURE_OUTPUTS = "capture_outputs"
_CAN_RETURN_TUPLE = "can_return_tuple"


def _decorator_simple_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    try:
        return _simple_name(full_name(target))
    except ValueError:
        return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        can_return_tuple: ast.expr | None = None
        has_capture_outputs = False
        for decorator in node.decorator_list:
            name = _decorator_simple_name(decorator)
            if name == _CAN_RETURN_TUPLE and can_return_tuple is None:
                can_return_tuple = decorator
            elif name == _CAPTURE_OUTPUTS:
                has_capture_outputs = True

        if can_return_tuple is None or not has_capture_outputs:
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=can_return_tuple.lineno,
                message=(
                    f"{RULE_ID}: {node.name} combines @{_CAN_RETURN_TUPLE} with @{_CAPTURE_OUTPUTS}. "
                    f"@{_CAPTURE_OUTPUTS} already handles returning tuples, drop @{_CAN_RETURN_TUPLE}."
                ),
            )
        )

    return violations
