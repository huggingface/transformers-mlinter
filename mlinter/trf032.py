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

"""TRF032: Masked positions must be filled with torch.finfo(dtype).min, not a magic negative number."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, call_leaf_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

FILL_FUNCTIONS = {"masked_fill", "masked_fill_", "full", "full_like", "new_full"}
# Anything this large is standing in for negative infinity rather than being a real value.
MAGIC_MAGNITUDE = 1e3


def _magic_negative(node: ast.AST) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, int | float):
            if not isinstance(inner.value, bool) and abs(inner.value) >= MAGIC_MAGNITUDE:
                return -float(inner.value)
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        leaf = call_leaf_name(node)
        if leaf not in FILL_FUNCTIONS:
            continue
        for argument in list(node.args) + [keyword.value for keyword in node.keywords]:
            value = _magic_negative(argument)
            if value is None:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                break
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{leaf}` fills with the magic value {value:g}. "
                        "Use `torch.finfo(dtype).min` so the fill is correct in every dtype."
                    ),
                )
            )
            break
    return violations
