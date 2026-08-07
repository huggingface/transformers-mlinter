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

"""TRF037: torch.einsum is hard to read; prefer explicit matmul/transpose operations."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            if full_name(node.func).split(".")[-1] != "einsum":
                continue
        except ValueError:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        equation = ""
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            equation = f" (`{node.args[0].value}`)"
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: `einsum`{equation} in a modeling file. "
                    "Write the operation out with matmul/transpose/reshape so the shapes are readable."
                ),
            )
        )
    return violations
