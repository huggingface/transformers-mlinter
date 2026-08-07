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

"""TRF027: Model files must raise explicit errors instead of using bare `assert`."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_", "configuration_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: bare `assert` in a model file. "
                    "Raise a ValueError with an actionable message instead; asserts vanish under `python -O`."
                ),
            )
        )
    return violations
