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

"""TRF051: modeling code must not branch on _attn_implementation; the attention interface owns dispatch."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(
            isinstance(side, ast.Attribute) and side.attr == "_attn_implementation"
            for side in [node.left, *node.comparators]
        ):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: modeling code compares _attn_implementation. Dispatch belongs to "
                    "ALL_ATTENTION_FUNCTIONS.get_interface, and backend-conditional tensor handling belongs "
                    "in the backend integration under integrations/."
                ),
            )
        )

    return violations
