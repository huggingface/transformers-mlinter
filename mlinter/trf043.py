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

"""TRF043: Attention classes must not declare position_ids in their forward signature."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _function_argument_names, _has_rule_suppression, _top_level_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        if not node.name.endswith("Attention"):
            continue

        forward = _class_methods(node).get("forward")
        if forward is None or "position_ids" not in _function_argument_names(forward):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, forward.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=forward.lineno,
                message=(
                    f"{RULE_ID}: {node.name}.forward declares position_ids in its signature. "
                    "Attention modules receive position_ids through **kwargs so padding-free "
                    "flash-attention can consume it; take position_embeddings and **kwargs instead."
                ),
            )
        )

    return violations
