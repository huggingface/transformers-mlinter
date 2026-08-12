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

"""TRF048: _tied_weights_keys must be a dict mapping target to source, not the pre-v5 list form."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _top_level_classes


RULE_ID = ""  # Set by discovery


def _tied_weights_assignment(class_node: ast.ClassDef) -> ast.stmt | None:
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_tied_weights_keys" for t in item.targets):
                return item
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and item.target.id == "_tied_weights_keys" and item.value is not None:
                return item
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        stmt = _tied_weights_assignment(node)
        if stmt is None or not isinstance(stmt.value, (ast.List, ast.Tuple, ast.Set)):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=stmt.lineno,
                message=(
                    f"{RULE_ID}: {node.name} declares _tied_weights_keys as a list. The v5 form is a dict "
                    'mapping target to source, e.g. {"lm_head.weight": "model.embed_tokens.weight"}.'
                ),
            )
        )

    return violations
