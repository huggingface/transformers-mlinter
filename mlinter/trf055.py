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

"""TRF055: `config` on a PreTrainedModel subclass must be an annotation, not an assignment."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _simple_name, full_name, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []

    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        for item in node.body:
            if not (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == "config"
                and isinstance(item.value, (ast.Name, ast.Attribute))
            ):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, item.lineno):
                continue
            config_name = _simple_name(full_name(item.value))
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=item.lineno,
                    message=(
                        f"{RULE_ID}: `{node.name}.config = {config_name}` assigns a class attribute that "
                        f"shadows `config_class` resolution, causing `{node.name}.config_class` to return "
                        f"the wrong config. Use an annotation instead: `config: {config_name}`."
                    ),
                )
            )

    return violations
