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

"""TRF034: Layer classes held in an nn.ModuleList must subclass GradientCheckpointingLayer."""

import ast
from pathlib import Path

from ._helpers import Violation, _collect_class_bases, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Only the repeated per-layer blocks are in scope; a ModuleList of projections or experts is not a
# gradient-checkpointing boundary.
LAYER_CLASS_SUFFIXES = ("Layer", "Block")


def _subclasses_gradient_checkpointing_layer(name: str, class_to_bases: dict[str, list[str]]) -> bool:
    seen: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for base in class_to_bases.get(current, []):
            simple = base.split(".")[-1]
            if simple == "GradientCheckpointingLayer":
                return True
            stack.append(simple)
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    class_to_bases = _collect_class_bases(tree)
    local_classes = set(class_to_bases)
    violations: list[Violation] = []
    reported: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            if full_name(node.func).split(".")[-1] != "ModuleList":
                continue
        except ValueError:
            continue
        for inner in ast.walk(node):
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
                continue
            layer_name = inner.func.id
            if layer_name not in local_classes or not layer_name.endswith(LAYER_CLASS_SUFFIXES):
                continue
            if layer_name in reported:
                continue
            if _subclasses_gradient_checkpointing_layer(layer_name, class_to_bases):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue
            reported.add(layer_name)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{layer_name}` is stacked in an `nn.ModuleList` but does not subclass "
                        "`GradientCheckpointingLayer`, so gradient checkpointing silently skips it."
                    ),
                )
            )
    return violations
