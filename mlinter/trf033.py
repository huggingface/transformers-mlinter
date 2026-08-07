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

"""TRF033: Hyperparameters must be set on the config, not mutated through a set_* method."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# The setters that are part of the PreTrainedModel contract.
SANCTIONED_SETTERS = {
    "set_input_embeddings",
    "set_output_embeddings",
    "set_decoder",
    "set_encoder",
    "set_attn_implementation",
    "set_default_language",
}


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for item in class_node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if not item.name.startswith("set_") or item.name in SANCTIONED_SETTERS:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, item.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=item.lineno,
                    message=(
                        f"{RULE_ID}: `{class_node.name}.{item.name}` mutates a hyperparameter after construction. "
                        "Put the value on the config and read it where it is used."
                    ),
                )
            )
    return violations
