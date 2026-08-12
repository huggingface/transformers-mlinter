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

"""TRF050: one rotary module per model; attention classes must not instantiate their own rotary embedding."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _has_rule_suppression, _top_level_classes, full_name


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        if not node.name.endswith("Attention"):
            continue

        init_method = _class_methods(node).get("__init__")
        if init_method is None:
            continue

        for sub in ast.walk(init_method):
            if not isinstance(sub, ast.Call):
                continue
            try:
                callee = full_name(sub.func).split(".")[-1]
            except ValueError:
                continue
            if not callee.endswith("RotaryEmbedding"):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, sub.lineno):
                continue

            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=sub.lineno,
                    message=(
                        f"{RULE_ID}: {node.name}.__init__ instantiates {callee}. The Model owns a single "
                        "rotary_emb and passes cos/sin down as position_embeddings; a per-attention rotary "
                        "module recomputes inv_freq per layer."
                    ),
                )
            )

    return violations
