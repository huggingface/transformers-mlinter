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

"""TRF053: no manual label shifting in modeling code; self.loss_function owns it."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression


RULE_ID = ""  # Set by discovery

_SHIFT_NAMES = ("shift_logits", "shift_labels", "shifted_logits", "shifted_labels")


def _slices_a_sequence(value: ast.AST) -> bool:
    """Whether `value` slices something, as in `labels[..., 1:]` or `logits[..., :-1, :]`."""
    for node in ast.walk(value):
        if isinstance(node, ast.Subscript) and any(isinstance(part, ast.Slice) for part in ast.walk(node.slice)):
            return True
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        shift_names = [t.id for t in targets if isinstance(t, ast.Name) and t.id in _SHIFT_NAMES]
        if not shift_names or node.value is None or not _slices_a_sequence(node.value):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: modeling code builds {shift_names[0]} by slicing. self.loss_function owns "
                    "shifting: pass unshifted labels as labels=..., and already-shifted labels (encoder-decoder, "
                    "packed sequences) as shift_labels=...; shifting twice is the recurring training-loss bug."
                ),
            )
        )

    return violations
