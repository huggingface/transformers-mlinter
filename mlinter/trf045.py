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

"""TRF045: forward must not declare output_attentions/output_hidden_states/return_dict parameters."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _function_argument_names,
    _has_rule_suppression,
    _module_and_method_functions,
    is_exempt_by_cutoff,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

_LEGACY_OUTPUT_ARGS = ("output_attentions", "output_hidden_states", "return_dict")


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in _module_and_method_functions(tree):
        if node.name != "forward":
            continue

        legacy_args = [name for name in _LEGACY_OUTPUT_ARGS if name in _function_argument_names(node)]
        if not legacy_args:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: forward declares {', '.join(legacy_args)}. The decorator stack owns these: "
                    "@capture_outputs resolves output_* flags and @can_return_tuple handles return_dict; "
                    "remove them from the signature."
                ),
            )
        )

    return violations
