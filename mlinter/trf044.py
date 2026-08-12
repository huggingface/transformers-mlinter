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

"""TRF044: cache_position is removed framework surface and must not reappear in modeling signatures."""

import ast
from pathlib import Path

from ._helpers import Violation, _function_argument_names, _has_rule_suppression, _module_and_method_functions


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _module_and_method_functions(tree):
        if "cache_position" not in _function_argument_names(node):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {node.name} declares a cache_position parameter. cache_position was removed "
                    "from all models; the cache update call is past_key_values.update(key_states, value_states, "
                    "self.layer_idx) with no position threading."
                ),
            )
        )

    return violations
