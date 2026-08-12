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

"""TRF046: forward must not write module attributes; modules are stateless in forward."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _class_methods,
    _has_rule_suppression,
    _self_attribute_targets,
    _top_level_classes,
)


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        forward = _class_methods(node).get("forward")
        if forward is None:
            continue

        # ast.walk deliberately descends into nested functions: a closure defined in forward still
        # mutates the module when it runs, so the write is forward-time state either way.
        for stmt in ast.walk(forward):
            for target in _self_attribute_targets(stmt):
                if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
                    continue
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=stmt.lineno,
                        message=(
                            f"{RULE_ID}: {node.name}.forward writes self.{target.attr}. Hidden state in forward "
                            "breaks batching, compile, and reuse; pass carried state explicitly (cache objects) "
                            "or compute config-derived values in __init__."
                        ),
                    )
                )

    return violations
