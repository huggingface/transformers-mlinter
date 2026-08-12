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

"""TRF054: processor media token ids are properties, never instance attributes set in __init__."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _has_rule_suppression, _top_level_classes, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

_TOKEN_ID_ATTRS = ("image_token_id", "video_token_id", "audio_token_id")


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith("processing_"):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        init_method = _class_methods(node).get("__init__")
        if init_method is None:
            continue

        for stmt in ast.walk(init_method):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in _TOKEN_ID_ATTRS
                ):
                    continue
                if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
                    continue

                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=stmt.lineno,
                        message=(
                            f"{RULE_ID}: {node.name}.__init__ sets self.{target.attr}. Instance attributes "
                            "serialize into processor_config.json, which v5 forbids; define it as a property "
                            "reading the tokenizer."
                        ),
                    )
                )

    return violations
