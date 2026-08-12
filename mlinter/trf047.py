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

"""TRF047: image/video processors are stateless; preprocess and post_process must not write self attributes."""

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


def _is_processing_method(name: str) -> bool:
    return name in ("preprocess", "_preprocess", "__call__") or name.startswith("post_process")


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("image_processing_", "video_processing_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        for method in _class_methods(node).values():
            if not _is_processing_method(method.name):
                continue

            # ast.walk deliberately descends into nested functions: a closure defined in the method
            # still mutates the processor when it runs, so the write is call-time state either way.
            for stmt in ast.walk(method):
                for target in _self_attribute_targets(stmt):
                    if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
                        continue
                    violations.append(
                        Violation(
                            file_path=file_path,
                            line_number=stmt.lineno,
                            message=(
                                f"{RULE_ID}: {node.name}.{method.name} writes self.{target.attr}. Processors are "
                                "stateless: carried state breaks preprocess-many-then-postprocess batching; "
                                "return the value or pass it through the method chain."
                            ),
                        )
                    )

    return violations
