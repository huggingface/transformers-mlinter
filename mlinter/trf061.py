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

"""TRF061: ProcessingKwargs modality fields must use the shared kwargs types."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml

_MODALITY_TYPES = {
    "text_kwargs": "TextKwargs",
    "images_kwargs": "ImagesKwargs",
    "audio_kwargs": "AudioKwargs",
    "videos_kwargs": "VideosKwargs",
}


def _type_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return ""
    try:
        name = full_name(node)
    except ValueError:
        return ""
    return aliases.get(name, name).rsplit(".", 1)[-1]


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    aliases = {
        alias.asname or alias.name: alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    # Resolve local inheritance to a fixed point; unknown imported bases stay inconclusive.
    processing_classes = {"ProcessingKwargs"}
    while True:
        subclasses = {
            node.name
            for node in classes
            if any(_type_name(base, aliases) in processing_classes for base in node.bases)
        }
        if subclasses <= processing_classes:
            break
        processing_classes.update(subclasses)

    violations: list[Violation] = []
    for node in classes:
        if not any(_type_name(base, aliases) in processing_classes for base in node.bases):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            expected = _MODALITY_TYPES.get(stmt.target.id)
            if expected is None or _type_name(stmt.annotation, aliases) == expected:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=stmt.lineno,
                    message=(
                        f"{RULE_ID}: `{node.name}.{stmt.target.id}` must be typed as `{expected}`, "
                        f"not `{ast.unparse(stmt.annotation)}`. Use the shared modality kwargs type "
                        "instead of a model-specific class."
                    ),
                )
            )
    return violations
