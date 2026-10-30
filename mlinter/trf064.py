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

"""TRF064: `<Model>ProcessorKwargs` must not override `_defaults` or the modality kwargs fields."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _simple_name, full_name


RULE_ID = ""  # Set by discovery

CHECKED_FILE_PREFIXES = ("processing_", "modular_")
# Fields of `ProcessingKwargs` that a subclass must inherit untouched.
INHERITED_FIELDS = frozenset({"_defaults", "text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"})
# `<Model>ProcessorKwargs` classes inherit `ProcessingKwargs` directly or another model's `<Model>ProcessorKwargs`.
PROCESSOR_KWARGS_SUFFIXES = ("ProcessingKwargs", "ProcessorKwargs")
# Image, video and audio processor kwargs end with `ProcessorKwargs` too, but are a different kind of TypedDict.
NON_PROCESSOR_KWARGS_SUFFIXES = ("ImageProcessorKwargs", "VideoProcessorKwargs", "AudioProcessorKwargs")
# Backward-compatibility shims that keep accepting the old nested keys are named `Deprecated<Model><Modality>Kwargs`.
DEPRECATED_PREFIX = "Deprecated"


def _is_processor_kwargs_class(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        try:
            name = _simple_name(full_name(base))
        except ValueError:
            continue
        if name.endswith(PROCESSOR_KWARGS_SUFFIXES) and not name.endswith(NON_PROCESSOR_KWARGS_SUFFIXES):
            return True
    return False


def _is_deprecated_shim(annotation: ast.expr) -> bool:
    try:
        return _simple_name(full_name(annotation)).startswith(DEPRECATED_PREFIX)
    except ValueError:
        return False


def _overridden_fields(class_node: ast.ClassDef) -> list[tuple[ast.stmt, str]]:
    """`(statement, field)` for every inherited field the class body assigns or re-annotates."""
    found = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            # Narrowing a modality field to a `Deprecated...Kwargs` shim keeps old nested keys working.
            if stmt.value is None and _is_deprecated_shim(stmt.annotation):
                continue
            targets = [stmt.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in INHERITED_FIELDS:
                found.append((stmt, target.id))
    return found


def _message(class_name: str, field: str, stmt: ast.stmt) -> str:
    if field == "_defaults":
        fix = (
            "Move each entry of `_defaults` to a class attribute of the processor, e.g. "
            '`class FooProcessor(ProcessorMixin): text_kwargs = {"padding": False}`. '
            "`ProcessingKwargs._defaults` is deprecated and only kept for backward compatibility."
        )
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is None:
        fix = (
            "Keep the type inherited from `ProcessingKwargs` and declare model-specific kwargs at the top level "
            f"of `{class_name}`. Only a `{DEPRECATED_PREFIX}<Model>...Kwargs` backward-compatibility shim may be "
            "used here."
        )
    else:
        fix = (
            f"Set defaults as a class attribute of the processor instead, e.g. "
            f"`class FooProcessor(ProcessorMixin): {field} = {{...}}`."
        )
    return f"{RULE_ID}: `{class_name}` overrides `{field}` inherited from `ProcessingKwargs`. {fix}"


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(CHECKED_FILE_PREFIXES):
        return []

    violations = []
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and _is_processor_kwargs_class(node)):
            continue
        for stmt, field in _overridden_fields(node):
            if _has_rule_suppression(source_lines, RULE_ID, stmt.lineno):
                continue
            violations.append(
                Violation(file_path=file_path, line_number=stmt.lineno, message=_message(node.name, field, stmt))
            )
    return violations
