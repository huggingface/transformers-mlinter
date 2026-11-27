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

"""TRF069: `_get_num_multimodal_tokens` must read its `**kwargs` through `self._merge_kwargs`."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_self_method_call, is_super_method_call


RULE_ID = ""  # Set by discovery

CHECKED_FILE_PREFIXES = ("processing_", "modular_")
COUNTER_NAME = "_get_num_multimodal_tokens"


def _is_allowed_sink(call: ast.Call) -> bool:
    return is_self_method_call(call, "_merge_kwargs") or is_super_method_call(call, COUNTER_NAME)


def _allowed_uses(function_node: ast.FunctionDef, kwargs_name: str) -> set[int]:
    """Ids of the `kwargs` names unpacked as `**kwargs` into `self._merge_kwargs` or the parent counter."""
    allowed = set()
    for node in ast.walk(function_node):
        if not (isinstance(node, ast.Call) and _is_allowed_sink(node)):
            continue
        for keyword in node.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id == kwargs_name:
                allowed.add(id(keyword.value))
    return allowed


def _message(kwargs_name: str) -> str:
    return (
        f"{RULE_ID}: `{COUNTER_NAME}` uses `{kwargs_name}` directly. Merge it with "
        f'`self._merge_kwargs(**{kwargs_name})["images_kwargs"]` (or `"videos_kwargs"`, `"audio_kwargs"`) so '
        "kwargs passed nested, e.g. `images_kwargs={...}`, count as they do in the processor call. To read an "
        "image or video processor kwarg yourself, pass the merged dict through "
        "`self.image_processor._merge_and_standardize_kwargs` first."
    )


def _iter_violations(tree: ast.Module):
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for function_node in class_node.body:
            if not (isinstance(function_node, ast.FunctionDef) and function_node.name == COUNTER_NAME):
                continue
            if function_node.args.kwarg is None:
                continue
            kwargs_name = function_node.args.kwarg.arg
            allowed = _allowed_uses(function_node, kwargs_name)
            lines = {
                node.lineno
                for node in ast.walk(function_node)
                if isinstance(node, ast.Name) and node.id == kwargs_name and id(node) not in allowed
            }
            for line_number in sorted(lines):
                yield line_number, _message(kwargs_name)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(CHECKED_FILE_PREFIXES):
        return []

    return [
        Violation(file_path=file_path, line_number=line_number, message=message)
        for line_number, message in _iter_violations(tree)
        if not _has_rule_suppression(source_lines, RULE_ID, line_number)
    ]
