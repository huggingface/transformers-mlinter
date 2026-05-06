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

"""TRF017: @auto_docstring must be placed above @dataclass on output classes."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _simple_name, full_name


RULE_ID = ""  # Set by discovery


def _decorator_simple_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    try:
        return _simple_name(full_name(target))
    except ValueError:
        return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        dataclass_idx: int | None = None
        dataclass_decorator: ast.expr | None = None
        autodoc_idx: int | None = None
        autodoc_decorator: ast.expr | None = None
        for idx, decorator in enumerate(node.decorator_list):
            name = _decorator_simple_name(decorator)
            if name == "dataclass" and dataclass_idx is None:
                dataclass_idx = idx
                dataclass_decorator = decorator
            elif name == "auto_docstring" and autodoc_idx is None:
                autodoc_idx = idx
                autodoc_decorator = decorator

        if dataclass_idx is None or autodoc_idx is None:
            continue
        # decorator_list is in source order (top first); top decorator is applied last.
        # @dataclass must run first to synthesize __init__, so it must sit BELOW @auto_docstring.
        if dataclass_idx >= autodoc_idx:
            continue

        suppress_lines = {
            node.lineno,
            getattr(autodoc_decorator, "lineno", node.lineno),
            getattr(dataclass_decorator, "lineno", node.lineno),
        }
        if any(_has_rule_suppression(source_lines, RULE_ID, lineno) for lineno in suppress_lines):
            continue

        line_number = getattr(autodoc_decorator, "lineno", node.lineno)
        violations.append(
            Violation(
                file_path=file_path,
                line_number=line_number,
                message=(
                    f"{RULE_ID}: {node.name} has @dataclass listed above @auto_docstring; "
                    "swap them so @dataclass sits directly above the class."
                ),
            )
        )

    return violations
