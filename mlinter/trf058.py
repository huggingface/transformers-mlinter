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

"""TRF058: buffers must be declared as `nn.Buffer` attributes, not via `register_buffer()`."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name


RULE_ID = ""  # Set by discovery


def _buffer_name(call: ast.Call) -> str | None:
    """Return the buffer name when `register_buffer` is called with a literal name.

    A non-literal name (a variable or an f-string, e.g. registering one buffer per layer in a loop)
    has no equivalent attribute assignment, so those calls are left alone.
    """
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.isidentifier():
        return first.value
    return None


def _persistent_suffix(call: ast.Call) -> str:
    """Render the `persistent=` keyword for the suggested `nn.Buffer(...)` call, if the call passes one."""
    for keyword in call.keywords:
        if keyword.arg == "persistent" and isinstance(keyword.value, ast.Constant):
            return f", persistent={keyword.value.value}"
    return ""


def _receiver_name(call: ast.Call) -> str | None:
    """Return the dotted name of the object `register_buffer` is called on, or None if unrenderable."""
    assert isinstance(call.func, ast.Attribute)
    try:
        return full_name(call.func.value)
    except ValueError:
        return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register_buffer"
        ):
            continue

        name = _buffer_name(node)
        if name is None:
            continue
        receiver = _receiver_name(node)
        if receiver is None:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        # `nn.Buffer` takes the same `persistent` flag, so carry it into the suggestion.
        persistent = _persistent_suffix(node)
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f'{RULE_ID}: `{receiver}.register_buffer("{name}", ...)` registers a buffer through a '
                    f"method call. Assign it instead: `{receiver}.{name} = nn.Buffer(...{persistent})`. "
                    "A buffer that is a plain attribute can be inherited and tweaked in a modular file; "
                    "one created by a method call cannot, so the whole `__init__` has to be redefined."
                ),
            )
        )

    return violations
