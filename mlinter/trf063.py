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

"""TRF063: _init_weights must use the transformers `initialization` module, not torch.nn.init."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name


RULE_ID = ""  # Set by discovery

_TORCH_INIT = "torch.nn.init"


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map each locally bound name that may lead to torch.nn.init to its fully qualified name."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    if alias.name == _TORCH_INIT or _TORCH_INIT.startswith(alias.name + "."):
                        aliases[alias.asname] = alias.name
                else:
                    # `import torch.nn.init` binds `torch`.
                    root = alias.name.split(".")[0]
                    if root == "torch":
                        aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                if qualified == _TORCH_INIT or _TORCH_INIT.startswith(qualified + ".") or node.module == _TORCH_INIT:
                    aliases[alias.asname or alias.name] = qualified
    return aliases


def _resolve_call_target(call: ast.Call, aliases: dict[str, str]) -> str | None:
    if not isinstance(call.func, (ast.Name, ast.Attribute)):
        return None
    try:
        dotted = full_name(call.func)
    except ValueError:
        return None
    head, _, rest = dotted.partition(".")
    if head not in aliases:
        return None
    return f"{aliases[head]}.{rest}" if rest else aliases[head]


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    aliases = _import_aliases(tree)
    if not aliases:
        return violations

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_init_weights"):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        for sub_node in ast.walk(node):
            if not isinstance(sub_node, ast.Call):
                continue
            target = _resolve_call_target(sub_node, aliases)
            if target is None or not target.startswith(_TORCH_INIT + "."):
                continue
            function_name = target[len(_TORCH_INIT) + 1 :]
            # Private helpers such as `_calculate_fan_in_and_fan_out` only compute values and never
            # write to a tensor, so they cannot bypass the initialization flags.
            if "." in function_name or function_name.startswith("_"):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, sub_node.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=sub_node.lineno,
                    message=(
                        f"{RULE_ID}: `_init_weights` calls `{_TORCH_INIT}.{function_name}`. Use "
                        f"`init.{function_name}` from the transformers initialization module instead, imported "
                        "as `from ... import initialization as init` (not `from torch.nn import init`)."
                    ),
                )
            )

    return violations
