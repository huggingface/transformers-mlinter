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

"""TRF056: `forward` must not materialize tensor values with `.item()` or `.tolist()`."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, call_leaf_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Both force a device-to-host sync and turn a tensor into a Python value, which dynamo cannot trace.
MATERIALIZING_METHODS = ("item", "tolist")


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _split_calls(function_node: FunctionNode) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call) and call_leaf_name(node) == "split" and len(node.args) >= 2
    ]


def _split_size_arguments(function_node: FunctionNode) -> tuple[set[int], set[str]]:
    """The split-size arguments of every `split(...)` call in the function.

    `torch.split` and `Tensor.split` need Python ints for the sizes, so a `.tolist()` feeding one has no
    tensor-only alternative. Returns the ids of directly-passed calls and the names of locals holding one.
    """
    call_ids: set[int] = set()
    names: set[str] = set()
    for split_call in _split_calls(function_node):
        for argument in split_call.args[1:] + [keyword.value for keyword in split_call.keywords]:
            if isinstance(argument, ast.Call):
                call_ids.add(id(argument))
            elif isinstance(argument, ast.Name):
                names.add(argument.id)
    return call_ids, names


def _locals_assigned_from(function_node: FunctionNode, call: ast.Call) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign) and node.value is call:
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for function_node in ast.walk(tree):
        if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) or function_node.name != "forward":
            continue
        split_call_ids, split_size_names = _split_size_arguments(function_node)
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Call):
                continue
            method = call_leaf_name(node)
            if method not in MATERIALIZING_METHODS:
                continue
            if method == "tolist" and (
                id(node) in split_call_ids or _locals_assigned_from(function_node, node) & split_size_names
            ):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `.{method}()` inside `forward` materializes a tensor on the host. "
                        "It breaks the dynamo graph and fails `torch.export`. Keep the value as a tensor."
                    ),
                )
            )
    return violations
