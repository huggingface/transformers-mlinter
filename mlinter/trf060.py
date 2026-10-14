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

"""TRF060: a sub-model call must not capture every hidden state when only some of them are read."""

import ast
from collections.abc import Iterator
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, call_leaf_name


RULE_ID = ""  # Set by discovery

FLAG = "output_hidden_states"
# Backbones under this decorator return a filtered subset of the stages they compute, so capturing every
# hidden state is how they work, not a leak.
FILTERING_DECORATOR = "filter_output_hidden_states"


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _own_nodes(function_node: FunctionNode) -> Iterator[ast.AST]:
    """Every node in the function body, without descending into nested functions, lambdas or classes."""
    stack: list[ast.AST] = list(function_node.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.append(child)


def _decorator_names(function_node: FunctionNode) -> set[str]:
    names: set[str] = set()
    for decorator in function_node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _forced_kwargs(nodes: list[ast.AST]) -> dict[str, int]:
    """`<name>["output_hidden_states"] = True` assignments, as dict name -> earliest line."""
    forced: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or not _is_true(node.value):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == FLAG
            ):
                name = target.value.id
                forced[name] = min(forced.get(name, node.lineno), node.lineno)
    return forced


def _full_capture_line(call: ast.Call, forced: dict[str, int]) -> int | None:
    """The line that switches every hidden state on for `call`, or None when it does not."""
    for keyword in call.keywords:
        if keyword.arg == FLAG and _is_true(keyword.value):
            return keyword.value.lineno
        if keyword.arg is None and isinstance(keyword.value, ast.Name):
            forced_line = forced.get(keyword.value.id)
            if forced_line is not None and forced_line <= call.lineno:
                return forced_line
    return None


class _Uses:
    def __init__(self) -> None:
        self.indexed = False
        self.whole = False


def _parents(nodes: list[ast.AST]) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in nodes:
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _classify(node: ast.AST, parents: dict[int, ast.AST], uses: _Uses) -> None:
    """Record how one reference to the hidden-states tuple is used."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Subscript) and parent.value is node:
        if isinstance(parent.slice, ast.Slice):
            uses.whole = True
        else:
            uses.indexed = True
    elif isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node:
        uses.whole = True
    elif isinstance(parent, ast.Starred):
        uses.whole = True
    elif isinstance(parent, ast.Call) and node in parent.args and call_leaf_name(parent) != "len":
        # torch.stack(...), torch.cat(...), zip(...), a weighted-sum helper: each consumes the whole tuple.
        uses.whole = True
    # Anything else -- `len(...)`, a `hidden_states=` keyword handing the tuple back in an output, a `return`,
    # an `is None` check -- neither reads a layer nor needs all of them.


def _hidden_states_uses(call: ast.Call, nodes: list[ast.AST], parents: dict[int, ast.AST]) -> _Uses:
    uses = _Uses()

    # `outputs = self.vision_tower(...)` -> every `outputs.hidden_states` after it.
    result_names: set[str] = set()
    parent = parents.get(id(call))
    if isinstance(parent, ast.Assign) and parent.value is call:
        result_names = {target.id for target in parent.targets if isinstance(target, ast.Name)}

    tuple_refs: list[ast.AST] = []
    for node in nodes:
        if not isinstance(node, ast.Attribute) or node.attr != "hidden_states":
            continue
        if node.value is call or (
            isinstance(node.value, ast.Name) and node.value.id in result_names and node.lineno >= call.lineno
        ):
            tuple_refs.append(node)

    # `hidden_states = outputs.hidden_states` -> every later `hidden_states` load is the tuple too.
    aliases: dict[str, int] = {}
    for ref in tuple_refs:
        ref_parent = parents.get(id(ref))
        if isinstance(ref_parent, ast.Assign) and ref_parent.value is ref:
            for target in ref_parent.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = ref.lineno
        else:
            _classify(ref, parents, uses)
    for node in nodes:
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in aliases
            and node.lineno > aliases[node.id]
        ):
            _classify(node, parents, uses)
    return uses


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for function_node in ast.walk(tree):
        if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if FILTERING_DECORATOR in _decorator_names(function_node):
            continue
        nodes = list(_own_nodes(function_node))
        forced = _forced_kwargs(nodes)
        parents = _parents(nodes)
        reported: set[int] = set()
        for call in nodes:
            if not isinstance(call, ast.Call):
                continue
            line = _full_capture_line(call, forced)
            if line is None or line in reported:
                continue
            uses = _hidden_states_uses(call, nodes, parents)
            if not uses.indexed or uses.whole:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, line):
                continue
            reported.add(line)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=line,
                    message=(
                        f"{RULE_ID}: `{FLAG}=True` keeps every layer's hidden states alive, but "
                        f"`{function_node.name}` only reads some of them by index. Request just those layers "
                        "with `hidden_states_to_capture(...)`."
                    ),
                )
            )
    return violations
