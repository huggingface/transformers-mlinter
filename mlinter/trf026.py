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

"""TRF026: A module whose forward only delegates to its single submodule adds nothing; inline it."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _class_methods,
    _collect_class_bases,
    _has_rule_suppression,
    _inherits_pretrained_model,
    _simple_name,
    is_exempt_by_cutoff,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Bases that are known not to be models. Everything under `torch.nn` is a plain layer, and
# GradientCheckpointingLayer is the library's own base for decoder layers.
_PLAIN_BASES = {"GradientCheckpointingLayer"}


def _is_plain_base(base_name: str) -> bool:
    return base_name.startswith(("nn.", "torch.nn.")) or base_name in _PLAIN_BASES


def _bases_are_known(class_name: str, class_to_bases: dict[str, list[str]], visiting: set[str] | None = None) -> bool:
    """Whether every base of *class_name* is resolvable from this file alone.

    A modular file subclasses another model's class by import (`class AcmeModel(LlamaModel)`), and
    `class_to_bases` only indexes the file under analysis, so such a base resolves to nothing and
    `_inherits_pretrained_model` cannot tell a PreTrainedModel from a plain block. Treating that as
    "not a model" would flag public model classes, so an unresolvable base means hands off.
    """
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return True
    visiting.add(class_name)

    for base_name in class_to_bases.get(class_name, []):
        if _is_plain_base(base_name):
            continue
        simple_base_name = _simple_name(base_name)
        if simple_base_name not in class_to_bases:
            return False
        if not _bases_are_known(simple_base_name, class_to_bases, visiting):
            return False
    return True


def _self_attribute_targets(function_node: ast.FunctionDef) -> list[str]:
    """Return the names of every `self.<name> = ...` assignment in the function, in source order."""
    names: list[str] = []
    for node in ast.walk(function_node):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.append(target.attr)
    return names


def _effective_body(function_node: ast.FunctionDef) -> list[ast.stmt]:
    """The function body with a leading docstring dropped."""
    body = function_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def _delegated_attribute(function_node: ast.FunctionDef) -> str | None:
    """If the body is exactly `return self.<attr>(...)`, return `<attr>`."""
    body = _effective_body(function_node)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    value = body[0].value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        return func.attr
    return None


def _contains_super_call(function_node: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
        for node in ast.walk(function_node)
    )


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    class_to_bases = _collect_class_bases(tree)
    violations: list[Violation] = []

    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        # PreTrainedModel subclasses are public API: they exist for `from_pretrained`, auto classes and
        # checkpoint layout even when the forward only delegates.
        if _inherits_pretrained_model(class_node.name, class_to_bases):
            continue
        # `class AcmeModel(LlamaModel)` in a modular file is a PreTrainedModel, but the base is
        # imported so the check above cannot see it. Skip anything whose bases do not resolve here.
        if not _bases_are_known(class_node.name, class_to_bases):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, class_node.lineno):
            continue

        methods = _class_methods(class_node)
        init_node, forward_node = methods.get("__init__"), methods.get("forward")
        if init_node is None or forward_node is None:
            continue
        if _contains_super_call(forward_node):
            continue
        # A class carrying anything besides __init__ and forward has behaviour of its own.
        if set(methods) - {"__init__", "forward"}:
            continue

        assigned = _self_attribute_targets(init_node)
        if len(assigned) != 1:
            continue

        delegated = _delegated_attribute(forward_node)
        if delegated is None or delegated != assigned[0]:
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=class_node.lineno,
                message=(
                    f"{RULE_ID}: `{class_node.name}` only forwards to `self.{delegated}`. "
                    "Drop the wrapper and use the inner module where it is called."
                ),
            )
        )

    return violations
