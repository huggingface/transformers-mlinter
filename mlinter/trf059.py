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

"""TRF059: routed Experts modules must expose the routing arguments expected by tensor parallelism."""

import ast
from pathlib import Path

from ._helpers import (
    MODELS_ROOT,
    Violation,
    _class_methods,
    _collect_class_bases,
    _has_rule_suppression,
    _model_dir_name,
    _simple_name,
)


RULE_ID = ""  # Set by discovery

_MOE_TP_EXPERTS_STYLE = "moe_tp_experts"
_CANONICAL_SIGNATURE = "(hidden_states, top_k_index, top_k_weights)"


def _dict_contains_moe_tp_experts(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Dict) and any(
        isinstance(value, ast.Constant) and value.value == _MOE_TP_EXPERTS_STYLE for value in node.values
    )


def _tree_declares_moe_tp_experts(tree: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign)) and _dict_contains_moe_tp_experts(node.value)
        for node in ast.walk(tree)
    )


def _model_dirs_with_moe_tp_experts() -> set[str]:
    model_dirs: set[str] = set()
    for pattern in ("configuration_*.py", "modular_*.py"):
        for config_path in MODELS_ROOT.rglob(pattern):
            try:
                source = config_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if _MOE_TP_EXPERTS_STYLE not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if not _tree_declares_moe_tp_experts(tree):
                continue
            model_dir = _model_dir_name(config_path)
            if model_dir is not None:
                model_dirs.add(model_dir)
    return model_dirs


_MOE_TP_MODEL_DIRS: set[str] | None = None


def _uses_moe_tp_experts(file_path: Path) -> bool:
    global _MOE_TP_MODEL_DIRS
    if _MOE_TP_MODEL_DIRS is None:
        _MOE_TP_MODEL_DIRS = _model_dirs_with_moe_tp_experts()
    model_dir = _model_dir_name(file_path)
    return model_dir is not None and model_dir in _MOE_TP_MODEL_DIRS


def _is_routed_experts_class(class_name: str) -> bool:
    return class_name.endswith("Experts") and "SharedExperts" not in class_name


def _inherits_experts_module(
    class_name: str, class_to_bases: dict[str, list[str]], visiting: set[str] | None = None
) -> bool:
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return False
    visiting.add(class_name)

    for base_name in class_to_bases.get(class_name, []):
        simple_base_name = _simple_name(base_name)
        if simple_base_name.endswith("Experts"):
            return True
        if simple_base_name in class_to_bases and _inherits_experts_module(simple_base_name, class_to_bases, visiting):
            return True
    return False


def _forward_positional_params(forward_method: ast.FunctionDef) -> list[str]:
    args = [*forward_method.args.posonlyargs, *forward_method.args.args]
    if args and args[0].arg == "self":
        args = args[1:]
    return [arg.arg for arg in args if arg.arg not in {"self", "cls"}]


def _matches_hidden_param(name: str) -> bool:
    lowered = name.lower()
    return "hidden" in lowered or lowered in {"x", "input"}


def _matches_index_param(name: str) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in ("index", "indices", "idx")):
        return True
    return "expert" in lowered and not any(token in lowered for token in ("weight", "score", "prob"))


def _matches_weight_param(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("weight", "score", "prob", "routing"))


def _validate_routing_signature(params: list[str]) -> str | None:
    if len(params) < 3:
        found = ", ".join(params) if params else "none"
        return f"expected at least 3 positional arguments after self in {_CANONICAL_SIGNATURE}, found {len(params)} ({found})"

    hidden_name, index_name, weight_name = params[:3]
    if not _matches_hidden_param(hidden_name):
        return f"expected arg 1 to be hidden states, found `{hidden_name}`"
    if not _matches_index_param(index_name):
        return f"expected arg 2 to be top-k expert indices, found `{index_name}`"
    if not _matches_weight_param(weight_name):
        return f"expected arg 3 to be top-k routing weights, found `{weight_name}`"
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if not _uses_moe_tp_experts(file_path):
        return []

    violations: list[Violation] = []
    class_to_bases = _collect_class_bases(tree)
    methods_by_class = {node.name: _class_methods(node) for node in tree.body if isinstance(node, ast.ClassDef)}

    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or not _is_routed_experts_class(class_node.name):
            continue

        forward_method = methods_by_class.get(class_node.name, {}).get("forward")
        if forward_method is None:
            if _inherits_experts_module(class_node.name, class_to_bases):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, class_node.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=class_node.lineno,
                    message=(
                        f"{RULE_ID}: {class_node.name} is used with `{_MOE_TP_EXPERTS_STYLE}` but does not define "
                        f"forward{_CANONICAL_SIGNATURE} or inherit it from another Experts module."
                    ),
                )
            )
            continue

        if _has_rule_suppression(source_lines, RULE_ID, forward_method.lineno):
            continue

        params = _forward_positional_params(forward_method)
        error = _validate_routing_signature(params)
        if error is not None:
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=forward_method.lineno,
                    message=(
                        f"{RULE_ID}: {class_node.name}.forward must use {_CANONICAL_SIGNATURE} for "
                        f"`{_MOE_TP_EXPERTS_STYLE}`; {error}."
                    ),
                )
            )

    return violations
