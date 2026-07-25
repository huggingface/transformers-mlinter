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

"""TRF020: MLA models must isolate the KV LoRA expansion (kv_b_proj) in a dedicated method called by forward()."""

import ast
from pathlib import Path

from ._helpers import (
    MODELS_ROOT,
    Violation,
    _has_rule_suppression,
    _model_dir_name,
)


RULE_ID = ""  # Set by discovery

# The config field that flags a Multi-head Latent Attention (MLA) model.
_KV_LORA_RANK = "kv_lora_rank"


def _references_kv_lora_rank(node: ast.AST) -> bool:
    """Whether *node* reads ``config.kv_lora_rank`` / ``self.kv_lora_rank`` / a bare ``kv_lora_rank``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == _KV_LORA_RANK:
            return True
        if isinstance(sub, ast.Name) and sub.id == _KV_LORA_RANK:
            return True
    return False


def _config_declares_kv_lora_rank(tree: ast.Module) -> bool:
    """Whether the configuration module declares a ``kv_lora_rank`` field (not just a docstring mention)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == _KV_LORA_RANK:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == _KV_LORA_RANK:
                    return True
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == _KV_LORA_RANK
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
    return False


def _mla_model_dirs() -> set[str]:
    """Return the set of model directory names whose configuration declares ``kv_lora_rank``."""
    dirs: set[str] = set()
    for config_path in MODELS_ROOT.rglob("configuration_*.py"):
        try:
            source = config_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _KV_LORA_RANK not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if not _config_declares_kv_lora_rank(tree):
            continue
        model_dir = _model_dir_name(config_path)
        if model_dir is not None:
            dirs.add(model_dir)
    return dirs


_MLA_MODEL_DIRS: set[str] | None = None


def _file_in_mla_model(file_path: Path) -> bool:
    global _MLA_MODEL_DIRS
    if _MLA_MODEL_DIRS is None:
        _MLA_MODEL_DIRS = _mla_model_dirs()
    model_dir = _model_dir_name(file_path)
    if model_dir is None:
        return False
    return model_dir in _MLA_MODEL_DIRS


def _local_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _self_call_nodes(function_node: ast.FunctionDef) -> list[ast.Call]:
    """Return every ``self.<attr>(...)`` call within *function_node*."""
    return [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]


def _is_kv_lora_linear(value: ast.AST) -> bool:
    """Whether *value* is ``nn.Linear(<kv_lora_rank>, ...)`` (input dim is the compressed KV rank)."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Attribute):
        func_name = func.attr
    elif isinstance(func, ast.Name):
        func_name = func.id
    else:
        return False
    if func_name != "Linear" or not value.args:
        return False
    return _references_kv_lora_rank(value.args[0])


def _init_expansion_proj_names(class_node: ast.ClassDef) -> set[str]:
    """Names of ``self.<name>`` projections assigned as ``nn.Linear(kv_lora_rank, ...)`` in ``__init__``."""
    names: set[str] = set()
    init = _local_method(class_node, "__init__")
    if init is None:
        return names
    for stmt in ast.walk(init):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
            and _is_kv_lora_linear(stmt.value)
        ):
            names.add(stmt.targets[0].attr)
    return names


def _method_applies_projection(function_node: ast.FunctionDef, proj_names: set[str]) -> bool:
    return any(call.func.attr in proj_names for call in _self_call_nodes(function_node))


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []
    if not _file_in_mla_model(file_path):
        return []

    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        # Identify the KV LoRA expansion projection(s) structurally: any nn.Linear whose input
        # dimension is kv_lora_rank. This deliberately does not assume the conventional `kv_b_proj`
        # name, so the rule holds for MLA models that name the up-projection differently.
        proj_names = _init_expansion_proj_names(node)
        if not proj_names:
            continue

        forward = _local_method(node, "forward")
        if forward is None:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        # (A) The expansion projection is applied directly inside forward().
        direct_calls = [call for call in _self_call_nodes(forward) if call.func.attr in proj_names]
        if direct_calls:
            call = direct_calls[0]
            if _has_rule_suppression(source_lines, RULE_ID, call.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=call.lineno,
                    message=(
                        f"{RULE_ID}: {node.name}.forward applies the KV LoRA expansion `self.{call.func.attr}` "
                        "directly. In MLA models the expansion must live in a dedicated method (e.g. `expand_kv`) "
                        "that forward calls, so external backends (vLLM/SGLang) can override it and consume the "
                        "compressed KV cache directly."
                    ),
                )
            )
            continue

        # (B) The class declares the projection in its own __init__ but forward does not route the
        # expansion through a dedicated method.
        forward_self_calls = {call.func.attr for call in _self_call_nodes(forward)}
        expansion_methods = {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and item.name != "forward"
            and _method_applies_projection(item, proj_names)
        }
        if expansion_methods & forward_self_calls:
            continue

        representative = sorted(proj_names)[0]
        violations.append(
            Violation(
                file_path=file_path,
                line_number=forward.lineno,
                message=(
                    f"{RULE_ID}: {node.name} declares the KV LoRA expansion `self.{representative}` but forward "
                    "does not call a dedicated expansion method. Move the expansion into a method "
                    "(e.g. `expand_kv(k_nope, k_pe) -> key_states, value_states`) and call it from forward, so "
                    "external backends (vLLM/SGLang) can override it and consume the compressed KV cache directly."
                ),
            )
        )

    return violations
