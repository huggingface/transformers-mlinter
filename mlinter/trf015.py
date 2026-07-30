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

"""TRF015: Models with non-empty _tied_weights_keys must have tie_word_embeddings in their Config."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _collect_class_bases,
    _find_config_file,
    _get_class_assignments,
    _parse_config_classes,
    _resolve_config_class_name_from_modeling_class,
    _resolve_target_config_class_name,
    _simple_name,
    full_name,
    iter_pretrained_classes,
)


RULE_ID = ""  # Set by discovery

_PRETRAINED_CONFIG_NAMES = {"PreTrainedConfig", "PretrainedConfig"}


def _is_non_empty_collection(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        return len(node.keys) > 0
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return len(node.elts) > 0
    return False


def _class_has_tie_word_embeddings(config_node: ast.ClassDef) -> bool:
    for base in config_node.bases:
        try:
            base_name = _simple_name(full_name(base))
        except ValueError:
            continue
        if base_name not in _PRETRAINED_CONFIG_NAMES and base_name.endswith("Config"):
            return True

    for item in config_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id == "tie_word_embeddings":
                return True
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "tie_word_embeddings":
                    return True
        if isinstance(item, ast.FunctionDef):
            for stmt in ast.walk(item):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "self"
                    and stmt.targets[0].attr == "tie_word_embeddings"
                ):
                    return True
    return False


def _config_has_tie_word_embeddings(
    config_classes: dict[str, ast.ClassDef], model_class_name: str, config_class_name: str | None
) -> bool:
    target_config_name = _resolve_target_config_class_name(config_classes, model_class_name, config_class_name)
    if target_config_name is None:
        return True

    target_config = config_classes.get(target_config_name)
    if target_config is None:
        return True

    return _class_has_tie_word_embeddings(target_config)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []

    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return violations

    classes_with_tied_keys: list[ast.ClassDef] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        assignments = _get_class_assignments(node)
        tied_keys = assignments.get("_tied_weights_keys")
        if tied_keys is not None and _is_non_empty_collection(tied_keys):
            classes_with_tied_keys.append(node)

    if not classes_with_tied_keys:
        return violations

    class_to_bases = _collect_class_bases(tree)
    class_to_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    class_to_assignments = {
        node.name: _get_class_assignments(node) for node in tree.body if isinstance(node, ast.ClassDef)
    }

    config_path = _find_config_file(file_path)
    if config_path is None:
        for node in classes_with_tied_keys:
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: {node.name} defines _tied_weights_keys but no configuration file "
                        f"was found in {file_path.parent}."
                    ),
                )
            )
        return violations

    config_classes = _parse_config_classes(config_path)
    if config_classes is None:
        return violations

    for node in classes_with_tied_keys:
        config_class_name = _resolve_config_class_name_from_modeling_class(
            node.name, class_to_bases, class_to_assignments, class_to_nodes
        )
        target_config_class_name = _resolve_target_config_class_name(config_classes, node.name, config_class_name)
        if target_config_class_name is None:
            continue
        if _config_has_tie_word_embeddings(config_classes, node.name, config_class_name):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {node.name} defines _tied_weights_keys but {config_path.name} maps to "
                    f"{target_config_class_name}, which does not declare tie_word_embeddings. Add a top-level "
                    f"'tie_word_embeddings: bool = ...' field to {target_config_class_name}."
                ),
            )
        )

    return violations
