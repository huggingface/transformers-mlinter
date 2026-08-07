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

"""TRF022: _no_split_modules entries must name module classes that exist in the model."""

import ast
from pathlib import Path

from ._helpers import Violation, _get_class_assignments, _has_rule_suppression


RULE_ID = ""  # Set by discovery

_ATTRIBUTE = "_no_split_modules"

# `torch.nn.utils.parametrize.register_parametrization` (used by `weight_norm`) swaps a module's
# class for a subclass it builds at runtime, named `"Parametrized" + cls.__name__`. Such a name is
# what `module.__class__.__name__` reports for a parametrized module, so it is a valid
# `_no_split_modules` entry even though no source file ever defines the class.
_DYNAMIC_CLASS_NAME_PREFIX = "Parametrized"


def _is_dynamic_class_name(name: str) -> bool:
    return name.startswith(_DYNAMIC_CLASS_NAME_PREFIX) and len(name) > len(_DYNAMIC_CLASS_NAME_PREFIX)


# Classes that live outside the model directory but are still legitimate entries. A timm backbone is
# built from third-party classes whose names transformers does not control, so the `TimmWrapper*`
# class is the smallest unit a timm-backed model can name -- which is why `timm_wrapper` itself sets
# `_no_split_modules = ["TimmWrapperModel"]`. Models embedding such a backbone name the wrapper for
# the same reason, so those names resolve from any model directory.
_EXTERNAL_CLASS_NAMES = frozenset({"TimmWrapperForImageClassification"})


def _local_class_names(tree: ast.Module) -> set[str]:
    """Every class name defined anywhere in *tree* (including inside ``if`` / ``try`` blocks)."""
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _imported_names(tree: ast.Module) -> set[str]:
    """Names bound by imports, including `TYPE_CHECKING` and try/except guarded ones."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _module_level_aliases(tree: ast.Module) -> set[str]:
    """Module-level ``Name = ...`` targets, which may alias a class defined elsewhere."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# Model directories hold a handful of Python files, so the per-directory class index is cheap; it is
# memoized because a directory is re-scanned for every modeling file it contains.
_MODEL_DIR_CLASS_NAMES: dict[Path, set[str]] = {}


def _model_dir_class_names(file_path: Path) -> set[str]:
    """Class names defined by the sibling modules of *file_path* (same model directory).

    Model packages split implementation across several modules (e.g. `vision.py`, `perceiver.py`,
    `modeling_<name>_fold.py`). A class defined in a sibling module is still part of the same model,
    and `device_map` resolves `_no_split_modules` against runtime class names regardless of which
    module defines them, so those names are accepted.

    This is also what makes `modular_*.py` checkable. A modular file inherits most of its classes
    implicitly, so their names are absent from its own tree, but the `modeling_*.py` file generated
    from it spells every one of them out — and it is a sibling, so it is indexed here.
    """
    model_dir = file_path.parent
    cached = _MODEL_DIR_CLASS_NAMES.get(model_dir)
    if cached is not None:
        return cached

    names: set[str] = set()
    if model_dir.is_dir():
        # The file under analysis is indexed too: its own classes are already resolved locally, and
        # including them keeps this per-directory cache independent of which file populated it.
        for sibling in sorted(model_dir.glob("*.py")):
            try:
                sibling_tree = ast.parse(sibling.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                continue
            names.update(_local_class_names(sibling_tree))

    _MODEL_DIR_CLASS_NAMES[model_dir] = names
    return names


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    # `_no_split_modules` only ever appears on model classes, so skip the configuration and
    # processor files the driver also feeds in. Modular files are checked as well: the classes they
    # inherit implicitly are materialized in the generated `modeling_*.py` sibling, which
    # `_model_dir_class_names` indexes. Reporting there rather than on the generated file matters,
    # because the driver skips generated files (an edit to one is overwritten on regeneration), so
    # a modular-based model would otherwise never be checked at all.
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    # The entry name is carried alongside the node so the string narrowing below survives.
    declared_entries: list[tuple[ast.ClassDef, ast.Constant, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        value = _get_class_assignments(node).get(_ATTRIBUTE)
        if not isinstance(value, (ast.List, ast.Tuple)):
            # `None`, and malformed values in general, are TRF005's concern.
            continue
        for element in value.elts:
            # Non-string and empty entries are reported by TRF005.
            if isinstance(element, ast.Constant) and isinstance(element.value, str) and element.value:
                declared_entries.append((node, element, element.value))

    if not declared_entries:
        return []

    known_names = _local_class_names(tree) | _imported_names(tree) | _module_level_aliases(tree)

    violations: list[Violation] = []
    for class_node, element, name in declared_entries:
        if name in known_names:
            continue
        if _is_dynamic_class_name(name) or name in _EXTERNAL_CLASS_NAMES:
            continue
        if name in _model_dir_class_names(file_path):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, element.lineno):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=element.lineno,
                message=(
                    f"{RULE_ID}: {class_node.name}.{_ATTRIBUTE} lists {name!r}, which is not defined or "
                    f"imported in {file_path.name} and does not exist in the model directory. Either remove the "
                    "entry, or correct it to the name of the layer class of this model. Do not name classes owned "
                    "by a submodel (e.g. a language model or vision tower built through `AutoModel`): `post_init` "
                    "already collects `_no_split_modules` from child submodels automatically."
                ),
            )
        )

    return violations
