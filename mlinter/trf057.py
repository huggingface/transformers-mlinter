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

"""TRF057: Public model, config, output and processor classes and their public methods must carry @auto_docstring."""

import ast
import re
from pathlib import Path

from ._helpers import (
    GENERATED_FILE_MARKER,
    Violation,
    _has_rule_suppression,
    _simple_name,
    full_name,
    is_exempt_by_cutoff,
    read_file_head,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

MODEL_KIND = "model"
IMAGE_PROCESSOR_KIND = "image processor"
PROCESSOR_KIND = "processor"
CONFIG_KIND = "config"
OUTPUT_KIND = "model output"

# Public methods requiring auto_docstring.
METHODS_BY_KIND = {
    MODEL_KIND: frozenset(
        {
            "forward",
            "get_image_features",
            "get_video_features",
            "get_audio_features",
            "get_text_features",
        }
    ),
    IMAGE_PROCESSOR_KIND: frozenset({"preprocess"}),
    PROCESSOR_KIND: frozenset({"__call__"}),
    CONFIG_KIND: frozenset(),
    OUTPUT_KIND: frozenset(),
}

# Base classes that identify the kind of a class. Models are matched on the `<Model>PreTrainedModel`
# suffix because every model declares its own base.
MODEL_BASE_SUFFIX = "PreTrainedModel"
KIND_BY_BASE = {
    "BaseImageProcessor": IMAGE_PROCESSOR_KIND,
    "PilBackend": IMAGE_PROCESSOR_KIND,
    "TorchvisionBackend": IMAGE_PROCESSOR_KIND,
    "ProcessorMixin": PROCESSOR_KIND,
    "PreTrainedConfig": CONFIG_KIND,
    "PretrainedConfig": CONFIG_KIND,
}
# Configs and output classes routinely inherit another model's config or output class.
KIND_BY_BASE_SUFFIX = {"Config": CONFIG_KIND, "Output": OUTPUT_KIND}

# Fallback for a processor whose base lives in another model, e.g. `ShieldGemma2Processor(Gemma3Processor)`:
# a processing file only holds processors, so the file name says what the unresolved base would have said.
# There is no such fallback for modeling files, where most classes are inner layers that are out of scope.
KIND_BY_FILE_PREFIX = {
    "processing_": PROCESSOR_KIND,
    "image_processing_": IMAGE_PROCESSOR_KIND,
}

# Model class names that need the decorator on the class itself, i.e. the ones that are part of the public
# API. Config, output and processor classes need no such filter as they are all public.
MODEL_CLASS_NAME_PATTERN = re.compile(r"(?:PreTrainedModel|Model|Backbone|WithProjection)$|For[A-Z]\w*$")

CHECKED_FILE_PREFIXES = ("modeling_", "modular_", "image_processing_", "processing_", "configuration_")
DECORATOR = "auto_docstring"
# Decorators that have to stay below @auto_docstring, in the order they appear in a stack.
INNER_DECORATORS = ("strict", "dataclass")

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
# Every class produced from a given modular file, keyed by that modular file.
_GENERATED_CLASSES_CACHE: dict[Path, dict[str, tuple[Path, ast.ClassDef]]] = {}


def _class_index(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _is_generated_from(path: Path, modular_path: Path) -> bool:
    head = read_file_head(path)
    return head is not None and GENERATED_FILE_MARKER in head and modular_path.name in head


def _generated_classes(modular_path: Path) -> dict[str, tuple[Path, ast.ClassDef]]:
    """Every top-level class the converter produced from `modular_path`, keyed by class name.

    A modular file can emit several files (`modeling_*.py`, `processing_*.py`,
    `image_processing_pil_*.py`, ...) whose names do not always follow from the modular file name, so
    the generation banner is what identifies them.
    """
    if modular_path in _GENERATED_CLASSES_CACHE:
        return _GENERATED_CLASSES_CACHE[modular_path]

    classes: dict[str, tuple[Path, ast.ClassDef]] = {}
    for sibling in sorted(modular_path.parent.glob("*.py")):
        if sibling == modular_path or not _is_generated_from(sibling, modular_path):
            continue
        try:
            tree = ast.parse(sibling.read_text(encoding="utf-8"), filename=str(sibling))
        except (OSError, SyntaxError, ValueError):
            continue
        for name, class_node in _class_index(tree).items():
            classes.setdefault(name, (sibling, class_node))
    _GENERATED_CLASSES_CACHE[modular_path] = classes
    return classes


def _base_names(class_node: ast.ClassDef) -> list[str]:
    names = []
    for base in class_node.bases:
        try:
            names.append(_simple_name(full_name(base)))
        except ValueError:
            continue
    return names


def _is_typed_dict(class_node: ast.ClassDef) -> bool:
    """Whether the class is a kwargs `TypedDict`, which documents its fields in its own docstring."""
    return any(keyword.arg == "total" for keyword in class_node.keywords) or any(
        base_name.endswith("Kwargs") for base_name in _base_names(class_node)
    )


def _class_kind(classes: dict[str, ast.ClassDef], class_node: ast.ClassDef, visited: set[str]) -> str | None:
    """Which kind of documented class this is, following its bases within the file.

    Bases that live in another file are not followed: a class whose kind is only visible through a
    cross-file parent is checked in the generated file, where the converter has inlined the hierarchy.
    """
    if class_node.name in visited:
        return None
    visited.add(class_node.name)

    base_names = _base_names(class_node)
    for base_name in base_names:
        if base_name.endswith(MODEL_BASE_SUFFIX):
            return MODEL_KIND
        if base_name in KIND_BY_BASE:
            return KIND_BY_BASE[base_name]
        for suffix, kind in KIND_BY_BASE_SUFFIX.items():
            if base_name.endswith(suffix) or f"{suffix}With" in base_name:
                return kind

    for base_name in base_names:
        base_node = classes.get(base_name)
        if base_node is None:
            continue
        kind = _class_kind(classes, base_node, visited)
        if kind is not None:
            return kind
    return None


def _decorator_names(function_node: FunctionNode | ast.ClassDef) -> list[str]:
    names = []
    for decorator in function_node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        try:
            names.append(_simple_name(full_name(target)))
        except ValueError:
            continue
    return names


def _find_method(class_node: ast.ClassDef, method_name: str) -> FunctionNode | None:
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
            return item
    return None


DecoratedNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _targets(class_node: ast.ClassDef, kind: str) -> list[tuple[str, DecoratedNode]]:
    """Target nodes we expect to be decorated with @auto_docstring"""
    if not class_node.bases or _is_typed_dict(class_node):
        return []

    base_names = _base_names(class_node)
    # The `Generic*` task mixins of `modeling_layers.py` are decorated themselves and their subclasses
    # are empty-bodied, so the convention is to leave those undecorated.
    inherits_class_doc = any(base_name.startswith("Generic") for base_name in base_names)

    targets: list[tuple[str, DecoratedNode]] = []
    if not inherits_class_doc and (kind != MODEL_KIND or MODEL_CLASS_NAME_PATTERN.search(class_node.name)):
        targets.append((class_node.name, class_node))
    targets += [
        (f"{class_node.name}.{method.name}", method)
        for method in class_node.body
        if isinstance(method, FunctionNode) and method.name in METHODS_BY_KIND[kind]
    ]
    return targets


def _subject(node: DecoratedNode) -> str:
    return "class" if isinstance(node, ast.ClassDef) else "method"


def _missing_documentation(node: DecoratedNode) -> str:
    if isinstance(node, ast.ClassDef):
        return "so its intro and the description of its parameters are not generated"
    return "so its arguments, return value and usage example are left undocumented"


def _placement_hint(*decorator_names: list[str]) -> str:
    """Where the decorator goes: innermost, except above `@strict` and `@dataclass`.

    Decorators are applied bottom-up, and both of those have to run first so that `@auto_docstring`
    sees the finished class: `@dataclass` synthesizes the `__init__` that gets documented, so it must
    not be the parent's (see TRF017), and `@strict` rewrites the fields that `@auto_docstring` reads.
    The stack is `@auto_docstring` -> `@strict` -> `@dataclass`, so the outermost of the two present
    is what the new decorator goes above.
    """
    declared = {name for names in decorator_names for name in names}
    for inner_decorator in INNER_DECORATORS:
        if inner_decorator in declared:
            return f"Add @{DECORATOR} above @{inner_decorator}"
    return f"Add @{DECORATOR} as the innermost decorator"


def _check_source_file(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    """Check a non-modular file."""
    classes = _class_index(tree)
    fallback_kind = next(
        (kind for prefix, kind in KIND_BY_FILE_PREFIX.items() if file_path.name.startswith(prefix)), None
    )
    violations: list[Violation] = []
    for class_node in classes.values():
        kind = _class_kind(classes, class_node, set()) or fallback_kind
        if kind is None:
            continue
        for label, node in _targets(class_node, kind):
            declared = _decorator_names(node)
            if DECORATOR in declared:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{label}` is a public {_subject(node)} without @{DECORATOR}, "
                        f"{_missing_documentation(node)}. {_placement_hint(declared)}."
                    ),
                )
            )
    return violations


def _check_modular_file(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    """Check a modular file against the files generated from it.

    A modular class or method that declares no decorator inherits the parent's decorators.
    One that declares any decorator drops the parent's whole set.
    """
    generated_classes = _generated_classes(file_path)
    classes_by_name = {name: node for name, (_, node) in generated_classes.items()}
    violations: list[Violation] = []
    for class_node in _class_index(tree).values():
        generated = generated_classes.get(class_node.name)
        if generated is None:
            continue
        generated_path, generated_class = generated
        kind = _class_kind(classes_by_name, generated_class, set())
        if kind is None:
            continue

        for label, node in _targets(generated_class, kind):
            source_node = class_node if isinstance(node, ast.ClassDef) else _find_method(class_node, node.name)
            if source_node is None:
                continue
            declared = _decorator_names(source_node)
            generated_decorators = _decorator_names(node)
            if DECORATOR in declared or DECORATOR in generated_decorators:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, source_node.lineno):
                continue

            noun = _subject(node)
            if declared:
                cause = (
                    f"Declaring {', '.join(f'@{name}' for name in declared)} here drops every decorator of the "
                    f"parent {noun}"
                )
            else:
                cause = f"No parent {noun} it could inherit the decorator from declares it"
            hint = _placement_hint(declared, generated_decorators)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=source_node.lineno,
                    message=(
                        f"{RULE_ID}: `{label}` in {generated_path.name} has no @{DECORATOR}. {cause}. {hint} here."
                    ),
                )
            )
    return violations


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(CHECKED_FILE_PREFIXES):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    if file_path.name.startswith("modular_"):
        return _check_modular_file(tree, file_path, source_lines)
    return _check_source_file(tree, file_path, source_lines)
