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

"""TRF021: Scalar tensors must be filled on-device with torch.full((), ...), not copied with torch.tensor(...)."""

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ._helpers import (
    Violation,
    _collect_class_bases,
    _find_config_file,
    _get_class_assignments,
    _has_rule_suppression,
    _parse_config_classes,
    _resolve_config_class_name_from_modeling_class,
    _resolve_target_config_class_name,
    full_name,
)


RULE_ID = ""  # Set by discovery

# Construction-time methods never run inside a CUDA graph capture region, so a host->device copy
# there is harmless.
_INIT_METHOD_NAMES = {"__init__", "__post_init__", "_init_weights", "post_init"}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_SCALAR_ANNOTATION_NAMES = {"int", "float", "bool"}
# Identifiers that carry no shape information and can be dropped before classifying an annotation.
_ANNOTATION_NOISE = {"none", "optional", "union", "typing"}

# torch.finfo(...) / torch.iinfo(...) expose Python scalars.
_INFO_FACTORY_NAMES = {"finfo", "iinfo"}
_INFO_SCALAR_FIELDS = {"min", "max", "eps", "tiny", "smallest_normal", "resolution", "bits"}

# Builtins that always return a Python scalar.
_SCALAR_BUILTIN_NAMES = {"abs", "bool", "float", "int", "len", "ord", "round"}


@dataclass
class _Scope:
    """Everything needed to decide whether an expression is a Python scalar at a given call site."""

    config_fields: dict[str, ast.AST] = field(default_factory=dict)
    config_attribute_map: dict[str, str] = field(default_factory=dict)
    self_attributes: dict[str, ast.AST] = field(default_factory=dict)
    local_assignments: dict[str, ast.AST] = field(default_factory=dict)


def _iter_body(node: ast.AST):
    """Walk *node* without descending into nested function or class definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _iter_body(child)


def _iter_functions(node: ast.AST, class_node: ast.ClassDef | None = None):
    """Yield ``(enclosing_class_or_None, function)`` for every function defined under *node*."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _iter_functions(child, child)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield class_node, child
            yield from _iter_functions(child, class_node)
        else:
            yield from _iter_functions(child, class_node)


def _is_torch_tensor_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tensor"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _device_keyword(call: ast.Call) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == "device":
            return keyword.value
    return None


def _is_cpu_device_string(value: str) -> bool:
    """Whether a PyTorch device string names the host, i.e. ``cpu`` or an indexed ``cpu:0``.

    Matched on the device type alone rather than as a substring, so a custom backend whose name
    merely contains "cpu" is not mistaken for the host.
    """
    return value.strip().lower().split(":")[0] == "cpu"


def _is_cpu_device(node: ast.AST) -> bool:
    """Whether *node* pins the tensor to the host, in which case there is no host->device copy."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _is_cpu_device_string(node.value)
    # torch.device("cpu"), torch.device("cpu", 0), torch.device(type="cpu")
    if isinstance(node, ast.Call):
        arguments = list(node.args) + [keyword.value for keyword in node.keywords if keyword.arg == "type"]
        return any(
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and _is_cpu_device_string(argument.value)
            for argument in arguments
        )
    return False


def _annotation_is_scalar(annotation: ast.AST) -> bool:
    """Whether a config field annotation describes a plain Python scalar.

    Anything that could also be a sequence (e.g. ``int | list[int] | None`` on ``eos_token_id``) is
    rejected, because ``torch.full((), ...)`` is only a valid rewrite for 0-d tensors.
    """
    try:
        text = ast.unparse(annotation)
    except Exception:
        return False
    names = {name.lower() for name in _IDENTIFIER.findall(text)} - _ANNOTATION_NOISE
    return bool(names) and names <= _SCALAR_ANNOTATION_NAMES


def _class_bases_in_file(config_classes: dict[str, ast.ClassDef], class_name: str) -> list[str]:
    class_node = config_classes.get(class_name)
    if class_node is None:
        return []
    names = []
    for base in class_node.bases:
        try:
            base_name = full_name(base).split(".")[-1]
        except ValueError:
            continue
        if base_name in config_classes:
            names.append(base_name)
    return names


def _config_field_annotations(
    config_classes: dict[str, ast.ClassDef], class_name: str, visiting: set[str] | None = None
) -> dict[str, ast.AST]:
    """Annotated fields of a config class, including those inherited from classes in the same file."""
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return {}
    visiting.add(class_name)

    annotations: dict[str, ast.AST] = {}
    for base_name in _class_bases_in_file(config_classes, class_name):
        annotations.update(_config_field_annotations(config_classes, base_name, visiting))

    class_node = config_classes.get(class_name)
    if class_node is not None:
        for item in class_node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                annotations[item.target.id] = item.annotation
    return annotations


def _config_attribute_map(
    config_classes: dict[str, ast.ClassDef], class_name: str, visiting: set[str] | None = None
) -> dict[str, str]:
    """The ``attribute_map`` renames of a config class, so ``config.image_token_id`` can be followed
    to the field it actually aliases (e.g. ``image_token_index``)."""
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return {}
    visiting.add(class_name)

    attribute_map: dict[str, str] = {}
    for base_name in _class_bases_in_file(config_classes, class_name):
        attribute_map.update(_config_attribute_map(config_classes, base_name, visiting))

    class_node = config_classes.get(class_name)
    if class_node is None:
        return attribute_map

    value = _get_class_assignments(class_node).get("attribute_map")
    if isinstance(value, ast.Dict):
        for key, mapped in zip(value.keys, value.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(mapped, ast.Constant)
                and isinstance(mapped.value, str)
            ):
                attribute_map[key.value] = mapped.value
    return attribute_map


def _config_field_is_scalar(scope: _Scope, name: str) -> bool:
    annotation = scope.config_fields.get(scope.config_attribute_map.get(name, name))
    return annotation is not None and _annotation_is_scalar(annotation)


def _is_scalar(node: ast.AST, scope: _Scope, seen: frozenset[int] = frozenset()) -> bool:
    """Whether *node* provably evaluates to a Python scalar. Unknown expressions return False."""
    if id(node) in seen:
        return False
    seen = seen | {id(node)}

    if isinstance(node, ast.Constant):
        # bool is a subclass of int, complex/str/bytes/None are not scalars we can rewrite.
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp):
        return _is_scalar(node.operand, scope, seen)
    if isinstance(node, ast.BinOp):
        return _is_scalar(node.left, scope, seen) and _is_scalar(node.right, scope, seen)
    if isinstance(node, ast.IfExp):
        return _is_scalar(node.body, scope, seen) and _is_scalar(node.orelse, scope, seen)
    if isinstance(node, ast.Call):
        return _is_scalar_call(node)
    if isinstance(node, ast.Attribute):
        return _is_scalar_attribute(node, scope, seen)
    if isinstance(node, ast.Name):
        assigned = scope.local_assignments.get(node.id)
        return assigned is not None and _is_scalar(assigned, scope, seen)
    return False


def _is_scalar_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in _SCALAR_BUILTIN_NAMES:
        return True
    # math.log(...), math.sqrt(...), ...
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math"


def _is_scalar_attribute(node: ast.Attribute, scope: _Scope, seen: frozenset[int]) -> bool:
    if node.attr in _INFO_SCALAR_FIELDS and isinstance(node.value, ast.Call):
        factory = node.value.func
        factory_name = factory.attr if isinstance(factory, ast.Attribute) else getattr(factory, "id", None)
        if factory_name in _INFO_FACTORY_NAMES:
            return True

    try:
        dotted = full_name(node)
    except ValueError:
        return False

    # `self.config.<field>` / `config.<field>`, resolved against the companion configuration file.
    # Deeper chains such as `self.config.text_config.<field>` are deliberately not resolved.
    if dotted in (f"self.config.{node.attr}", f"config.{node.attr}"):
        return _config_field_is_scalar(scope, node.attr)

    if dotted == f"self.{node.attr}":
        assigned = scope.self_attributes.get(node.attr)
        return assigned is not None and _is_scalar(assigned, scope, seen)

    return False


def _unique_assignments(nodes, target_matches, target_key) -> dict[str, ast.AST]:
    """Map name -> assigned value, keeping only names bound exactly once (rebinding is ambiguous)."""
    values: dict[str, ast.AST] = {}
    counts: Counter[str] = Counter()
    for node in nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        for target in targets:
            if not target_matches(target):
                continue
            name = target_key(target)
            counts[name] += 1
            values[name] = node.value
    return {name: value for name, value in values.items() if counts[name] == 1}


def _self_attributes(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    """`self.<name> = <value>` bindings made anywhere in the class body."""
    nodes = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.extend(_iter_body(item))
    return _unique_assignments(
        nodes,
        lambda t: isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self",
        lambda t: t.attr,
    )


def _local_assignments(function_node: ast.AST) -> dict[str, ast.AST]:
    return _unique_assignments(
        _iter_body(function_node),
        lambda t: isinstance(t, ast.Name),
        lambda t: t.id,
    )


def _resolve_config_context(
    tree: ast.Module, file_path: Path, class_node: ast.ClassDef
) -> tuple[dict[str, ast.AST], dict[str, str]]:
    """Resolve the config class *class_node* targets and return its fields and attribute_map."""
    config_path = _find_config_file(file_path)
    if config_path is None:
        return {}, {}
    config_classes = _parse_config_classes(config_path)
    if not config_classes:
        return {}, {}

    class_to_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    class_to_assignments = {name: _get_class_assignments(node) for name, node in class_to_nodes.items()}
    declared = _resolve_config_class_name_from_modeling_class(
        class_node.name, _collect_class_bases(tree), class_to_assignments, class_to_nodes
    )
    target = _resolve_target_config_class_name(config_classes, class_node.name, declared)
    if target is None:
        return {}, {}
    return _config_field_annotations(config_classes, target), _config_attribute_map(config_classes, target)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []

    violations: list[Violation] = []
    config_cache: dict[str, tuple[dict[str, ast.AST], dict[str, str]]] = {}

    for class_node, function_node in _iter_functions(tree):
        if function_node.name in _INIT_METHOD_NAMES:
            continue

        candidates = [
            node
            for node in _iter_body(function_node)
            if _is_torch_tensor_call(node)
            and node.args
            and (device := _device_keyword(node)) is not None
            and not _is_cpu_device(device)
        ]
        if not candidates:
            continue

        scope = _Scope(local_assignments=_local_assignments(function_node))
        if class_node is not None:
            scope.self_attributes = _self_attributes(class_node)
            if class_node.name not in config_cache:
                config_cache[class_node.name] = _resolve_config_context(tree, file_path, class_node)
            scope.config_fields, scope.config_attribute_map = config_cache[class_node.name]

        for call in candidates:
            value = call.args[0]
            if not _is_scalar(value, scope):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, call.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=call.lineno,
                    message=(
                        f"{RULE_ID}: `torch.tensor({ast.unparse(value)}, ..., device=...)` copies a scalar from "
                        "host to device, which CUDA graph capture forbids. Use "
                        f"`torch.full((), {ast.unparse(value)}, dtype=..., device=...)` to fill the scalar "
                        "directly on-device."
                    ),
                )
            )

    return violations
