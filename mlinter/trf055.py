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

"""TRF055: PreTrainedModel `config` must be an annotation, matched with annotations in `self.__init__`; never `config_class`.
Every `<Model>PreTrainedModel` must declare a valid `config` annotation that cannot be skipped!"""

import ast
from pathlib import Path

from ._helpers import (
    GENERATED_FILE_MARKER,
    Violation,
    _has_rule_suppression,
    _simple_name,
    full_name,
    iter_pretrained_classes,
    read_file_head,
)


RULE_ID = ""  # Set by discovery


def _target_name(item: ast.stmt) -> str | None:
    """Name bound by a simple class-body assignment or annotation, else None."""
    if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
        return item.targets[0].id
    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
        return item.target.id
    return None


def _annotation_name(annotation: ast.expr | None) -> str | None:
    """Simple name of a config annotation (`X`, `mod.X`, or the string "X"); None for anything else."""
    if isinstance(annotation, (ast.Name, ast.Attribute)):
        return _simple_name(full_name(annotation))
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return _annotation_name(ast.parse(annotation.value, mode="eval").body)
        except SyntaxError:
            return None
    return None


def _own_config(node: ast.ClassDef) -> str | None:
    """Config name from a bare `config: X` annotation in the class body, if present."""
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and item.value is None and _target_name(item) == "config":
            return _annotation_name(item.annotation)
    return None


def _effective_config(
    node: ast.ClassDef, classes: dict[str, ast.ClassDef], seen: frozenset = frozenset()
) -> str | None:
    """Own `config` annotation, else the one inherited from a base defined in this file, else None."""
    own = _own_config(node)
    if own is not None:
        return own
    for base in node.bases:
        base_name = _simple_name(full_name(base)) if isinstance(base, (ast.Name, ast.Attribute)) else None
        if base_name and base_name in classes and base_name not in seen:
            inherited = _effective_config(classes[base_name], classes, seen | {node.name})
            if inherited is not None:
                return inherited
    return None


def _init_config_arg(node: ast.ClassDef) -> tuple[ast.FunctionDef, ast.arg] | None:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            args = item.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                if arg.arg == "config":
                    return item, arg
    return None


def _generated_modeling_files(modular_path: Path) -> list[Path]:
    """The `modeling_*.py` files the converter produced from `modular_path`, identified by their banner."""
    generated = []
    for sibling in sorted(modular_path.parent.glob("modeling_*.py")):
        head = read_file_head(sibling)
        if head is not None and GENERATED_FILE_MARKER in head and modular_path.name in head:
            generated.append(sibling)
    return generated


def _check_modular(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    """Check what a modular file generates, reporting on the class in the modular file.

    The class hierarchy and the `config` annotations are only complete in the generated modeling file, since a
    modular class often inherits them from another model. A generated class the modular file does not define was
    copied from another model, whose own files report it.
    """
    modular_classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    violations: list[Violation] = []
    for generated in _generated_modeling_files(file_path):
        try:
            generated_source = generated.read_text(encoding="utf-8")
            generated_tree = ast.parse(generated_source)
        except (OSError, SyntaxError):
            continue
        for class_name, line, message in _problems(generated_tree, generated_source.splitlines()):
            modular_class = modular_classes.get(class_name)
            if modular_class is None or _has_rule_suppression(source_lines, RULE_ID, modular_class.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=modular_class.lineno,
                    message=f"{RULE_ID}: generated {generated.name}:{line}: {message}",
                )
            )
    return violations


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if file_path.name.startswith("modular_"):
        return _check_modular(tree, file_path, source_lines)
    if not file_path.name.startswith("modeling_"):
        return []
    return [
        Violation(file_path=file_path, line_number=line, message=f"{RULE_ID}: {message}")
        for _, line, message in _problems(tree, source_lines)
    ]


def _problems(tree: ast.Module, source_lines: list[str]) -> list[tuple[str, int, str]]:
    """`(class name, line, message)` for every problem in a modeling file, suppressions already applied."""
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    problems: list[tuple[str, int, str]] = []
    class_name = ""

    def flag(line: int, message: str) -> None:
        if not _has_rule_suppression(source_lines, RULE_ID, line):
            problems.append((class_name, line, message))

    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        class_name = node.name
        has_config_annotation = False

        # 1. Class body: `config_class` is forbidden, `config` must be a bare annotation.
        for item in node.body:
            name = _target_name(item)
            if name == "config_class":
                flag(
                    item.lineno,
                    f"`{node.name}` must not define `config_class`. It is derived from the `config` annotation "
                    f"by `PreTrainedModel.__init_subclass__`. Use `config: <Model>Config` instead.",
                )
            elif name == "config":
                has_config_annotation = True
                if isinstance(item, ast.AnnAssign) and item.value is None:
                    continue
                value = item.value
                config_name = (
                    _simple_name(full_name(value)) if isinstance(value, (ast.Name, ast.Attribute)) else "<Model>Config"
                )
                flag(
                    item.lineno,
                    f"`{node.name}.config = {config_name}` assigns a class attribute that shadows `config_class` "
                    f"resolution, causing `{node.name}.config_class` to return the wrong config. "
                    f"Use an annotation instead: `config: {config_name}`.",
                )

        # 2. The base <Model>PreTrainedModel must declare `config`.
        if node.name.endswith("PreTrainedModel") and not has_config_annotation:
            flag(node.lineno, f"`{node.name}` must declare a `config: <Model>Config` annotation.")

        # 3. `__init__(config: X)` must be annotated and agree with the effective config.
        init_and_arg = _init_config_arg(node)
        if init_and_arg is None:
            continue  # no __init__ taking `config`: the parent's is inherited
        init, arg = init_and_arg
        effective = _effective_config(node, classes)
        init_config = _annotation_name(arg.annotation)

        if arg.annotation is None:
            flag(
                init.lineno,
                f"`{node.name}.__init__` must annotate its config argument: `config: {effective or '<Model>Config'}`, "
                f"matching the config attribute it inherits or overrides.",
            )
        elif init_config is None:
            continue  # exotic annotation (Optional[...], union, ...): not our business
        elif effective is None:
            flag(
                arg.lineno,
                f"`{node.name}.__init__` takes `config: {init_config}` but no `config` annotation is declared or "
                f"inherited. Declare `config: {init_config}` in the class body.",
            )
        elif init_config != effective:
            where = "overrides" if _own_config(node) is not None else "inherits"
            flag(
                arg.lineno,
                f"`{node.name}.__init__` takes `config: {init_config}` but the class {where} `config: {effective}`. "
                f"Make them match; add or update `config: {init_config}` in the class body if that is intended.",
            )

    return problems
