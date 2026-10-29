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

"""TRF066: declare backend requirements with non-redundant @requires decorators."""

import ast
from pathlib import Path

from ._helpers import GENERATED_FILE_MARKER, Violation, _has_rule_suppression, is_exempt_by_cutoff, read_file_head


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption
_DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _base_requirements(tree: ast.Module, file_path: Path) -> set[str]:
    """Static mirror of transformers.utils.import_utils.BASE_FILE_REQUIREMENTS."""
    name = file_path.stem
    if name.startswith(("modeling_", "generation_")):
        return {"torch"}
    if name.startswith("tokenization_") and name.endswith("_fast"):
        return {"tokenizers"}
    if name.startswith("video_processing_"):
        return {"vision", "torch", "torchvision"}
    if name.startswith("image_processing_"):
        if any(
            isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "TorchvisionBackend" for base in node.bases)
            for node in tree.body
        ):
            return {"vision", "torch", "torchvision"}
        return {"vision"}
    return set()


def _generated_requirements(file_path: Path) -> dict[str, set[str]]:
    """Match modular definitions to the file requirements of their generated companions."""
    requirements: dict[str, set[str]] = {}
    if not file_path.name.startswith("modular_"):
        return requirements
    for sibling in file_path.parent.glob("*.py"):
        head = read_file_head(sibling)
        if head is None or GENERATED_FILE_MARKER not in head or file_path.name not in head:
            continue
        try:
            tree = ast.parse(sibling.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        backends = _base_requirements(tree, sibling)
        for node in tree.body:
            if isinstance(node, _DEFINITIONS):
                # A helper may appear in several generated files: only their common requirements are certain.
                requirements[node.name] = requirements.get(node.name, backends) & backends
    return requirements


def _strings(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List)):
        values = set()
        for item in node.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                return None
            values.add(item.value)
        return values
    return None


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    return ""


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    names = {"requires": "requires", "requires_backends": "requires_backends"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "utils",
            "utils.import_utils",
            "transformers.utils",
            "transformers.utils.import_utils",
        }:
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"transformers.utils", "transformers.utils.import_utils"}:
                    prefix = alias.asname or alias.name
                    for helper in ("requires", "requires_backends"):
                        names[f"{prefix}.{helper}"] = helper

    base = _base_requirements(tree, file_path)
    generated = _generated_requirements(file_path)
    object_requirements = {
        node: generated[node.name] for node in tree.body if isinstance(node, _DEFINITIONS) and node.name in generated
    }
    violations = []

    def report(node: ast.expr, message: str):
        if not _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            violations.append(Violation(file_path, node.lineno, f"{RULE_ID}: {message}"))

    def decorators(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        result = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and names.get(_name(decorator.func)) == "requires":
                for keyword in decorator.keywords:
                    if keyword.arg == "backends" and isinstance(keyword.value, (ast.Tuple, ast.List)):
                        backends = _strings(keyword.value)
                        if backends is not None:
                            result.append((decorator, backends))
        return result

    def visit(node: ast.AST, class_backends: set[str], file_backends: set[str]):
        if isinstance(node, _DEFINITIONS):
            file_backends = object_requirements.get(node, file_backends)
            declared = decorators(node)
            available = file_backends | class_backends if not isinstance(node, ast.ClassDef) else file_backends
            for decorator, backends in declared:
                redundant = backends & available
                if redundant or not backends:
                    detail = ", ".join(f"`{backend}`" for backend in sorted(redundant)) or "an empty backend list"
                    action = (
                        "Remove this `@requires(backends=...)` decorator."
                        if backends <= available
                        else f"Remove {detail} from `@requires(backends=...)`."
                    )
                    report(
                        decorator,
                        f"`{node.name}` repeats requirements already supplied by the file or class ({detail}). "
                        f"{action}",
                    )

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body[1:] if ast.get_docstring(node) is not None else node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
                    call = body[0].value
                    if names.get(_name(call.func)) == "requires_backends":
                        backend_arg = (
                            call.args[1]
                            if len(call.args) == 2
                            else next((kw.value for kw in call.keywords if kw.arg == "backends"), None)
                        )
                        if backend_arg is not None and _strings(backend_arg) is not None:
                            report(
                                call,
                                f"`{node.name}` starts with `requires_backends(...)`. "
                                "Remove it and use a `@requires(backends=...)` decorator instead.",
                            )
                # A nested function is not a method of the enclosing class.
                class_backends = set()
            else:
                class_backends = set().union(*(backends for _, backends in declared))
            for child in node.body:
                visit(child, class_backends, file_backends)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, class_backends, file_backends)

    visit(tree, set(), base)
    return violations
