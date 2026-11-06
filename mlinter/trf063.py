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

"""TRF063: a module's top-level imports must not need backends its exported objects do not declare."""

import ast
import re
from pathlib import Path

from ._helpers import GENERATED_FILE_MARKER, Violation, _has_rule_suppression, read_file_head


RULE_ID = ""  # Set by discovery

# Top-level package -> transformers backend name (keys of `BACKENDS_MAPPING`). Core dependencies such as
# `tokenizers` and `numpy` are always installed and never listed.
_PACKAGE_BACKENDS = {
    "Levenshtein": "levenshtein",
    "PIL": "vision",
    "av": "av",
    "bs4": "bs4",
    "cv2": "cv2",
    "datasets": "datasets",
    "decord": "decord",
    "detectron2": "detectron2",
    "essentia": "essentia",
    "faiss": "faiss",
    "g2p_en": "g2p_en",
    "librosa": "librosa",
    "natten": "natten",
    "nltk": "nltk",
    "pandas": "pandas",
    "phonemizer": "phonemizer",
    "pretty_midi": "pretty_midi",
    "pyctcdecode": "pyctcdecode",
    "pytesseract": "pytesseract",
    "rjieba": "rjieba",
    "sacremoses": "sacremoses",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "sklearn": "sklearn",
    "soundfile": "soundfile",
    "timm": "timm",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchcodec": "torchcodec",
    "torchvision": "torchvision",
    "uroman": "uroman",
}

# Backends that cannot be installed without another one.
_IMPLIED_BACKENDS = {
    "natten": {"torch"},
    "timm": {"torch", "torchvision"},
    "torchaudio": {"torch"},
    "torchcodec": {"torch"},
    "torchvision": {"torch"},
}

_GUARD_RE = re.compile(r"^is_(\w+?)(?:_v2)?_available$")
_TORCHVISION_BACKEND_SUBCLASS = re.compile(r"^class\s+\w+\s*\([^)]*\bTorchvisionBackend\b", re.MULTILINE)
_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

# (backend, reason) pairs, one per backend, where reason says which import pulled it in.
Requirements = dict[str, str]


def _closure(backends: set[str]) -> set[str]:
    result = set(backends)
    for backend in backends:
        result |= _IMPLIED_BACKENDS.get(backend, set())
    return result


def _base_requirements(module_name: str, source: str) -> set[str]:
    """Mirror of `BASE_FILE_REQUIREMENTS` in transformers' `import_utils.py`."""
    if "modeling_" in module_name:
        return {"torch"}
    if "image_processing_" in module_name and _TORCHVISION_BACKEND_SUBCLASS.search(source):
        return {"vision", "torch", "torchvision"}
    if "image_processing_" in module_name:
        return {"vision"}
    if "video_processing_" in module_name:
        return {"vision", "torch", "torchvision"}
    if "generation_" in module_name:
        return {"torch"}
    return set()


def _string_tuple(node: ast.AST) -> set[str] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    values = set()
    for elt in node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            return None
        # Versioned backends such as "torch>=2.4" still need the package itself.
        values.add(re.split(r"[=<>!~]", elt.value, maxsplit=1)[0])
    return values


def _requires_backends(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    for decorator in node.decorator_list:
        if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)):
            continue
        if decorator.func.id != "requires":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "backends":
                return _string_tuple(keyword.value) or set()
    return set()


def _exported_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            values = _string_tuple(node.value)
            return values or set()
    return set()


def _declared_backends(tree: ast.Module, file_path: Path, source: str) -> set[str]:
    """The fewest backends under which transformers' lazy module will import this file.

    Every exported object is importable once its `@requires` backends plus the file's base requirements are
    installed, so the weakest exported object decides the environment the module must import cleanly in.
    """
    base = _base_requirements(file_path.stem, source)
    exported = _exported_names(tree)
    per_object = [
        _requires_backends(node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in exported
    ]
    if not per_object:
        return _closure(base)
    return _closure(base | set.intersection(*per_object))


def _guard_backends(test: ast.AST) -> set[str] | None:
    """Backends an `if` test proves available, or None when the test is not a pure availability guard."""
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and not test.args:
        match = _GUARD_RE.match(test.func.id)
        return {match.group(1)} if match else None
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        backends: set[str] = set()
        for value in test.values:
            guarded = _guard_backends(value)
            if guarded is None:
                return None
            backends |= guarded
        return backends
    return None


def _is_type_checking(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(name, ast.Name) and name.id in _IMPORT_ERRORS for name in names)


def _iter_module_imports(body: list[ast.stmt], guard: frozenset[str] = frozenset()):
    """Yield `(import_node, guard)` for imports that run when the module is imported.

    Function and class bodies run later, `TYPE_CHECKING` blocks never, and imports wrapped in a `try` that catches
    `ImportError` are optional by construction.
    """
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, guard
        elif isinstance(node, ast.If):
            if _is_type_checking(node.test):
                yield from _iter_module_imports(node.orelse, guard)
                continue
            guarded = _guard_backends(node.test)
            yield from _iter_module_imports(node.body, guard | guarded if guarded else guard)
            if (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and (negated := _guard_backends(node.test.operand))
            ):
                yield from _iter_module_imports(node.orelse, guard | negated)
            else:
                yield from _iter_module_imports(node.orelse, guard)
        elif isinstance(node, ast.Try):
            if any(_catches_import_error(handler) for handler in node.handlers):
                continue
            yield from _iter_module_imports(node.body, guard)
            yield from _iter_module_imports(node.orelse, guard)
            yield from _iter_module_imports(node.finalbody, guard)


def _resolve_relative(file_path: Path, node: ast.ImportFrom) -> list[Path]:
    """Model-local files a relative import loads; imports of library modules (`from ...utils`) are trusted."""
    if node.level == 0 or node.level > 2:
        return []
    package_dir = file_path.parent if node.level == 1 else file_path.parent.parent
    targets = []
    if node.module:
        target = package_dir.joinpath(*node.module.split("."))
        targets.append(target.with_suffix(".py"))
    else:
        # `from . import image_processing_acme`
        targets.extend(package_dir / f"{alias.name}.py" for alias in node.names)
    return [target for target in targets if target.is_file()]


def _import_label(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return f"import {', '.join(alias.name for alias in node.names)}"
    return f"from {'.' * node.level}{node.module or ''} import ..."


def _module_requirements(file_path: Path, memo: dict[Path, Requirements], stack: set[Path]) -> Requirements:
    """Backends a model-local module needs just to be imported, with the import that pulls each one in."""
    file_path = file_path.resolve()
    if file_path in memo:
        return memo[file_path]
    if file_path in stack:
        return {}
    stack.add(file_path)
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        stack.discard(file_path)
        return {}

    requirements: Requirements = {}
    for node, guard in _iter_module_imports(tree.body):
        if guard:
            # A guarded import is the imported module's own business; TRF063 checks it when linting that file.
            continue
        for backend, reason in _import_requirements(file_path, node, memo, stack).items():
            requirements.setdefault(backend, reason)
    stack.discard(file_path)
    memo[file_path] = requirements
    return requirements


def _import_requirements(
    file_path: Path, node: ast.Import | ast.ImportFrom, memo: dict[Path, Requirements], stack: set[Path]
) -> Requirements:
    requirements: Requirements = {}
    location = f"{file_path.name}:{node.lineno} `{_import_label(node)}`"
    if isinstance(node, ast.Import):
        packages = [alias.name.split(".")[0] for alias in node.names]
    elif node.level == 0 and node.module:
        packages = [node.module.split(".")[0]]
    else:
        packages = []
    for package in packages:
        backend = _PACKAGE_BACKENDS.get(package)
        if backend is not None:
            for implied in sorted(_closure({backend})):
                requirements.setdefault(implied, location)
    if isinstance(node, ast.ImportFrom):
        for target in _resolve_relative(file_path, node):
            for backend, reason in _module_requirements(target, memo, stack).items():
                requirements.setdefault(backend, f"{location} -> {reason}")
    return requirements


def _check_tree(
    tree: ast.Module, file_path: Path, source: str
) -> list[tuple[ast.ImportFrom, set[str], set[str], Requirements]]:
    declared = _declared_backends(tree, file_path, source)
    memo: dict[Path, Requirements] = {}
    problems = []
    for node, guard in _iter_module_imports(tree.body):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            # Third-party imports at the top of a file are covered by the file's base requirements or by
            # `@requires`; only model-local imports hide a backend behind another file.
            continue
        available = _closure(declared | guard)
        requirements = _import_requirements(file_path, node, memo, {file_path.resolve()})
        missing = {backend for backend in requirements if backend not in available}
        if missing:
            problems.append((node, missing, available, requirements))
    return problems


def _message(
    import_node: ast.ImportFrom, missing: set[str], available: set[str], requirements: Requirements, where: str
) -> str:
    missing_list = ", ".join(sorted(missing))
    available_list = ", ".join(sorted(available)) or "no optional backend"
    first = sorted(missing)[0]
    return (
        f"{RULE_ID}: {where}`{_import_label(import_node)}` needs {missing_list} at import time "
        f"({requirements[first]}), but this module is imported with only {available_list} installed. "
        "Import from a module with matching backends (e.g. the `*_pil` image processor), guard the import with "
        f"`is_{first}_available()`, or declare the backend in `@requires(backends=...)`."
    )


def _generated_siblings(file_path: Path) -> list[Path]:
    siblings = []
    for sibling in sorted(file_path.parent.glob("*.py")):
        if sibling == file_path:
            continue
        head = read_file_head(sibling)
        if head is not None and GENERATED_FILE_MARKER in head and file_path.name in head:
            siblings.append(sibling)
    return siblings


def _modular_anchor(tree: ast.Module, import_node: ast.ImportFrom, generated_tree: ast.Module) -> int:
    """Line in the modular file to report a generated-file problem on.

    The same import when the modular file has it, else the first class the generated file exports, else the top.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == import_node.level and node.module == import_node.module:
            return node.lineno
    exported = _exported_names(generated_tree)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in exported:
            return node.lineno
    return 1


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    if file_path.name.startswith("modular_"):
        # Generated files are not linted, so check what this modular file generates and report it here.
        for generated in _generated_siblings(file_path):
            try:
                generated_source = generated.read_text(encoding="utf-8")
                generated_tree = ast.parse(generated_source)
            except (OSError, SyntaxError):
                continue
            for import_node, missing, available, requirements in _check_tree(
                generated_tree, generated, generated_source
            ):
                line_number = _modular_anchor(tree, import_node, generated_tree)
                if _has_rule_suppression(source_lines, RULE_ID, line_number):
                    continue
                where = f"generated {generated.name}:{import_node.lineno} "
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=line_number,
                        message=_message(import_node, missing, available, requirements, where),
                    )
                )
        return violations

    source = "\n".join(source_lines)
    for import_node, missing, available, requirements in _check_tree(tree, file_path, source):
        if _has_rule_suppression(source_lines, RULE_ID, import_node.lineno):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=import_node.lineno,
                message=_message(import_node, missing, available, requirements, ""),
            )
        )
    return violations
