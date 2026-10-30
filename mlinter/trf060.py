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

import ast
import difflib
from pathlib import Path

from ._helpers import Violation


"""`PreTrainedModel._no_split_modules` must only name classes that exist."""


RULE_ID = ""  # Set by discovery

GENERATED_MARKER = "automatically generated from"
# torch.nn.utils.parametrize builds classes named "Parametrized" + <original name> at runtime,
# so they can never be found in a source file. Check `hubert` in transformers source code for an ex
RUNTIME_CLASS_PREFIXES = ("Parametrized",)


def _known_names(tree: ast.AST) -> set[str]:
    """Class names defined anywhere in the file, plus every imported / aliased top-level name."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _literal_names(value: ast.AST) -> list[str] | None:
    """Strings of a literal list/tuple/set, or None when the value cannot be read statically."""
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts):
            return [e.value for e in value.elts]
    return None


def _attr_assignments(tree: ast.AST):
    """Yield (class_node, assignment_stmt, names) for every class that sets `_no_split_modules`."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            else:
                continue
            if any(isinstance(t, ast.Name) and t.id == "_no_split_modules" for t in targets):
                names = _literal_names(value)
                if names is not None:  # non-literal (`= Other._no_split_modules`) cannot be checked
                    yield cls, stmt, names


def _bad_entries(tree: ast.AST):
    """Yield (class_name, lineno, bad_name, suggestions) for entries that are not classes of `tree`."""
    known = _known_names(tree)
    candidates = [n for n in known if n[:1].isupper()]
    for cls, stmt, names in _attr_assignments(tree):
        for name in names:
            if name in known or name.startswith(RUNTIME_CLASS_PREFIXES):
                continue
            yield cls.name, stmt.lineno, name, difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)


def _generated_siblings(modular_path: Path):
    """`modeling_*.py` files in the same directory whose header says they come from this modular file."""
    for sibling in sorted(modular_path.parent.glob("modeling_*.py")):
        try:
            head = "\n".join(sibling.read_text(encoding="utf-8").splitlines()[:20])
        except OSError:
            continue
        if GENERATED_MARKER in head and modular_path.name in head:
            yield sibling


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []

    violations: list[Violation] = []
    if file_path.name.startswith("modeling_"):
        for cls, line, name, hints in _bad_entries(tree):
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=line,
                    message=(
                        f"{cls}._no_split_modules lists `{name}`, which is not a class in {file_path.name}, so accelerate "
                        f"will never match it. Did you mean one of these: {hints}?"
                    ),
                )
            )
    # check modular generated files similar way as TRF055 does for backbone/multimodal models
    elif file_path.name.startswith("modular_"):
        modular_classes = {c.name: c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)}
        for sibling in _generated_siblings(file_path):
            try:
                generated_tree = ast.parse(sibling.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for cls, _, name, hints in _bad_entries(generated_tree):
                node = modular_classes.get(cls)
                if node is None:
                    # class pulled in by the converter without being defined here: no better anchor
                    line = 1
                else:
                    own = [s for c, s, _ in _attr_assignments(node) if c is node]
                    line = own[0].lineno if own else node.lineno
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=line,
                        message=(
                            f"{cls}._no_split_modules lists `{name}`, which is not a class in {sibling.name} generated from {file_path.name}, "
                            f"so accelerate will never match it. Did you mean one of these: {hints}?"
                        ),
                    )
                )

    return violations
