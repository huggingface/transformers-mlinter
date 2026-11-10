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

"""`PreTrainedModel._no_split_modules` must only name classes that exist."""

import ast
import difflib
import re
from pathlib import Path

from ._helpers import GENERATED_FILE_MARKER, Violation, _has_rule_suppression, read_file_head


RULE_ID = ""  # Set by discovery

# torch.nn.utils.parametrize builds classes named "Parametrized" + <original name> at runtime,
# so they can never be found in a source file. Check `hubert` in transformers source code for an ex
RUNTIME_CLASS_PREFIXES = ("Parametrized",)
# Every tree produced from a given modular file, keyed by that modular file.
_GENERATED_TREES_CACHE: dict[Path, dict[Path, ast.Module]] = {}


def _known_names(tree: ast.AST) -> set[str]:
    """Class names defined anywhere in the file, plus every imported name (under its real, un-aliased name)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[0] for a in node.names)
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


def _is_generated_from(path: Path, modular_path: Path) -> bool:
    head = read_file_head(path)
    return head is not None and GENERATED_FILE_MARKER in head and modular_path.name in head


def _generated_trees(modular_path: Path) -> dict[Path, ast.Module]:
    """Parsed AST of every file the converter produced from `modular_path`, keyed by file path.

    A modular file can emit several files (`modeling_*.py`, `processing_*.py`,
    `image_processing_pil_*.py`, ...) whose names do not always follow from the modular file name, so
    the generation banner is what identifies them.
    """
    if modular_path in _GENERATED_TREES_CACHE:
        return _GENERATED_TREES_CACHE[modular_path]

    trees: dict[Path, ast.Module] = {}
    for sibling in sorted(modular_path.parent.glob("*.py")):
        if sibling == modular_path or not _is_generated_from(sibling, modular_path):
            continue
        try:
            trees[sibling] = ast.parse(sibling.read_text(encoding="utf-8"), filename=str(sibling))
        except (OSError, SyntaxError, ValueError):
            continue
    _GENERATED_TREES_CACHE[modular_path] = trees
    return trees


def _first_mention(source_lines: list[str], name: str) -> int:
    """1-based line of the first whole-word occurrence of `name`, or 1 when the file never names it."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for lineno, text in enumerate(source_lines, start=1):
        if pattern.search(text):
            return lineno
    return 1


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    violations: list[Violation] = []
    if file_name.startswith("modeling_"):
        for cls, line, name, hints in _bad_entries(tree):
            if _has_rule_suppression(source_lines, RULE_ID, line):
                continue
            hints = f" Did you mean one of these: {', '.join(f'{h}' for h in hints)}?" if hints else ""
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=line,
                    message=(
                        f"{RULE_ID}: {cls}._no_split_modules lists `{name}`, which is not a class in {file_path.name}, so accelerate "
                        f"will never match it.{hints}"
                    ),
                )
            )
    # check modular generated files similar way as TRF055 does for backbone/multimodal models
    elif file_name.startswith("modular_"):
        modular_classes = {c.name: c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)}
        for generated_path, generated_tree in _generated_trees(file_path).items():
            for cls, _, name, hints in _bad_entries(generated_tree):
                node = modular_classes.get(cls)
                if node is None:
                    # The converter pulled the class in from another model without it being defined here.
                    # Report it on the first line naming it (usually `__all__`), where a suppression can go.
                    line = _first_mention(source_lines, cls)
                else:
                    own = [s for c, s, _ in _attr_assignments(node) if c is node]
                    line = own[0].lineno if own else node.lineno
                if _has_rule_suppression(source_lines, RULE_ID, line):
                    continue
                hints = f" Did you mean one of these: {', '.join(hints)}?" if hints else ""
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=line,
                        message=(
                            f"{RULE_ID}: {cls}._no_split_modules lists `{name}`, which is not a class in "
                            f"{generated_path.name} generated from {file_path.name}, "
                            f"so accelerate will never match it.{hints}"
                        ),
                    )
                )
    return violations
