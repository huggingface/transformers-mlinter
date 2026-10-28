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

"""TRF060: model files must import from transformers with relative imports."""

import ast
from pathlib import Path

from ._helpers import MODELS_ROOT, Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

_PACKAGE = "transformers"
_MAX_LISTED_NAMES = 3


def _package_parts(file_path: Path) -> list[str] | None:
    """The package containing `file_path`, as `transformers`-rooted parts, or None outside MODELS_ROOT.

    None also covers linting a standalone model repo, where `transformers` is a dependency and an
    absolute import is the only correct form.
    """
    for root in (MODELS_ROOT.resolve(), MODELS_ROOT):
        try:
            relative = (file_path.resolve() if root.is_absolute() else file_path).relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            return None
        return [_PACKAGE, "models", *relative.parts[:-1]]
    return None


def _relative_module(package: list[str], target: list[str]) -> str:
    """Spell `target` (a dotted module path split into parts) relative to `package`."""
    common = 0
    while common < min(len(package), len(target)) and package[common] == target[common]:
        common += 1
    dots = "." * (len(package) - common + 1)
    return dots + ".".join(target[common:])


def _suggestion(node: ast.Import | ast.ImportFrom, package: list[str]) -> str | None:
    if isinstance(node, ast.ImportFrom):
        if node.level != 0 or node.module is None:
            return None
        target = node.module.split(".")
        if target[0] != _PACKAGE:
            return None
        # Spelling out a long import list would only bury the part that changes, the module path.
        if len(node.names) > _MAX_LISTED_NAMES:
            names = "..."
        else:
            names = ", ".join(
                alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in node.names
            )
        return f"from {_relative_module(package, target)} import {names}"

    for alias in node.names:
        target = alias.name.split(".")
        if target[0] != _PACKAGE:
            continue
        if len(target) == 1:
            return "a relative `from ... import <name>` for each name used"
        bound = alias.asname or target[-1]
        suffix = f" as {bound}" if bound != target[-1] else ""
        return f"from {_relative_module(package, target[:-1])} import {target[-1]}{suffix}"
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    # Tests import the installed package by design, and `convert_*.py` scripts are run directly
    # (`python convert_acme.py`), where a relative import has no parent package to resolve against.
    if file_path.name.startswith(("test_", "convert_")):
        return []
    package = _package_parts(file_path)
    if package is None or is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        suggestion = _suggestion(node, package)
        if suggestion is None:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        replacement = f"`{suggestion}`" if suggestion.startswith("from ") else suggestion
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: absolute `transformers` import. Use a relative import like {replacement} instead."
                ),
            )
        )
    return violations
