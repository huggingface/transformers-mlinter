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

"""TRF019: ModelNameProcessorKwargs must not define _defaults; put them in processor_config.json."""

import ast
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path

from ._helpers import Violation


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption


def _is_processing_file(file_path: Path) -> bool:
    return file_path.suffix == ".py" and file_path.name.startswith("processing_")


def _defaults_assignment(class_node: ast.ClassDef) -> ast.stmt | None:
    """Return the AST statement for `_defaults = ...` inside the class body, or None."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_defaults" for t in item.targets):
                return item
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and item.target.id == "_defaults" and item.value is not None:
                return item
    return None


def _is_non_empty_dict(node: ast.AST) -> bool:
    return isinstance(node, ast.Dict) and len(node.keys) > 0


@lru_cache(maxsize=256)
def _file_first_commit_date(file_path: Path) -> date | None:
    """Return the date of the earliest commit that added this file, or None if not in git."""
    try:
        result = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%as", "--", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        # git log is newest-first; the last entry is the initial addition commit
        return date.fromisoformat(lines[-1])
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not _is_processing_file(file_path):
        return []

    if CUTOFF_DATE:
        cutoff = date.fromisoformat(CUTOFF_DATE)
        first_commit = _file_first_commit_date(file_path)
        if first_commit is not None and first_commit < cutoff:
            return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.endswith("ProcessorKwargs"):
            continue

        stmt = _defaults_assignment(node)
        if stmt is None:
            continue

        value = stmt.value  # type: ignore[union-attr]
        if not _is_non_empty_dict(value):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=stmt.lineno,
                message=(
                    f"{RULE_ID}: `{node.name}` sets `_defaults` in code. "
                    "Move processor defaults to `processor_config.json` instead."
                ),
            )
        )

    return violations
