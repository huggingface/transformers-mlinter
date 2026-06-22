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

"""TRF019: `ProcessorKwargs` must not define non-empty `_defaults`; move them in `processor_config.json` in the hub."""

import ast
import re
from datetime import date
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _model_dir_name


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

DOCS_ROOT = Path("docs/source/en/model_doc")

_CONTRIBUTION_DATE_RE = re.compile(
    r"\n\*This model was (?:published in HF papers on (.*) and )?"
    r"contributed to Hugging Face Transformers on (\d{4}-\d{2}-\d{2})\.\*"
)


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


def model_contribution_date(file_path: Path) -> date | None:
    """Return the Transformers contribution date from the model's doc page, or None if not found."""
    model_name = _model_dir_name(file_path)
    if model_name is None:
        return None
    doc_path = DOCS_ROOT / f"{model_name}.md"
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _CONTRIBUTION_DATE_RE.search(text)
    return date.fromisoformat(match.group(2)) if match is not None else match


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not _is_processing_file(file_path):
        return []

    if CUTOFF_DATE:
        cutoff = date.fromisoformat(CUTOFF_DATE)
        contribution_date = model_contribution_date(file_path)
        if contribution_date is not None and contribution_date < cutoff:
            return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        if not any(base.id == "ProcessingKwargs" for base in node.bases if isinstance(base, ast.Name)):
            continue

        if any(_has_rule_suppression(source_lines, RULE_ID, lineno) for lineno in node.lineno):
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
