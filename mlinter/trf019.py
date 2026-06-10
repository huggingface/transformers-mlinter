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
import subprocess
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ._helpers import Violation


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Check the main `transformers` repo, not the fork/working branch/etc
GITHUB_RAW_URL = "https://raw.githubusercontent.com/huggingface/transformers/main"


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


# The helper for date are copied from `transformers/utils/add_dates.py` which checks models' release date
def check_file_exists_on_github(file_path: str) -> bool:
    """Check if a file exists on the main branch of the GitHub repository.

    Args:
        file_path: Relative path from repository root

    Returns:
        True if file exists on GitHub main branch (or if check failed), False only if confirmed 404

    Note:
        On network errors or other issues, returns True (assumes file exists) with a warning.
        This prevents the script from failing due to temporary network issues.
    """
    # Convert absolute path to relative path from repository root if needed
    # if file_path.startswith(ROOT):
    #     file_path = file_path[len(ROOT) :].lstrip("/")

    # Construct the raw GitHub URL for the file
    url = f"{GITHUB_RAW_URL}/{file_path}"

    try:
        # Make a HEAD request to check if file exists (more efficient than GET)
        request = Request(url, method="HEAD")
        request.add_header("User-Agent", "transformers-add-dates-script")

        with urlopen(request, timeout=10) as response:
            return response.status == 200
    except HTTPError as e:
        if e.code == 404:
            # File doesn't exist on GitHub
            return False
        # HTTP error (non-404): assume file exists and continue with local git history
        return True
    except Exception:
        # Network/timeout error: assume file exists and continue with local git history
        return True


def get_first_commit_date(file_path: Path) -> datetime | None:
    """Get the first commit date of the model's init file or model.md. This date is considered as the date the model was added to HF transformers"""

    # Check if file exists on GitHub main branch
    file_exists_on_github = check_file_exists_on_github(file_path)

    if not file_exists_on_github:
        # File does not exist on GitHub main branch (new model), use today's date
        final_date = date.today().isoformat()
    else:
        # File exists on GitHub main branch, get the first commit date from local git history
        final_date = subprocess.check_output(
            ["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", file_path], text=True
        )
    out = final_date.strip()
    return date.fromisoformat(out.split("\n")[0][:10]) if out else None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not _is_processing_file(file_path):
        return []

    if CUTOFF_DATE:
        cutoff = date.fromisoformat(CUTOFF_DATE)
        first_commit = get_first_commit_date(file_path)
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
