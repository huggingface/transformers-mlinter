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

"""TRF028: Model files must carry a complete license header."""

import ast
import re
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# The header sits at the very top of the file, above the module docstring and imports. Generated
# modeling files push it down by a six-line banner, which still lands well inside this window.
HEADER_SCAN_LINES = 25

# The boilerplate every header shares, in the order it appears. Matching only "Apache License"
# accepts a header truncated mid-paragraph or mangled by a bad search-and-replace, which is what
# every header defect in the library actually looks like.
#
# Only the license-independent clauses are required. Not every model is Apache 2.0 — BLIP ships
# BSD-3-clause and Sapiens2 ships Meta's own license — but all of them carry the same warranty
# paragraph, so requiring that catches broken headers without flagging a deliberate license choice.
# The copyright line is not checked either: its year and attribution legitimately vary per model.
_LICENSED_UNDER = re.compile(r"licensed under the .{0,60}?license")
REQUIRED_CLAUSES = (
    "you may obtain a copy of the license at",
    "unless required by applicable law or agreed to in writing, software",
    'distributed under the license is distributed on an "as is" basis,',
    "without warranties or conditions of any kind, either express or implied.",
    "see the license for the specific language governing permissions and",
    "limitations under the license.",
)


def _normalized_header(source_lines: list[str]) -> str:
    """The comment block at the top of the file as one lowercase, whitespace-collapsed string.

    Flattening the lines lets a clause be matched whether or not the file wraps it differently,
    and dropping the `#` prefix keeps the clause literals readable.
    """
    text = " ".join(line.lstrip().lstrip("#") for line in source_lines[:HEADER_SCAN_LINES])
    return re.sub(r"\s+", " ", text).lower()


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(
        ("modeling_", "modular_", "configuration_", "processing_", "image_processing_", "video_processing_")
    ):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    header = _normalized_header(source_lines)
    missing = [clause for clause in REQUIRED_CLAUSES if clause not in header]
    has_license_line = _LICENSED_UNDER.search(header) is not None
    if has_license_line and not missing:
        return []
    if _has_rule_suppression(source_lines, RULE_ID, 1):
        return []

    # A header missing every clause is absent; one missing a few is truncated or mangled, and
    # naming the first missing clause points straight at where it diverges.
    if not has_license_line and len(missing) == len(REQUIRED_CLAUSES):
        detail = "missing the license header"
    elif not has_license_line:
        detail = "incomplete license header, does not state what license the file is under"
    else:
        detail = f"incomplete license header, does not contain `{missing[0]}`"

    return [
        Violation(
            file_path=file_path,
            line_number=1,
            message=f"{RULE_ID}: {detail}. Copy it verbatim from any neighbouring model file.",
        )
    ]
