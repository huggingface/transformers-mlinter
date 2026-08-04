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

"""TRF027: Model files must carry the Apache 2.0 license header."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# The header sits at the very top of the file, above the module docstring and imports.
HEADER_SCAN_LINES = 25
HEADER_MARKER = "Apache License"


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(
        ("modeling_", "modular_", "configuration_", "processing_", "image_processing_", "video_processing_")
    ):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []
    if any(HEADER_MARKER in line for line in source_lines[:HEADER_SCAN_LINES]):
        return []
    if _has_rule_suppression(source_lines, RULE_ID, 1):
        return []

    return [
        Violation(
            file_path=file_path,
            line_number=1,
            message=(f"{RULE_ID}: missing the Apache 2.0 license header. Copy it from any neighbouring model file."),
        )
    ]
