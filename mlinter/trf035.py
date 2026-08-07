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

"""TRF035: Model files must not silence the linter with `# noqa`."""

import ast
import re
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

NOQA = re.compile(r"#\s*noqa\b(?::\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*))?", re.IGNORECASE)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_", "configuration_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for index, line in enumerate(source_lines, start=1):
        match = NOQA.search(line)
        if match is None:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, index):
            continue
        codes = match.group("codes")
        detail = f" (`{codes}`)" if codes else ""
        violations.append(
            Violation(
                file_path=file_path,
                line_number=index,
                message=(
                    f"{RULE_ID}: `# noqa`{detail} in a model file. "
                    "Fix the underlying issue; model files should not need linter suppressions."
                ),
            )
        )
    return violations
