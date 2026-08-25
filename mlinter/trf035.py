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

# Ruff's undefined-name family, which a modular file trips by construction rather than by mistake.
# A modular file is a generation source, not shipped code: `__all__` lists names that only exist once
# the converter has expanded it (F822), the body references classes defined in the parent model and
# never locally (F821), and some imports are there purely to be re-exported into the generated file
# (F401). There is no underlying issue behind any of the three, so asking for one is asking for the
# impossible. Every other code, and a suppression naming no code at all, still has something to fix.
MODULAR_EXEMPT_CODES = frozenset({"F401", "F821", "F822"})


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
        codes = [code.strip().upper() for code in (match.group("codes") or "").split(",") if code.strip()]
        is_modular = file_path.name.startswith("modular_")
        if is_modular and codes:
            # A coded suppression survives on the codes that are not exempt, so the message keeps saying
            # what is left to fix. A bare one is never exempt: it hides every future violation too.
            codes = [code for code in codes if code not in MODULAR_EXEMPT_CODES]
            if not codes:
                continue
        if is_modular and not codes:
            # Half the bare ones in transformers are an exempt code left unwritten -- a re-exported
            # import, an `__all__` entry the converter fills in -- so the actionable ask is the code,
            # not a rewrite of the line.
            message = (
                f"{RULE_ID}: bare `# noqa` in a modular file. Name the code it silences: "
                f"{', '.join(sorted(MODULAR_EXEMPT_CODES))} are accepted here, because a modular file "
                "does not define every name it uses. A bare one also hides every future violation."
            )
        else:
            detail = f" (`{', '.join(codes)}`)" if codes else ""
            message = (
                f"{RULE_ID}: `# noqa`{detail} in a model file. "
                "Fix the underlying issue; model files should not need linter suppressions."
            )
        violations.append(Violation(file_path=file_path, line_number=index, message=message))
    return violations
