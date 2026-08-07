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

"""TRF030: Reaching more than one sub-config deep means the module was handed the wrong config."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# `config.hidden_size` is one hop and `config.text_config.hidden_size` is two, which is the normal
# sub-config access. Three or more means the module is digging through a hierarchy it should have been
# given a branch of.
MAX_CONFIG_HOPS = 2


def _config_hops(node: ast.Attribute) -> int:
    """Number of attribute hops after the `config` root, or 0 when the chain is not rooted there."""
    try:
        dotted = full_name(node)
    except ValueError:
        return 0
    body = dotted.removeprefix("self.")
    if not body.startswith("config."):
        return 0
    return body.count(".")


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    reported: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        hops = _config_hops(node)
        if hops <= MAX_CONFIG_HOPS or node.lineno in reported:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        reported.add(node.lineno)
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: `{full_name(node)}` reaches {hops} levels into the config. "
                    "Pass the relevant sub-config to the module instead of walking the hierarchy."
                ),
            )
        )
    return violations
