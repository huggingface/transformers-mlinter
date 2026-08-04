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

"""TRF023: Config fields must use the canonical dimension names, not the upstream paper's abbreviations."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Legacy field name -> canonical replacement. Only names that are unambiguously the same quantity as
# their canonical counterpart are listed. Deliberately excluded because they are genuinely ambiguous
# or still idiomatic in parts of the library: `num_heads`, `num_layers`, `embed_dim`, `mlp_ratio`.
LEGACY_CONFIG_FIELDS = {
    "d_model": "hidden_size",
    "n_embd": "hidden_size",
    "d_ff": "intermediate_size",
    "d_inner": "intermediate_size",
    "ffn_dim": "intermediate_size",
    "ffn_hidden_size": "intermediate_size",
    "expansion_ratio": "intermediate_size",
    "d_head": "head_dim",
    "n_head": "num_attention_heads",
    "n_heads": "num_attention_heads",
    "n_layer": "num_hidden_layers",
    "n_layers": "num_hidden_layers",
    "num_blocks": "num_hidden_layers",
}


def _declared_field_names(class_node: ast.ClassDef) -> list[tuple[str, int]]:
    """Return (name, lineno) for every field the config class declares in its own body or __init__."""
    fields: list[tuple[str, int]] = []
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            fields.append((item.target.id, item.lineno))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    fields.append((target.id, item.lineno))
        elif isinstance(item, ast.FunctionDef) and item.name in {"__init__", "__post_init__"}:
            # Dataclass-style configs declare in the body; older ones assign in __init__.
            for sub in ast.walk(item):
                targets = []
                if isinstance(sub, ast.Assign):
                    targets = sub.targets
                elif isinstance(sub, ast.AnnAssign):
                    targets = [sub.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        fields.append((target.attr, sub.lineno))
            # Keyword-only defaults in the signature are declarations too.
            for arg in list(item.args.args) + list(item.args.kwonlyargs):
                if arg.arg != "self":
                    fields.append((arg.arg, arg.lineno))
    return fields


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("configuration_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith("Config"):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        seen: set[str] = set()
        for field_name, lineno in _declared_field_names(node):
            canonical = LEGACY_CONFIG_FIELDS.get(field_name)
            if canonical is None or field_name in seen:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, lineno):
                continue
            seen.add(field_name)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=lineno,
                    message=(
                        f"{RULE_ID}: `{node.name}` declares `{field_name}`. "
                        f"Use the canonical name `{canonical}` and derive `{field_name}` on conversion if the "
                        "checkpoint needs it."
                    ),
                )
            )

    return violations
