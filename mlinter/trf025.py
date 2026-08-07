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

"""TRF025: Attention masks must be built once in the model, not rebuilt inside a layer or attention module."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# The `masking_utils` entry points. Any `create_*_mask` helper is treated the same way, so a model
# adding its own `create_foo_mask` follows the rule without this list having to grow.
MASK_FACTORIES = {
    "create_causal_mask",
    "create_bidirectional_mask",
    "create_sliding_window_causal_mask",
    "create_chunked_causal_mask",
    "create_masks_for_generate",
}

# Only per-layer blocks are in scope. A model or an encoder that builds the mask once and hands it to
# its layer stack is doing exactly the right thing, so those names are not matched. Suffixes are
# checked on the class name because in modular files a layer's base class is another model's layer,
# which no local-inheritance walk can resolve.
PER_LAYER_CLASS_SUFFIXES = ("Layer", "Attention", "Block")


def _is_mask_factory(call: ast.Call) -> str | None:
    try:
        leaf = full_name(call.func).split(".")[-1]
    except ValueError:
        return None
    if leaf in MASK_FACTORIES:
        return leaf
    if leaf.startswith("create_") and leaf.endswith("_mask"):
        return leaf
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []

    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        if not class_node.name.endswith(PER_LAYER_CLASS_SUFFIXES):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, class_node.lineno):
            continue

        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call):
                continue
            factory = _is_mask_factory(node)
            if factory is None:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{class_node.name}` calls `{factory}`. "
                        "Build the mask once in the model and pass it down, so it is not rebuilt per layer."
                    ),
                )
            )

    return violations
