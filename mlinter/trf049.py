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

"""TRF049: weight initialization belongs in _init_weights, never in __init__ (meta-device init discards it)."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _has_rule_suppression, _top_level_classes, full_name


RULE_ID = ""  # Set by discovery

_INIT_FUNCTIONS = {
    "normal_",
    "trunc_normal_",
    "uniform_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "constant_",
    "ones_",
    "zeros_",
    "zero_",
    "fill_",
}


def _is_init_call(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    try:
        parts = full_name(call.func).split(".")
    except ValueError:
        return False
    if parts[-1] not in _INIT_FUNCTIONS:
        return False
    # nn.init.normal_, torch.nn.init.normal_, init.normal_ (the transformers initialization module).
    if parts[-2:-1] == ["init"] and parts[0] in ("nn", "torch", "init"):
        return True
    # In-place initialization on an own parameter/buffer: self.weight.data.normal_().
    return parts[0] == "self"


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in _top_level_classes(tree):
        init_method = _class_methods(node).get("__init__")
        if init_method is None:
            continue

        for sub in ast.walk(init_method):
            if not isinstance(sub, ast.Call) or not _is_init_call(sub):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, sub.lineno):
                continue

            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=sub.lineno,
                    message=(
                        f"{RULE_ID}: {node.name}.__init__ initializes weight values. Models instantiate on the "
                        "meta device, so values written in __init__ are discarded; allocate with torch.empty "
                        "and initialize in _init_weights."
                    ),
                )
            )

    return violations
