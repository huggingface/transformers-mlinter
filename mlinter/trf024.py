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

"""TRF024: Layer dimensions must come from the config, not from an integer literal in the modeling file."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Layer constructor -> positional indices that carry a model dimension. Shape-only arguments
# (kernel_size, stride, padding, num_groups, ...) are not listed: they describe the operator, not the
# architecture's width, and hardcoding them is normal.
DIMENSION_ARGUMENTS: dict[str, tuple[int, ...]] = {
    "Linear": (0, 1),
    "LazyLinear": (0,),
    "Bilinear": (0, 1, 2),
    "Embedding": (0, 1),
    "EmbeddingBag": (0, 1),
    "LayerNorm": (0,),
    "RMSNorm": (0,),
    "GroupNorm": (1,),
    "InstanceNorm1d": (0,),
    "InstanceNorm2d": (0,),
    "InstanceNorm3d": (0,),
    "BatchNorm1d": (0,),
    "BatchNorm2d": (0,),
    "BatchNorm3d": (0,),
    "Conv1d": (0, 1),
    "Conv2d": (0, 1),
    "Conv3d": (0, 1),
    "ConvTranspose1d": (0, 1),
    "ConvTranspose2d": (0, 1),
    "ConvTranspose3d": (0, 1),
    "MultiheadAttention": (0,),
}
DIMENSION_KEYWORDS = {
    "in_features",
    "out_features",
    "in_channels",
    "out_channels",
    "num_embeddings",
    "embedding_dim",
    "embed_dim",
    "normalized_shape",
    "num_channels",
    "hidden_size",
}

# Small integers are almost always a genuine constant of the operator rather than a model width:
# a 1-unit scalar head, 2 for a binary classifier, 3 for RGB, 4 for a quaternion. Above this the
# literal is a width that belongs in the config.
MAX_INLINE_DIMENSION = 8


def _literal_dimension(node: ast.AST) -> int | None:
    """Return the integer a dimension argument resolves to, if it is a bare literal above the bound."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value if node.value > MAX_INLINE_DIMENSION else None
    # `nn.LayerNorm((1024,))` and `nn.LayerNorm([1024])` are the same declaration.
    if isinstance(node, ast.Tuple | ast.List):
        for element in node.elts:
            found = _literal_dimension(element)
            if found is not None:
                return found
    return None


def _layer_name(call: ast.Call) -> str | None:
    """Return the torch.nn layer being constructed, for `nn.Linear(...)` / `torch.nn.Linear(...)` / `Linear(...)`."""
    try:
        dotted = full_name(call.func)
    except ValueError:
        return None
    parts = dotted.split(".")
    leaf = parts[-1]
    if leaf not in DIMENSION_ARGUMENTS:
        return None
    # Only accept the bare name when it is unqualified or reached through an `nn`/`torch.nn` prefix,
    # so `self.something.Linear(...)` on an unrelated object is not mistaken for a layer.
    if len(parts) == 1 or parts[-2] == "nn":
        return leaf
    return None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        layer = _layer_name(node)
        if layer is None:
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        offenders: list[int] = []
        for index in DIMENSION_ARGUMENTS[layer]:
            if index < len(node.args):
                found = _literal_dimension(node.args[index])
                if found is not None:
                    offenders.append(found)
        for keyword in node.keywords:
            if keyword.arg in DIMENSION_KEYWORDS:
                found = _literal_dimension(keyword.value)
                if found is not None:
                    offenders.append(found)

        if not offenders:
            continue

        rendered = ", ".join(str(value) for value in dict.fromkeys(offenders))
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: `nn.{layer}` is built with the hardcoded dimension(s) {rendered}. "
                    "Read the value from the config so checkpoints of other sizes can load."
                ),
            )
        )

    return violations
