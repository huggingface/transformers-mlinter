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

"""TRF029: A module taking `config` must not also take arguments that live on the config."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Argument names that are unambiguously config fields. Passing one of these next to `config` means the
# same number now has two sources of truth, and the caller decides which one wins.
CONFIG_FIELD_ARGUMENTS = {
    "attention_dropout",
    "d_model",
    "dropout",
    "embed_dim",
    "eps",
    "head_dim",
    "hidden_act",
    "hidden_dropout",
    "hidden_size",
    "image_size",
    "initializer_range",
    "intermediate_size",
    "layer_norm_eps",
    "max_position_embeddings",
    "mlp_ratio",
    "n_heads",
    "n_layers",
    "num_attention_heads",
    "num_channels",
    "num_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "patch_size",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
}


def _parameters(args: ast.arguments) -> list[tuple[str, ast.expr | None]]:
    """Every named parameter with its default expression, or None when it has none."""
    # `defaults` covers the tail of posonlyargs + args, so the two lists are paired from the right.
    positional = args.posonlyargs + args.args
    padding: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    paired = list(zip(positional, padding + list(args.defaults)))
    paired += list(zip(args.kwonlyargs, args.kw_defaults))
    return [(arg.arg, default) for arg, default in paired]


def _is_optional_override(default: ast.expr | None) -> bool:
    """Whether a parameter is optional with a literal `None` default, making it an override.

    `def __init__(self, config, intermediate_size=None)` is not a second source of truth: the config
    stays the source, and the parameter is an override a caller passes when one class has to serve two
    widths -- the dense MLP and the expert MLP of a MoE model, built from the same class. Nothing is
    ambiguous, because omitting it is the normal case and means "read the config". A parameter that is
    required, or one carrying a hardcoded default such as `intermediate_size=4096`, is a different
    matter: there the caller has to supply the number, or the signature already decided it.
    """
    return isinstance(default, ast.Constant) and default.value is None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    violations: list[Violation] = []
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for item in class_node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            parameters = _parameters(item.args)
            if not any(name == "config" for name, _ in parameters):
                continue
            redundant = [
                name
                for name, default in parameters
                if name in CONFIG_FIELD_ARGUMENTS and not _is_optional_override(default)
            ]
            if not redundant:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, item.lineno):
                continue
            rendered = ", ".join(f"`{name}`" for name in redundant)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=item.lineno,
                    message=(
                        f"{RULE_ID}: `{class_node.name}.__init__` takes `config` and also {rendered}. "
                        "Read those off the config inside the module so there is one source of truth."
                    ),
                )
            )
    return violations
