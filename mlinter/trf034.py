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

"""TRF034: Layer classes held in an nn.ModuleList must subclass GradientCheckpointingLayer."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _collect_class_bases,
    _has_rule_suppression,
    full_name,
    imported_classes,
    is_exempt_by_cutoff,
    is_exempt_by_inherited_cutoff,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Only the repeated per-layer blocks are in scope; a ModuleList of projections or experts is not a
# gradient-checkpointing boundary.
LAYER_CLASS_SUFFIXES = ("Layer", "Block")
_MAX_INHERITANCE_HOPS = 12


def _parse_file(path: Path, cache: dict[Path, ast.Module | None]) -> ast.Module | None:
    if path not in cache:
        try:
            cache[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, ValueError):
            cache[path] = None
    return cache[path]


def _subclasses_gradient_checkpointing_layer(
    name: str,
    tree: ast.Module,
    file_path: Path,
    cache: dict[Path, ast.Module | None],
    seen: set[tuple[Path, str]] | None = None,
    hops: int = 0,
) -> tuple[bool | None, Path]:
    """Whether the chain reaches GradientCheckpointingLayer, and which file settles the question.

    True if it reaches it, False if the chain resolves without one, None if some base could not be
    followed. The second element is the file that owns the answer: for a False verdict, the file
    defining the topmost ancestor that stops at `nn.Module` -- which is where the base class would have
    to change, and so which model the violation belongs to. `DFineRepVggBlock(RTDetrRepVggBlock)` is a
    plain module because rt_detr says so, not because d_fine did anything.
    """
    if seen is None:
        seen = set()
    key = (file_path, name)
    if key in seen or hops >= _MAX_INHERITANCE_HOPS:
        return None, file_path
    seen.add(key)

    class_to_bases = _collect_class_bases(tree)
    imports = imported_classes(tree, file_path)
    if name not in class_to_bases:
        return None, file_path

    found_unknown = False
    owner = file_path
    for base in class_to_bases[name]:
        simple = base.split(".")[-1]
        if simple == "GradientCheckpointingLayer":
            return True, file_path
        if base.startswith(("nn.", "torch.nn.")) or simple in {"Module", "object"}:
            continue
        if simple in class_to_bases:
            resolved, resolved_owner = _subclasses_gradient_checkpointing_layer(
                simple, tree, file_path, cache, seen, hops + 1
            )
        elif simple in imports:
            imported_path, imported_name = imports[simple]
            imported_tree = _parse_file(imported_path, cache)
            if imported_tree is None:
                resolved, resolved_owner = None, imported_path
            else:
                resolved, resolved_owner = _subclasses_gradient_checkpointing_layer(
                    imported_name, imported_tree, imported_path, cache, seen, hops + 1
                )
        else:
            resolved, resolved_owner = None, file_path

        if resolved is True:
            return True, resolved_owner
        if resolved is None:
            found_unknown = True
        elif owner == file_path:
            # The first base that resolves to a plain module is the one to name.
            owner = resolved_owner

    return (None, file_path) if found_unknown else (False, owner)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    class_to_bases = _collect_class_bases(tree)
    local_classes = set(class_to_bases)
    parsed_files: dict[Path, ast.Module | None] = {file_path: tree}
    violations: list[Violation] = []
    reported: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            if full_name(node.func).split(".")[-1] != "ModuleList":
                continue
        except ValueError:
            continue
        for inner in ast.walk(node):
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
                continue
            layer_name = inner.func.id
            if layer_name not in local_classes or not layer_name.endswith(LAYER_CLASS_SUFFIXES):
                continue
            if layer_name in reported:
                continue
            inheritance_status, owner_path = _subclasses_gradient_checkpointing_layer(
                layer_name, tree, file_path, parsed_files
            )
            if inheritance_status is not False:
                continue
            # The base class is another model's, and that model is grandfathered: reporting it here
            # asks this author to edit a model their PR does not touch.
            if is_exempt_by_inherited_cutoff(owner_path, file_path, CUTOFF_DATE):
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue
            reported.add(layer_name)
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{layer_name}` is stacked in an `nn.ModuleList` but does not subclass "
                        "`GradientCheckpointingLayer`, so gradient checkpointing silently skips it."
                    ),
                )
            )
    return violations
