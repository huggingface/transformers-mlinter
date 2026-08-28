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
    call_leaf_name,
    full_name,
    is_exempt_by_cutoff,
)


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption

# Only the repeated per-layer blocks are in scope; a ModuleList of projections or experts is not a
# gradient-checkpointing boundary.
LAYER_CLASS_SUFFIXES = ("Layer", "Block")
_MAX_INHERITANCE_HOPS = 12

# Checkpointing only pays for itself on the stack whose activations dominate memory: the model's main
# sequence-processing trunk. A conv backbone, a DPT or segmentation head, a vocoder upsampler and an
# adapter are all `nn.ModuleList`s of `*Layer`/`*Block` classes too, but whether to trade compute for
# memory there is the model author's call, not a defect. The trunk is recognised by the module that
# does the token mixing -- attention in most models, and a named modulation/mixer/SSM block in the
# architectures that have no attention at all.
TOKEN_MIXING_HINTS = ("attention", "attn", "modulation", "mixer", "mamba", "ssm")

# Checkpointing recomputes the layer's forward during the backward pass. A module holding running
# statistics would fold every batch in twice, so asking it to checkpoint trades a memory saving for
# corrupted statistics. Never report one.
RUNNING_STATS_HINTS = ("batchnorm", "instancenorm")

_SUPPORT_FLAG = "supports_gradient_checkpointing"


def _imported_classes(tree: ast.Module, file_path: Path) -> dict[str, tuple[Path, str]]:
    imports: dict[str, tuple[Path, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level == 0 or node.module is None:
            continue
        base_dir = file_path.parent
        for _ in range(node.level - 1):
            base_dir = base_dir.parent
        imported_path = base_dir.joinpath(*node.module.split(".")).with_suffix(".py")
        for alias in node.names:
            imports[alias.asname or alias.name] = (imported_path, alias.name)
    return imports


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
) -> bool | None:
    """Return True if the chain reaches GradientCheckpointingLayer, False if fully resolved, None if not."""
    if seen is None:
        seen = set()
    key = (file_path, name)
    if key in seen or hops >= _MAX_INHERITANCE_HOPS:
        return None
    seen.add(key)

    class_to_bases = _collect_class_bases(tree)
    imports = _imported_classes(tree, file_path)
    if name not in class_to_bases:
        return None

    found_unknown = False
    for base in class_to_bases[name]:
        simple = base.split(".")[-1]
        if simple == "GradientCheckpointingLayer":
            return True
        if base.startswith(("nn.", "torch.nn.")) or simple in {"Module", "object"}:
            continue
        if simple in class_to_bases:
            resolved = _subclasses_gradient_checkpointing_layer(simple, tree, file_path, cache, seen, hops + 1)
        elif simple in imports:
            imported_path, imported_name = imports[simple]
            imported_tree = _parse_file(imported_path, cache)
            resolved = (
                None
                if imported_tree is None
                else _subclasses_gradient_checkpointing_layer(
                    imported_name, imported_tree, imported_path, cache, seen, hops + 1
                )
            )
        else:
            resolved = None

        if resolved is True:
            return True
        if resolved is None:
            found_unknown = True

    return None if found_unknown else False


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _declares_checkpointing_support(tree: ast.Module) -> bool | None:
    """True if any class here turns the flag on, False if one sets it off, None if it is never named.

    A composite model carries several PreTrainedModel classes; one of them turning the flag on is
    enough for the stack to be reachable, so a True anywhere wins over a False elsewhere.
    """
    verdict = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.Assign):
                targets = item.targets
            elif isinstance(item, ast.AnnAssign):
                targets = [item.target]
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == _SUPPORT_FLAG for t in targets):
                continue
            if isinstance(item.value, ast.Constant):
                if item.value.value:
                    return True
                verdict = False
    return verdict


def _model_supports_checkpointing(tree: ast.Module, file_path: Path, cache: dict[Path, ast.Module | None]) -> bool:
    """Whether the model owning `file_path` can be gradient-checkpointed at all.

    `PreTrainedModel.supports_gradient_checkpointing` defaults to False, so a model that never turns
    it on raises from `gradient_checkpointing_enable()` rather than skipping a layer -- the finding
    describes an outcome that model cannot reach. The flag sits on the XxxPreTrainedModel in
    modeling_*.py, which a modular file does not repeat, so the model's sibling files are read before
    concluding anything. A model that never names the flag is taking the default: not supported.
    """
    own = _declares_checkpointing_support(tree)
    if own is not None:
        return own

    verdict = None
    try:
        siblings = sorted(file_path.parent.glob("*.py"))
    except OSError:
        siblings = []
    for sibling in siblings:
        if sibling == file_path or not sibling.name.startswith(("modeling_", "modular_")):
            continue
        sibling_tree = _parse_file(sibling, cache)
        if sibling_tree is None:
            continue
        found = _declares_checkpointing_support(sibling_tree)
        if found is True:
            return True
        if found is False:
            verdict = False
    return bool(verdict)


def _submodule_signals(
    name: str,
    tree: ast.Module,
    file_path: Path,
    cache: dict[Path, ast.Module | None],
    seen: set[tuple[Path, str]] | None = None,
    hops: int = 0,
) -> set[str]:
    """Lowercased names of what a class is built from: its `self.x = Y(...)` attributes and their classes.

    Only assignments count. A bare load such as `config._attn_implementation`, or an `attention_mask`
    argument threaded through a forward, says nothing about whether this layer holds an attention
    module -- counting those would mark almost every layer in the library as trunk. Submodules defined
    in the same file are followed one level down, since a block can delegate its mixing to a child
    (`Florence2VisionBlock` holds its attention inside `Florence2VisionSpatialBlock`), and base classes
    are followed the same way the checkpointing chain is.
    """
    if seen is None:
        seen = set()
    key = (file_path, name)
    if key in seen or hops >= _MAX_INHERITANCE_HOPS:
        return set()
    seen.add(key)

    class_node = _class_node(tree, name)
    if class_node is None:
        return set()

    signals: set[str] = set()
    instantiated: set[str] = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                signals.add(target.attr.lower())
        leaf = call_leaf_name(node.value)
        if leaf:
            signals.add(leaf.lower())
            instantiated.add(leaf)

    imports = _imported_classes(tree, file_path)
    for child in instantiated:
        if _class_node(tree, child) is not None:
            signals |= _submodule_signals(child, tree, file_path, cache, seen, hops + 1)

    for base in _collect_class_bases(tree).get(name, []):
        simple = base.split(".")[-1]
        if base.startswith(("nn.", "torch.nn.")) or simple in {"Module", "object"}:
            continue
        if _class_node(tree, simple) is not None:
            signals |= _submodule_signals(simple, tree, file_path, cache, seen, hops + 1)
        elif simple in imports:
            imported_path, imported_name = imports[simple]
            imported_tree = _parse_file(imported_path, cache)
            if imported_tree is not None:
                signals |= _submodule_signals(imported_name, imported_tree, imported_path, cache, seen, hops + 1)
    return signals


def _matches(signals: set[str], hints: tuple[str, ...]) -> bool:
    return any(hint in signal for signal in signals for hint in hints)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    class_to_bases = _collect_class_bases(tree)
    local_classes = set(class_to_bases)
    parsed_files: dict[Path, ast.Module | None] = {file_path: tree}
    if not _model_supports_checkpointing(tree, file_path, parsed_files):
        return []

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
            inheritance_status = _subclasses_gradient_checkpointing_layer(layer_name, tree, file_path, parsed_files)
            if inheritance_status is not False:
                continue
            signals = _submodule_signals(layer_name, tree, file_path, parsed_files)
            if _matches(signals, RUNNING_STATS_HINTS):
                continue
            if not _matches(signals, TOKEN_MIXING_HINTS):
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
