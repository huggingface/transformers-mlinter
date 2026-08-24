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

"""Shared AST helper functions used across mlinter rule modules."""

import ast
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


MODELS_ROOT = Path("src/transformers/models")
TESTS_ROOT = Path("tests/models")
DOCS_ROOT = Path("docs/source/en/model_doc")

# Banner that the modular converter writes near the top of every file it generates, followed by the
# path of the modular source, so it also says which modular file a generated file was produced from.
GENERATED_FILE_MARKER = "This file was automatically generated from"
# How much of a file to search for the banner. The converter always writes it in the first lines.
GENERATED_FILE_HEAD_SIZE = 1024

_CONTRIBUTION_DATE_RE = re.compile(
    r"\n\*This model was (?:published in HF papers on (.*) and )?"
    r"contributed to Hugging Face Transformers on (\d{4}-\d{2}-\d{2})\.\*"
)


@dataclass(frozen=True)
class Violation:
    file_path: Path
    line_number: int
    message: str
    rule_id: str | None = None


def read_file_head(path: Path) -> str | None:
    """The first `GENERATED_FILE_HEAD_SIZE` bytes of a file, or None when it cannot be read."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return handle.read(GENERATED_FILE_HEAD_SIZE)
    except OSError:
        return None


def full_name(node: ast.AST):
    """Return full dotted name from an Attribute or Name node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return full_name(node.value) + "." + node.attr
    else:
        raise ValueError("Not a Name or Attribute node")


def _simple_name(name: str) -> str:
    return name.split(".")[-1]


def call_leaf_name(call: ast.Call) -> str | None:
    """Return the final identifier of a call target, or None when there isn't one.

    Unlike `full_name`, this resolves through a chained call: in `x.masked_fill(...).masked_fill(...)`
    the outer target's base is a Call, which `full_name` cannot render, so a rule matching on the leaf
    method name would silently miss every call but the innermost one.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _model_dir_name(file_path: Path) -> str | None:
    # A model's files live under two roots: the implementation under MODELS_ROOT and its tests under
    # TESTS_ROOT. Both are laid out as <root>/<model>/<file>, so either resolves to the same model name.
    for root in (MODELS_ROOT, TESTS_ROOT):
        try:
            relative = file_path.resolve().relative_to(root.resolve())
        except ValueError:
            try:
                relative = file_path.relative_to(root)
            except ValueError:
                continue
        if len(relative.parts) < 2:
            continue
        return relative.parts[0]
    return None


def _known_model_dirs() -> set[str]:
    # Empty outside a transformers checkout — e.g. when linting a standalone model repo, where the
    # rules that resolve sibling models have nothing to resolve against.
    try:
        return {path.name for path in MODELS_ROOT.iterdir() if path.is_dir()}
    except OSError:
        return set()


def model_contribution_date(file_path: Path) -> date | None:
    """Return the Transformers contribution date from the model's doc page, or None if not found."""
    model_name = _model_dir_name(file_path)
    if model_name is None:
        return None
    # Doc pages usually match the model directory, but some spell it with hyphens (`blenderbot_small`
    # -> `blenderbot-small.md`), so try that too before giving up.
    for candidate in (model_name, model_name.replace("_", "-")):
        try:
            text = (DOCS_ROOT / f"{candidate}.md").read_text(encoding="utf-8")
        except OSError:
            continue
        match = _CONTRIBUTION_DATE_RE.search(text)
        if match is not None:
            return date.fromisoformat(match.group(2))
    return None


def is_exempt_by_cutoff(file_path: Path, cutoff_date: str) -> bool:
    """Whether the model owning `file_path` predates `cutoff_date` and is therefore grandfathered.

    Rules that encode a convention introduced at some point in time use this so they do not need to
    carry an allowlist of every model added before it. Models whose doc page has no contribution date
    are checked, so a missing date never silently disables a rule.
    """
    if not cutoff_date:
        return False
    contribution_date = model_contribution_date(file_path)
    return contribution_date is not None and contribution_date < date.fromisoformat(cutoff_date)


def _has_rule_suppression(lines: list[str], rule_id: str, line_number: int) -> bool:
    if line_number <= 0:
        return False
    token = f"trf-ignore: {rule_id}".lower()
    # Accept the suppression on the target line itself (inline comment).
    idx = line_number - 1
    if 0 <= idx < len(lines) and token in lines[idx].lower():
        return True
    # Walk upward from the line directly above the target, skipping any decorator lines, so the
    # comment can sit above decorators (e.g. above `@torch.no_grad()`) and not only squeezed
    # between the decorator and the `def`. Stop at the first non-decorator, non-matching line.
    idx -= 1
    while 0 <= idx < len(lines):
        stripped = lines[idx].strip()
        if token in stripped.lower():
            return True
        if stripped.startswith("@"):
            idx -= 1
            continue
        break
    return False


def _collect_class_bases(tree: ast.Module) -> dict[str, list[str]]:
    class_to_bases: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = []
        for base in node.bases:
            try:
                base_names.append(full_name(base))
            except ValueError:
                continue
        class_to_bases[node.name] = base_names
    return class_to_bases


def _inherits_pretrained_model(
    class_name: str, class_to_bases: dict[str, list[str]], visiting: set[str] | None = None
) -> bool:
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return False
    visiting.add(class_name)

    for base_name in class_to_bases.get(class_name, []):
        simple_base_name = _simple_name(base_name)
        if simple_base_name.endswith("PreTrainedModel"):
            return True
        if simple_base_name in class_to_bases and _inherits_pretrained_model(
            simple_base_name, class_to_bases, visiting
        ):
            return True
    return False


def _base_chain_has_unresolved_import(
    class_name: str,
    class_to_bases: dict[str, list[str]],
    *,
    known_external_bases: set[str] | None = None,
    visiting: set[str] | None = None,
) -> bool:
    """Whether *class_name* inherits from a base that is not defined in this file.

    Modular model files often inherit from another model's imported class. Rules that need to reason
    about inherited methods or base-class structure should treat that as inconclusive rather than as
    proof that the expected inherited behavior is missing.
    """
    if visiting is None:
        visiting = set()
    if known_external_bases is None:
        known_external_bases = set()
    if class_name in visiting:
        return False
    visiting.add(class_name)

    for base_name in class_to_bases.get(class_name, []):
        if base_name.startswith(("nn.", "torch.nn.")) or base_name in known_external_bases:
            continue
        simple_base_name = _simple_name(base_name)
        if simple_base_name in known_external_bases:
            continue
        if simple_base_name not in class_to_bases:
            return True
        if _base_chain_has_unresolved_import(
            simple_base_name,
            class_to_bases,
            known_external_bases=known_external_bases,
            visiting=visiting,
        ):
            return True
    return False


def iter_pretrained_classes(tree: ast.Module, source_lines: list[str], rule_id: str) -> list[ast.ClassDef]:
    """Yield ClassDef nodes that inherit from PreTrainedModel (transitively), skipping suppressed ones."""
    class_to_bases = _collect_class_bases(tree)
    results = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _inherits_pretrained_model(node.name, class_to_bases):
            continue
        if _has_rule_suppression(source_lines, rule_id, node.lineno):
            continue
        results.append(node)
    return results


def _get_class_assignments(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for item in class_node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            assignments[item.targets[0].id] = item.value
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
            assignments[item.target.id] = item.value
    return assignments


def _class_methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)}


def _top_level_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [node for node in tree.body if isinstance(node, ast.ClassDef)]


def _module_and_method_functions(tree: ast.Module):
    """Yield module-level functions and methods of top-level classes, without walking full bodies."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (item for item in node.body if isinstance(item, ast.FunctionDef))


def _self_attribute_targets(node: ast.AST) -> list[ast.Attribute]:
    """Return the `self.<attr>` targets an assignment statement writes to."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        return []
    return [
        target
        for target in targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    ]


def _function_argument_names(function_node: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in function_node.args.args}
    names.update(arg.arg for arg in function_node.args.kwonlyargs)
    if function_node.args.vararg is not None:
        names.add(function_node.args.vararg.arg)
    if function_node.args.kwarg is not None:
        names.add(function_node.args.kwarg.arg)
    return names


def _function_uses_name(function_node: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        for node in ast.walk(function_node)
    )


def is_self_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a method call on `self`, such as `self.method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method
    )


def is_super_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a call to `super().method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and node.func.attr == method
    )


def _find_config_file(file_path: Path) -> Path | None:
    """Return the companion configuration file for a modeling/modular file, preferring an exact suffix match."""
    model_dir = file_path.parent
    file_name = file_path.name
    for prefix in ("modeling_", "modular_"):
        if file_name.startswith(prefix):
            suffix = file_name[len(prefix) :]
            exact = model_dir / f"configuration_{suffix}"
            if exact.exists():
                return exact
            break

    candidates = sorted(model_dir.glob("configuration_*.py"))
    return candidates[0] if candidates else None


def _parse_config_classes(config_path: Path) -> dict[str, ast.ClassDef] | None:
    """Return the top-level classes defined in a configuration file, keyed by name."""
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(config_path))
    except (OSError, SyntaxError):
        return None

    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _annotated_config_class_name(class_node: ast.ClassDef) -> str | None:
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name) or item.target.id != "config":
            continue

        annotation = item.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation_name = _simple_name(annotation.value)
        else:
            try:
                annotation_name = _simple_name(full_name(annotation))
            except ValueError:
                continue

        if annotation_name.endswith("Config"):
            return annotation_name

    return None


def _resolve_config_class_name_from_modeling_class(
    class_name: str,
    class_to_bases: dict[str, list[str]],
    class_to_assignments: dict[str, dict[str, ast.AST]],
    class_to_nodes: dict[str, ast.ClassDef],
) -> str | None:
    """Resolve the config class a modeling class targets, following local modeling inheritance."""

    def _resolve(name: str, visiting: set[str]) -> str | None:
        if name in visiting:
            return None
        visiting.add(name)

        assignments = class_to_assignments.get(name, {})
        config_class = assignments.get("config_class")
        if config_class is not None:
            if isinstance(config_class, ast.Constant) and isinstance(config_class.value, str):
                return config_class.value
            try:
                return _simple_name(full_name(config_class))
            except ValueError:
                pass

        class_node = class_to_nodes.get(name)
        if class_node is not None:
            annotated_config = _annotated_config_class_name(class_node)
            if annotated_config is not None:
                return annotated_config

        for base_name in class_to_bases.get(name, []):
            if base_name not in class_to_assignments:
                continue
            resolved = _resolve(base_name, visiting)
            if resolved is not None:
                return resolved

        return None

    return _resolve(class_name, set())


def _infer_config_class_name(model_class_name: str, config_class_names: list[str]) -> str | None:
    """Pick the config class whose stem is the longest prefix of the modeling class name."""
    candidates = []
    for config_class_name in config_class_names:
        if not config_class_name.endswith("Config"):
            continue
        config_stem = config_class_name.removesuffix("Config")
        if model_class_name.startswith(config_stem):
            candidates.append((len(config_stem), config_class_name))

    if not candidates:
        return None

    return max(candidates)[1]


def _resolve_target_config_class_name(
    config_classes: dict[str, ast.ClassDef], model_class_name: str, config_class_name: str | None
) -> str | None:
    target_config_name = config_class_name
    if target_config_name not in config_classes:
        target_config_name = _infer_config_class_name(model_class_name, list(config_classes))

    if target_config_name not in config_classes:
        return None

    return target_config_name


def _is_direct_pretrained_config_subclass(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        try:
            if _simple_name(full_name(base)) in {"PreTrainedConfig", "PretrainedConfig"}:
                return True
        except ValueError:
            continue
    return False


def _has_strict_decorator(class_node: ast.ClassDef) -> bool:
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "strict":
            return True

    return False
