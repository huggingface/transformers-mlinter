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

"""TRF009: model files should avoid importing implementation code from another model package."""

import ast
import re
from pathlib import Path

from ._helpers import MODELS_ROOT, Violation, _has_rule_suppression, _known_model_dirs, _model_dir_name


RULE_ID = ""  # Set by discovery

# The model-directory file kinds the one-file-one-definition policy covers. `modular_*.py` is
# deliberately absent: inheriting another model's classes is exactly what a modular file is for, and
# the converter flattens those imports away in the file it generates. Test files are absent for the
# same practical reason -- a test importing another model's test case is the normal way to write one.
_CHECKED_PREFIXES = (
    "modeling_",
    "configuration_",
    "processing_",
    "image_processing_",
    "video_processing_",
    "feature_extraction_",
    "tokenization_",
)

# Model directories every model is meant to reach through rather than around. `auto` holds the mappings
# a composite model resolves its sub-models with, and `timm_wrapper` is the adapter that exposes any
# timm backbone as a transformers model, so importing `TimmWrapperConfig` is of a kind with importing
# `AutoConfig`: it names the shared entry point, not another model's implementation.
_SHARED_MODEL_DIRS = frozenset({"auto", "timm_wrapper"})

_CLASS_DEF_RE = re.compile(r"^class (\w+)", re.MULTILINE)

# Class names defined by a model directory, keyed by directory name. A lint run resolves the same
# few directories over and over (one entry per model a file imports from), and each miss costs a
# read of every source file in that directory, so the answers are kept for the life of the process.
_DEFINED_CLASS_NAMES: dict[str, frozenset[str]] = {}


def _defined_class_names(model_dir: str) -> frozenset[str]:
    """Every class name defined at module level by the sources in `MODELS_ROOT/model_dir`."""
    if model_dir not in _DEFINED_CLASS_NAMES:
        names: set[str] = set()
        try:
            source_files = sorted((MODELS_ROOT / model_dir).glob("*.py"))
        except OSError:
            source_files = []
        for source_file in source_files:
            try:
                text = source_file.read_text(encoding="utf-8")
            except OSError:
                continue
            names.update(_CLASS_DEF_RE.findall(text))
        _DEFINED_CLASS_NAMES[model_dir] = frozenset(names)
    return _DEFINED_CLASS_NAMES[model_dir]


def _model_dir_defining(name: str, known_models: set[str]) -> str | None:
    """The model directory that defines the public class `name`, or None when none does.

    `from transformers import CLIPModel` names a class without naming the package it comes from, so
    the package has to be recovered from the class name. Transformers names a model's classes after
    its directory (`clip` -> `CLIPModel`, `qwen2_5_vl` -> `Qwen2_5_VLForConditionalGeneration`) but
    leaves the casing of that prefix to the model author, so candidates are matched on lowercased,
    underscore-stripped text, longest directory first (`bitnet` wins over `bit`).

    A prefix match alone is not enough: `BitsAndBytesConfig` starts like the `bit` directory without
    being model code. Each candidate is confirmed against the classes that directory actually
    defines, so a shared library class that merely reads like a model prefix is left alone -- and so
    is anything this rule cannot resolve, such as a name imported outside a transformers checkout,
    where there are no directories to match against.
    """
    flattened = name.replace("_", "").lower()
    candidates = sorted(
        (model_dir for model_dir in known_models if flattened.startswith(model_dir.replace("_", "").lower())),
        key=len,
        reverse=True,
    )
    return next((candidate for candidate in candidates if name in _defined_class_names(candidate)), None)


def _imported_model_from_module(module: str, level: int, known_models: set[str]) -> str | None:
    """The model directory an `import from` reaches into, from the module part of the statement."""
    if level == 0:
        if module.startswith("transformers.models."):
            return module.split("transformers.models.", 1)[1].split(".", 1)[0]
        return None
    if level < 2:
        return None
    # `from ..clip.modeling_clip import X` names the model directory first, while
    # `from ...models.clip.modeling_clip import X` walks up past it and names the package too.
    parts = module.split(".")
    if parts[0] == "models":
        parts = parts[1:]
    return parts[0] if parts and parts[0] in known_models else None


def _is_exempt(imported_model: str, current_model: str) -> bool:
    """Whether importing from `imported_model` is allowed from inside `current_model`."""
    return imported_model == current_model or imported_model in _SHARED_MODEL_DIRS


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(_CHECKED_PREFIXES):
        return []

    current_model = _model_dir_name(file_path)
    if current_model is None:
        return []

    violations: list[Violation] = []
    known_models = _known_model_dirs()

    def report(node: ast.stmt, imported_model: str) -> None:
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {file_path.name} imports implementation code from "
                    f"`{imported_model}`. Keep model code local to this model's own files; only "
                    f"modular_*.py may build on another model."
                ),
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue

            imported_model = _imported_model_from_module(node.module, node.level, known_models)
            if imported_model is None and node.level == 0 and node.module in {"transformers", "transformers.models"}:
                # `from transformers import CLIPModel` reaches another model through the public API
                # rather than through its package path, and each imported name resolves on its own.
                for alias in node.names:
                    imported_model = _model_dir_defining(alias.name, known_models)
                    if imported_model is not None and not _is_exempt(imported_model, current_model):
                        report(node, imported_model)
                continue

            if imported_model is None or _is_exempt(imported_model, current_model):
                continue

            report(node, imported_model)
            continue

        if isinstance(node, ast.Import):
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue

            for alias in node.names:
                if not alias.name.startswith("transformers.models."):
                    continue
                remaining = alias.name.split("transformers.models.", 1)[1]
                imported_model = remaining.split(".", 1)[0]
                if _is_exempt(imported_model, current_model):
                    continue
                report(node, imported_model)

    return violations
