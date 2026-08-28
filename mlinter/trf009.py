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
from pathlib import Path

from ._helpers import MODELS_ROOT, Violation, _has_rule_suppression, _known_model_dirs, _model_dir_name


RULE_ID = ""  # Set by discovery

# The model-directory file kinds the one-file-one-definition policy covers: the files that make up a
# model's shipped implementation. Three other kinds live in a model directory and are deliberately
# absent. `modular_*.py`, because inheriting another model's classes is exactly what a modular file
# is for, and the converter flattens those imports away in the file it generates. `convert_*.py`,
# because a conversion script is a one-off tool rather than part of the model: it legitimately builds
# a checkpoint out of whatever the original release used, and transformers has 256 cross-model
# imports in those scripts that are all working as intended. And `__init__.py`, because the handful
# of cross-model aliases it carries (`from ..roberta.tokenization_roberta import RobertaTokenizer as
# BartTokenizer`) are the same coupling already reported on the tokenizer file itself, so checking
# both would report one problem twice. Test files are out for the practical reason that a test
# importing another model's test case is the normal way to write one.
_CHECKED_PREFIXES = (
    "modeling_",
    "configuration_",
    "processing_",
    "image_processing_",
    "video_processing_",
    "feature_extraction_",
    "tokenization_",
    "generation_",
)

# Model directories every model is meant to reach through rather than around. `auto` holds the mappings
# a composite model resolves its sub-models with, and `timm_wrapper` is the adapter that exposes any
# timm backbone as a transformers model, so importing `TimmWrapperConfig` is of a kind with importing
# `AutoConfig`: it names the shared entry point, not another model's implementation.
_SHARED_MODEL_DIRS = frozenset({"auto", "timm_wrapper"})

# Class names defined by a model directory, keyed by its resolved path. A lint run resolves the same
# few directories over and over (one entry per model a file imports from), and each miss costs a
# parse of every source file in that directory, so the answers are kept for the life of the process.
# The key is the path rather than the directory name because `MODELS_ROOT` is not a constant in
# practice: a test points it at a temporary tree, and a caller using mlinter as a library can change
# working directory between runs. Two different roots that both hold a `clip` directory must not
# answer each other's lookups, so `_reset_defined_class_names` exists for the remaining case, where
# the contents behind one path change while the process lives.
_DEFINED_CLASS_NAMES: dict[Path, frozenset[str]] = {}


def _reset_defined_class_names() -> None:
    """Drop the memoized directory scans, for a caller that has changed what is on disk."""
    _DEFINED_CLASS_NAMES.clear()


def _defined_class_names(model_dir: str) -> frozenset[str]:
    """Every class name defined at module level by the sources in `MODELS_ROOT/model_dir`."""
    directory = (MODELS_ROOT / model_dir).resolve()
    if directory not in _DEFINED_CLASS_NAMES:
        names: set[str] = set()
        try:
            source_files = sorted(directory.glob("*.py"))
        except OSError:
            source_files = []
        for source_file in source_files:
            try:
                module = ast.parse(source_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                # A directory that cannot be read or parsed resolves no names, which leaves the
                # import unreported rather than reported on a guess.
                continue
            names.update(node.name for node in module.body if isinstance(node, ast.ClassDef))
        _DEFINED_CLASS_NAMES[directory] = frozenset(names)
    return _DEFINED_CLASS_NAMES[directory]


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
        key=lambda model_dir: len(model_dir),
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
    # The `auto` package is exempt as an importer as well as as a target: naming every model's
    # classes is the whole job of `configuration_auto.py` and `tokenization_auto.py`, so the imports
    # it makes are the mapping layer working as intended rather than one model reaching into another.
    if current_model == "auto":
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
