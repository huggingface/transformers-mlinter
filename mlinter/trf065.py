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

"""TRF065: Processors must not subclass the shared modality kwargs (`ImagesKwargs`, `TextKwargs`, ...)."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _simple_name, full_name


RULE_ID = ""  # Set by discovery

CHECKED_FILE_PREFIXES = ("processing_", "modular_")
# Shared modality kwargs, keyed by the `ProcessingKwargs` field that holds them.
MODALITY_KWARGS = {
    "ImagesKwargs": "images_kwargs",
    "TextKwargs": "text_kwargs",
    "AudioKwargs": "audio_kwargs",
    "VideosKwargs": "videos_kwargs",
}
# Backward-compatibility shims that keep accepting the old nested keys are named `Deprecated<Model><Modality>Kwargs`.
DEPRECATED_PREFIX = "Deprecated"
# A modular file also holds image, video and audio processors, which legitimately extend the shared kwargs.
# Those have to follow the sub-processor naming so they can be told apart from processor-level subclasses.
PROCESSOR_KWARGS_SUFFIXES_BY_BASE = {
    "ImagesKwargs": ("ImageProcessorKwargs",),
    "VideosKwargs": ("VideoProcessorKwargs", "VideoProcessorInitKwargs"),
    "AudioKwargs": ("AudioProcessorKwargs",),
}


def _base_names(class_node: ast.ClassDef) -> list[str]:
    names = []
    for base in class_node.bases:
        try:
            names.append(_simple_name(full_name(base)))
        except ValueError:
            continue
    return names


def _modality_base(class_node: ast.ClassDef) -> str | None:
    """The shared modality kwargs `class_node` extends, directly or through another model's `<Model>ImagesKwargs`."""
    for base in _base_names(class_node):
        for modality_base in MODALITY_KWARGS:
            if base == modality_base or (base.endswith(modality_base) and not base.startswith(DEPRECATED_PREFIX)):
                return modality_base
    return None


def _subclass_message(class_name: str, modality_base: str) -> str:
    field = MODALITY_KWARGS[modality_base]
    return (
        f"{RULE_ID}: `{class_name}` subclasses `{modality_base}`. Model-specific processor kwargs belong at the top "
        f"level of `<Model>ProcessorKwargs`, not nested under `{field}`; defaults go in a `{field}` class attribute "
        "of the processor. If the class only keeps old nested keys working, name it "
        f"`{DEPRECATED_PREFIX}<Model>...Kwargs`. Image, video and audio processor kwargs live in "
        "`image_processing_*.py`/`video_processing_*.py`/`audio_processing_*.py`."
    )


def _modular_subclass_message(class_name: str, modality_base: str) -> str:
    suffixes = PROCESSOR_KWARGS_SUFFIXES_BY_BASE.get(modality_base, ())
    allowed = " or ".join(f"`<Model>{suffix}`" for suffix in suffixes)
    sub_processor_hint = f"If it belongs to an image, video or audio processor, name it {allowed}. " if allowed else ""
    return (
        f"{RULE_ID}: `{class_name}` subclasses `{modality_base}` in a modular file. {sub_processor_hint}"
        "Processors must not extend the shared modality kwargs: put model-specific kwargs at the top level of "
        f"`<Model>ProcessorKwargs`, or name a backward-compatibility shim `{DEPRECATED_PREFIX}<Model>...Kwargs`."
    )


def _iter_violations(tree: ast.Module, file_name: str):
    is_modular = file_name.startswith("modular_")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        modality_base = _modality_base(node)
        if modality_base is not None and not node.name.startswith(DEPRECATED_PREFIX):
            if not is_modular:
                yield node.lineno, _subclass_message(node.name, modality_base)
            elif not node.name.endswith(PROCESSOR_KWARGS_SUFFIXES_BY_BASE.get(modality_base, ())):
                yield node.lineno, _modular_subclass_message(node.name, modality_base)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    file_name = file_path.name
    if not file_name.startswith(CHECKED_FILE_PREFIXES):
        return []

    return [
        Violation(file_path=file_path, line_number=line_number, message=message)
        for line_number, message in _iter_violations(tree, file_name)
        if not _has_rule_suppression(source_lines, RULE_ID, line_number)
    ]
