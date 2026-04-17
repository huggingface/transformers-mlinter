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

"""Supported public API for transformers-mlinter.

Import public helpers from ``mlinter`` instead of implementation modules such as
``mlinter.mlinter`` or ``mlinter._helpers``.
"""

from ._helpers import (
    MODELS_ROOT,
    Violation,
    full_name,
    is_self_method_call,
    is_super_method_call,
    iter_pretrained_classes,
)
from ._helpers import (
    _collect_class_bases as collect_class_bases,
)
from ._helpers import (
    _has_rule_suppression as has_rule_suppression,
)
from ._helpers import (
    _inherits_pretrained_model as inherits_pretrained_model,
)
from ._helpers import (
    _model_dir_name as model_dir_name,
)
from .mlinter import (
    DEFAULT_ENABLED_TRF_RULES,
    TRF_MODEL_DIR_ALLOWLISTS,
    TRF_RULE_CHECKS,
    TRF_RULE_SPECS,
    TRF_RULES,
    analyze_file,
    colored_error_message,
    emit_violation,
    format_rule_details,
    format_rule_summary,
    format_violation,
    get_changed_modeling_files,
    iter_modeling_files,
    main,
    maybe_handle_rule_docs_cli,
    parse_args,
    render_rules_reference,
    resolve_enabled_rules,
    should_show_progress,
)
from .mlinter import (
    _is_rule_allowlisted_for_file as is_rule_allowlisted_for_file,
)


__all__ = [
    "DEFAULT_ENABLED_TRF_RULES",
    "MODELS_ROOT",
    "TRF_MODEL_DIR_ALLOWLISTS",
    "TRF_RULE_CHECKS",
    "TRF_RULE_SPECS",
    "TRF_RULES",
    "Violation",
    "analyze_file",
    "collect_class_bases",
    "colored_error_message",
    "emit_violation",
    "format_rule_details",
    "format_rule_summary",
    "format_violation",
    "full_name",
    "get_changed_modeling_files",
    "has_rule_suppression",
    "inherits_pretrained_model",
    "is_rule_allowlisted_for_file",
    "is_self_method_call",
    "is_super_method_call",
    "iter_modeling_files",
    "iter_pretrained_classes",
    "main",
    "maybe_handle_rule_docs_cli",
    "model_dir_name",
    "parse_args",
    "render_rules_reference",
    "resolve_enabled_rules",
    "should_show_progress",
]

for rule_id in sorted(TRF_RULE_CHECKS):
    globals()[rule_id] = rule_id
    __all__.append(rule_id)
del rule_id
