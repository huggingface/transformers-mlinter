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

"""TRF041: A config-gated branch must document which models diverge, with a `# CODEPATH:` comment."""

import ast
import re
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption
IGNORED_ATTRIBUTES: frozenset[str] = frozenset()  # Set by discovery from rules.toml ignored_attributes

# Config fields that gate framework plumbing rather than architecture. A branch on one of them exists in
# the same shape in every model that has the head or the feature -- `problem_type` picks a loss function,
# `hidden_act` looks up an activation, `use_cache` asks whether to keep a cache -- so no checkpoint takes
# one side and no checkpoint the other, and there is nothing for a `# CODEPATH:` note to name. Exempting
# them here is what the rule used to ask every model to write by hand as a `# trf-ignore: TRF041` list.
# A `# CODEPATH:` note on such a branch stays legal; it just stops being required.
DEFAULT_EXEMPT_ATTRIBUTES = frozenset(
    {
        # head and loss selection
        "problem_type",
        "num_labels",
        "loss_type",
        "classifier_dropout",
        # activation and initialisation lookups
        "hidden_act",
        "initializer_range",
        # special token ids
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        # generic PretrainedConfig / PreTrainedModel plumbing
        "tie_word_embeddings",
        "_attn_implementation",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "use_cache",
        "is_decoder",
        "is_encoder_decoder",
        "add_cross_attention",
        "chunk_size_feed_forward",
        "gradient_checkpointing",
        "torch_dtype",
        "architectures",
        "_name_or_path",
    }
)

# `summary_type`, `summary_use_proj`, `summary_activation`, `summary_proj_to_labels`, `summary_first_dropout`:
# the whole family configures the same generic sequence-summary head, so it is exempted by prefix.
EXEMPT_ATTRIBUTE_PREFIXES = ("summary_",)

# Callables whose only effect is to tell the user something. A branch whose body is one of these does not
# fork the graph, so it has no second checkpoint to name.
WARN_FUNCTIONS = frozenset({"warn", "warn_explicit", "warning", "warning_once", "info", "debug", "error", "critical"})

# Borrowed from Rust's `// SAFETY:` convention: the construct stays legal, but the author has to write
# down the reasoning that makes it legal. Here the reasoning is which checkpoints take which path.
MARKER = "# codepath:"


def _normalise_attribute(dotted: str) -> str:
    """`self.config.problem_type`, `config.problem_type` and `problem_type` all name the same field."""
    return dotted.strip().removeprefix("self.").removeprefix("config.")


def _is_exempt_attribute(attribute: str) -> bool:
    """Whether a config field is plumbing the rule never asks about."""
    field = _normalise_attribute(attribute)
    return field in DEFAULT_EXEMPT_ATTRIBUTES or field.startswith(EXEMPT_ATTRIBUTE_PREFIXES)


def _file_scoped_ignores(source_lines: list[str], rule_id: str) -> set[str]:
    """Config attributes exempted by a module-level `# trf-ignore: <RULE> <attr>, ...` directive.

    `DEFAULT_EXEMPT_ATTRIBUTES` covers the plumbing fields every model shares; this is how a model
    exempts one of its own. When the same field gates the same non-divergent branch in several places
    in a file, a per-branch suppression would mean repeating one comment a dozen times: a directive at
    column 0 names the field once, and keeps the exemption reviewable in the diff instead of buried in
    a global config. A bare directive with no attributes is left alone for `_has_rule_suppression` to
    handle, so this never widens an existing per-line suppression into a whole-file mute.
    """
    # ponytail: lives here until a second rule wants file-scoped subjects, then lift into _helpers.
    token = f"trf-ignore: {rule_id}".lower()
    ignored: set[str] = set()
    for line in source_lines:
        if not line.startswith("#") or token not in line.lower():
            continue
        _, _, tail = line.lower().partition(token)
        for word in tail.replace(",", " ").split():
            # Stop at the first word that is not an attribute path so trailing prose does not leak in.
            if not re.fullmatch(r"[a-z_][a-z0-9_.]*", word):
                break
            ignored.add(_normalise_attribute(word))
    ignored.discard("")
    return ignored


def _is_default_coalesce(node: ast.expr) -> bool:
    """Whether `node` is `X if X is not None else fallback`, i.e. a default rather than a fork.

    When the field under test is itself one of the two results, the expression cannot fork the graph:
    it yields the field when set and a fallback when not, which is `getattr(config, x, default)` spelled
    long. No checkpoint diverges, so there is no path to name and a `# CODEPATH:` note would be noise.
    A gate that merely mentions None is not enough — `config.vision_config is not None` selects a whole
    extra tower, and that one has to keep explaining itself.
    """
    if not isinstance(node, ast.IfExp):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    if not (isinstance(comparator, ast.Constant) and comparator.value is None):
        return False
    if not isinstance(test.ops[0], (ast.Is, ast.IsNot)):
        return False
    try:
        target = full_name(test.left)
    except ValueError:
        return False
    for result in (node.body, node.orelse):
        try:
            if full_name(result) == target:
                return True
        except ValueError:
            continue
    return False


def _is_warn_call(node: ast.stmt) -> bool:
    """Whether `node` is a bare `logger.warning(...)` / `warnings.warn(...)`-style call."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return isinstance(func, ast.Attribute) and func.attr in WARN_FUNCTIONS


def _is_message_assignment(node: ast.stmt) -> bool:
    """Whether `node` binds a local name, as guards do when they build an error message first."""
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name)
    if not isinstance(node, ast.Assign):
        return False
    return all(isinstance(target, ast.Name) for target in node.targets)


def _is_guard_branch(node: ast.If) -> bool:
    """Whether `node` rejects or warns about a configuration instead of forking the graph.

    `if config.num_experts and not config.expert_capacity: raise ValueError(...)` is validation: one side
    aborts, so every checkpoint that gets past the branch took the same path and there is no divergence to
    write down. The same holds for a body that only logs. An `else` disqualifies it -- that is a real fork,
    with the guard shape borrowed for one of its two arms.
    """
    if node.orelse or not node.body:
        return False
    if any(isinstance(statement, ast.Raise) for statement in node.body):
        # Statements around the raise can only be message building: the branch leaves by exception.
        return all(
            isinstance(statement, ast.Raise) or _is_warn_call(statement) or _is_message_assignment(statement)
            for statement in node.body
        )
    return all(_is_warn_call(statement) for statement in node.body)


def _config_attributes(test: ast.expr) -> list[str]:
    """Every `config.*` / `self.config.*` attribute the branch condition reads.

    All of them, not just the first: `if config.problem_type and config.use_cache` gates on two fields,
    and a file-scoped exemption for one of them must not carry the other along. Duplicates are dropped
    so a condition reading the same field twice reports it once.
    """
    attributes: list[str] = []
    for node in ast.walk(test):
        if not isinstance(node, ast.Attribute):
            continue
        try:
            dotted = full_name(node)
        except ValueError:
            continue
        if dotted.removeprefix("self.").startswith("config.") and dotted not in attributes:
            attributes.append(dotted)
    return attributes


def _has_marker(source_lines: list[str], line_number: int) -> bool:
    """Whether a `# CODEPATH:` comment sits on the branch line or on the comment block above it."""
    index = line_number - 1
    if 0 <= index < len(source_lines) and MARKER in source_lines[index].lower():
        return True
    # Walk up through a contiguous comment block so the note can head a multi-line explanation.
    index -= 1
    while 0 <= index < len(source_lines):
        stripped = source_lines[index].strip()
        if MARKER in stripped.lower():
            return True
        if stripped.startswith("#"):
            index -= 1
            continue
        break
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if is_exempt_by_cutoff(file_path, CUTOFF_DATE):
        return []

    # The spec-level list extends the built-in exempt set for a project that keeps its own rules.toml.
    ignored_attributes = _file_scoped_ignores(source_lines, RULE_ID) | {
        _normalise_attribute(attribute) for attribute in IGNORED_ATTRIBUTES
    }

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if _is_guard_branch(node):
                continue
            kind = "branch"
        elif isinstance(node, ast.IfExp):
            if _is_default_coalesce(node):
                continue
            kind = "conditional expression"
        else:
            continue

        attributes = _config_attributes(node.test)
        if not attributes:
            continue
        # Only the attributes still on the hook can be reported: a branch gated on an exempt field and
        # a live one still has to name its checkpoints, so it is skipped only when every field is exempt.
        reportable = [
            a for a in attributes if _normalise_attribute(a) not in ignored_attributes and not _is_exempt_attribute(a)
        ]
        if not reportable:
            continue
        attribute = reportable[0]
        if _has_marker(source_lines, node.lineno):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {kind} on `{attribute}` has no `# CODEPATH:` note. "
                    "Add one naming the checkpoints that take each path, or delete the branch."
                ),
            )
        )
    return violations
