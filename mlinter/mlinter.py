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

import argparse
import ast
import datetime
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

from rich import print
from rich.console import Console

from ._helpers import (
    GENERATED_FILE_MARKER,
    MODELS_ROOT,
    TESTS_ROOT,
    Violation,
    _model_dir_name,
    read_file_head,
)
from ._version import __version__


try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback


MODELING_PATTERNS = (
    "modeling_*.py",
    "modular_*.py",
    "configuration_*.py",
    "image_processing_*.py",
    "video_processing_*.py",
    "processing_*.py",
    "feature_extraction_*.py",
    "tokenization_*.py",
    "generation_*.py",
)
# Test files a rule may target, discovered under TESTS_ROOT rather than MODELS_ROOT. Only tokenization
# tests are walked because TRF042 is the only rule that looks at a test file; `test_modeling_*.py` and
# `test_processing_*.py` belong here as soon as a rule targets them. Every rule gates on the file name
# prefix, so widening discovery does not expose existing rules to the files it adds.
TEST_PATTERNS = ("test_tokenization_*.py",)
ALL_PATTERNS = MODELING_PATTERNS + TEST_PATTERNS
FILE_PREFIXES = tuple(pattern.removesuffix("*.py") for pattern in ALL_PATTERNS)
DEFAULT_RULE_SPECS_PATH = Path(__file__).with_name("rules.toml")
RULE_SPECS_VERSION = 1
_RULE_REGISTRY_GLOBALS = (
    "ACTIVE_RULE_SPECS_PATH",
    "RULE_SPECS_HASH",
    "TRF_RULE_SPECS",
    "TRF_RULES",
    "DEFAULT_ENABLED_TRF_RULES",
    "DEPRECATED_TRF_RULES",
    "DEPRECATED_TRF_RULE_SPECS",
    "TRF_MODEL_DIR_ALLOWLISTS",
    "TRF_RULE_CHECKS",
)
ACTIVE_RULE_SPECS_PATH = DEFAULT_RULE_SPECS_PATH
RULE_SPECS_HASH = ""
TRF_RULE_SPECS: dict[str, dict[str, object]] = {}
TRF_RULES: dict[str, str] = {}
DEFAULT_ENABLED_TRF_RULES: set[str] = set()
DEPRECATED_TRF_RULES: frozenset[str] = frozenset()
DEPRECATED_TRF_RULE_SPECS: dict[str, dict[str, object]] = {}
TRF_MODEL_DIR_ALLOWLISTS: dict[str, set[str]] = {}
TRF_RULE_CHECKS: dict[str, Callable[[ast.Module, Path, list[str]], list[Violation]]] = {}


def _deprecation_description(rule_id: str, spec: dict) -> str:
    """The tombstone's one-line prose, for the rule reference page a retired rule keeps."""
    description = spec.get("description")
    if description is None:
        return f"{rule_id} was removed from mlinter."
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Invalid rule spec for {rule_id}: description must be a non-empty string")
    return description


def _read_deprecated_rule_specs(rule_specs_path: Path) -> dict[str, dict[str, object]]:
    """Tombstones a spec file marks with ``deprecated = true``, keyed by rule id.

    Read separately from :func:`_load_rule_specs` so the bundled file can be consulted as the
    authority on what has been retired, even when the CLI is pointed at a different TOML. The
    description is carried along so the docs site can keep publishing a retired rule's page — a
    number that used to fire needs to stay findable by whoever hits it in an old CI log.
    """
    rules = tomllib.loads(rule_specs_path.read_text(encoding="utf-8")).get("rules")
    if not isinstance(rules, dict):
        return {}
    return {
        rule_id: {"description": _deprecation_description(rule_id, spec)}
        for rule_id, spec in rules.items()
        if isinstance(spec, dict) and spec.get("deprecated") is True
    }


def _load_rule_specs(rule_specs_path: Path) -> tuple[dict[str, dict], dict[str, dict[str, object]], str]:
    raw_text = rule_specs_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw_text)
    version = data.get("version")
    if version != RULE_SPECS_VERSION:
        raise ValueError(
            f"Invalid rule spec file: expected version {RULE_SPECS_VERSION}, found {version!r} in {rule_specs_path}"
        )
    rules = data.get("rules")
    if not isinstance(rules, dict):
        raise ValueError(f"Invalid rule spec file: missing [rules] table in {rule_specs_path}")

    # A retired rule has no code left, so a spec file that still describes it as a live rule would
    # silently lint nothing under that id. Fail loudly instead and name the fix.
    stale_rule_ids = sorted(
        rule_id
        for rule_id in BUNDLED_DEPRECATED_TRF_RULES
        if isinstance(rules.get(rule_id), dict) and rules[rule_id].get("deprecated") is not True
    )
    if stale_rule_ids:
        raise ValueError(
            f"Deprecated rule(s) still active in {rule_specs_path}: {', '.join(stale_rule_ids)}. "
            "These rules were removed from mlinter: drop their rule tables from the file, or mark each "
            "with `deprecated = true` and no `default_enabled`."
        )

    required_explanation_keys = {"what_it_does", "why_bad", "diff"}
    specs: dict[str, dict] = {}
    deprecated: dict[str, dict[str, object]] = {}
    for rule_id, spec in rules.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid rule spec for {rule_id}: expected table")

        is_deprecated = spec.get("deprecated", False)
        if not isinstance(is_deprecated, bool):
            raise ValueError(f"Invalid rule spec for {rule_id}: deprecated must be bool")
        if is_deprecated:
            # Deprecated rules keep only a tombstone: a description, no live metadata. That one line
            # is what the retired rule's reference page is built from.
            if spec.get("default_enabled") is True:
                raise ValueError(
                    f"Invalid rule spec for {rule_id}: deprecated rules cannot be enabled, "
                    "drop `default_enabled = true`"
                )
            deprecated[rule_id] = {"description": _deprecation_description(rule_id, spec)}
            continue

        description = spec.get("description")
        default_enabled = spec.get("default_enabled")
        explanation = spec.get("explanation")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Invalid rule spec for {rule_id}: missing non-empty description")
        if not isinstance(default_enabled, bool):
            raise ValueError(f"Invalid rule spec for {rule_id}: default_enabled must be bool")
        if not isinstance(explanation, dict) or not required_explanation_keys.issubset(explanation):
            raise ValueError(f"Invalid rule spec for {rule_id}: incomplete explanation block")
        if any(not isinstance(explanation[key], str) for key in required_explanation_keys):
            raise ValueError(f"Invalid rule spec for {rule_id}: explanation values must be strings")

        allowlist_models = spec.get("allowlist_models", [])
        if not isinstance(allowlist_models, list) or any(not isinstance(item, str) for item in allowlist_models):
            raise ValueError(f"Invalid rule spec for {rule_id}: allowlist_models must be list[str]")

        # A rule that exempts config attributes (TRF041) reads its extra exemptions from here, so a
        # project pointing `--rules-toml` at its own copy can widen the list without an mlinter release.
        ignored_attributes = spec.get("ignored_attributes", [])
        if not isinstance(ignored_attributes, list) or any(not isinstance(item, str) for item in ignored_attributes):
            raise ValueError(f"Invalid rule spec for {rule_id}: ignored_attributes must be list[str]")

        # Some rules are applied on new models, released after cutoff date. We don't have to maintain a long
        # allowlist of old models where the rule is allowed due to BC, if we filter by model addition date!
        cutoff_date = spec.get("cutoff_date")
        if cutoff_date is not None:
            if not isinstance(cutoff_date, str):
                raise ValueError(f"Invalid rule spec for {rule_id}: cutoff_date must be a string")
            try:
                datetime.date.fromisoformat(cutoff_date)
            except ValueError:
                raise ValueError(
                    f"Invalid rule spec for {rule_id}: cutoff_date must be YYYY-MM-DD, got {cutoff_date!r}"
                )

        specs[rule_id] = {
            "description": description,
            "default_enabled": default_enabled,
            "explanation": explanation,
            "allowlist_models": set(allowlist_models),
            "cutoff_date": cutoff_date,
            "ignored_attributes": frozenset(ignored_attributes),
        }

    return specs, deprecated, hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


# The bundled file is the authority on which rule ids have been retired: a project pointing
# `--rules-toml` at its own copy must not be able to resurrect a rule whose code is gone.
BUNDLED_DEPRECATED_TRF_RULE_SPECS = _read_deprecated_rule_specs(DEFAULT_RULE_SPECS_PATH)
BUNDLED_DEPRECATED_TRF_RULES = frozenset(BUNDLED_DEPRECATED_TRF_RULE_SPECS)

CONSOLE = Console(stderr=True)
CACHE_FILENAME = ".mlinter_cache.json"


def _is_rule_allowlisted_for_file(rule_id: str, file_path: Path) -> bool:
    model_name = _model_dir_name(file_path)
    if model_name is None:
        return False
    return model_name in TRF_MODEL_DIR_ALLOWLISTS.get(rule_id, set())


def _find_companion_files(file_path: Path) -> list[Path]:
    """Return companion config files whose content may affect rule results."""
    file_name = file_path.name
    if not (file_name.startswith("modeling_") or file_name.startswith("modular_")):
        return []

    model_dir = file_path.parent
    for prefix in ("modeling_", "modular_"):
        if file_name.startswith(prefix):
            suffix = file_name[len(prefix) :]
            exact = model_dir / f"configuration_{suffix}"
            if exact.exists():
                return [exact]
            break

    return sorted(model_dir.glob("configuration_*.py"))


def _content_hash(text: str, enabled_rules: set[str], companion_files: list[Path] | None = None) -> str:
    h = hashlib.sha256(text.encode("utf-8"))
    h.update(",".join(sorted(enabled_rules)).encode("utf-8"))
    h.update(RULE_SPECS_HASH.encode("utf-8"))
    if companion_files:
        for companion in companion_files:
            try:
                h.update(companion.read_bytes())
            except OSError:
                pass
    return h.hexdigest()


def _cache_dir() -> Path:
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local_appdata:
            return Path(local_appdata) / "mlinter"
        return Path.home() / "AppData" / "Local" / "mlinter"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mlinter"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "mlinter"
    return Path.home() / ".cache" / "mlinter"


def _cache_path() -> Path:
    return _cache_dir() / CACHE_FILENAME


def _load_cache() -> dict[str, str]:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        cache_path = _cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _validate_rule_ids(rule_ids: set[str]) -> set[str]:
    deprecated = sorted(rule_id for rule_id in rule_ids if rule_id in DEPRECATED_TRF_RULES)
    if deprecated:
        raise ValueError(f"Deprecated rule id(s): {', '.join(deprecated)}. These rules were removed from mlinter.")
    unknown = sorted(rule_id for rule_id in rule_ids if rule_id not in TRF_RULES)
    if unknown:
        raise ValueError(f"Unknown rule id(s): {', '.join(unknown)}. Valid rules: {', '.join(sorted(TRF_RULES))}")
    return rule_ids


def _rule_id_from_module_name(name: str) -> str | None:
    if len(name) != 6 or not name.startswith("trf") or not name[3:].isdigit():
        return None
    return name.upper()


def _is_generated_file(path: Path) -> bool:
    """Whether ``path`` is a derived file produced from a ``modular_*.py`` source.

    Generated files (e.g. ``modeling_*.py`` / ``configuration_*.py`` emitted by the modular
    converter) carry an auto-generation banner near the top. They are derived artifacts: the
    modular source is linted instead, so scanning them only produces violations that cannot be
    fixed in place (edits get overwritten on the next generation).
    """
    head = read_file_head(path)
    return head is not None and GENERATED_FILE_MARKER in head


def resolve_search_paths(paths: list[Path]) -> list[Path] | None:
    """Validate the files/directories given on the command line, or None when none were given."""
    if not paths:
        return None
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"No such file or directory: {', '.join(missing)}")
    return paths


def _iter_pattern_matches(directory: Path, patterns: tuple[str, ...]):
    for pattern in patterns:
        yield from directory.rglob(pattern)


def iter_modeling_files(selected_paths: set[Path] | None = None, search_paths: list[Path] | None = None):
    """Yield the files to lint, skipping files generated from a `modular_*.py` source.

    `selected_paths` short-circuits discovery with an already chosen set (what `--changed-only`
    produces). Otherwise files are discovered under `search_paths` when the caller gave any — any
    directory holding model integration files, so a standalone model repo can be linted without
    mirroring the transformers layout — and under the transformers roots relative to the current
    directory when not.
    """
    if selected_paths is not None:
        for path in sorted(selected_paths):
            if path.exists() and not _is_generated_file(path):
                yield path
        return

    if search_paths is not None:
        candidates: set[Path] = set()
        for search_path in search_paths:
            # A file named explicitly is linted as given: rules gate on the file name themselves, so a
            # path the patterns would not have matched simply runs no rules rather than being an error.
            if search_path.is_dir():
                candidates.update(_iter_pattern_matches(search_path, ALL_PATTERNS))
            else:
                candidates.add(search_path)
        for path in sorted(candidates):
            if not _is_generated_file(path):
                yield path
        return

    for root, patterns in ((MODELS_ROOT, MODELING_PATTERNS), (TESTS_ROOT, TEST_PATTERNS)):
        for path in _iter_pattern_matches(root, patterns):
            if not _is_generated_file(path):
                yield path


def colored_error_message(file_path: str, line_number: int, message: str) -> str:
    return f"[bold red]{file_path}[/bold red]:[bold yellow]L{line_number}[/bold yellow]: {message}"


def _path_is_within(file_path: Path, search_path: Path) -> bool:
    """Whether `file_path` is the searched path itself or lives under it."""
    resolved_file_path = file_path.resolve()
    resolved_search_path = search_path.resolve()
    return resolved_file_path == resolved_search_path or resolved_search_path in resolved_file_path.parents


def _is_modeling_candidate(file_path: Path, search_paths: list[Path] | None = None) -> bool:
    """Whether `file_path` is a file this run should lint.

    `search_paths` are the files and directories asked for on the command line; without them a
    candidate is anything under the transformers roots.
    """
    if file_path.suffix != ".py" or not file_path.name.startswith(FILE_PREFIXES):
        return False
    if search_paths is not None:
        return any(_path_is_within(file_path, search_path) for search_path in search_paths)
    return MODELS_ROOT in file_path.parents or TESTS_ROOT in file_path.parents


def _git_name_only(command: list[str]) -> list[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_diff(base_ref: str, triple_dot: bool) -> list[str]:
    diff_operator = "..." if triple_dot else ".."
    range_ref = f"{base_ref}{diff_operator}HEAD"
    return _git_name_only(["git", "diff", "--name-only", "--diff-filter=ACMR", range_ref])


def _git_worktree_changes() -> set[Path]:
    changed_paths = set(_git_name_only(["git", "diff", "--name-only", "--diff-filter=ACMR"]))
    changed_paths.update(_git_name_only(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]))
    changed_paths.update(_git_name_only(["git", "ls-files", "--others", "--exclude-standard"]))
    return {Path(path_str) for path_str in changed_paths}


def get_changed_modeling_files(base_ref: str, search_paths: list[Path] | None = None) -> set[Path]:
    changed_paths = _git_diff(base_ref, triple_dot=True)
    if not changed_paths:
        changed_paths = _git_diff(base_ref, triple_dot=False)

    filtered_paths: set[Path] = set()
    for path in {Path(path_str) for path_str in changed_paths}.union(_git_worktree_changes()):
        if _is_modeling_candidate(path, search_paths):
            filtered_paths.add(path)
    return filtered_paths


CheckFn = Callable[[ast.Module, Path, list[str]], list[Violation]]


def _build_rule_checks(rule_specs: dict[str, dict], deprecated_rules: frozenset[str]) -> dict[str, CheckFn]:
    """Auto-discover check() functions from trf*.py modules in this package."""
    checks: dict[str, CheckFn] = {}
    package_dir = Path(__file__).parent
    for module_path in sorted(package_dir.glob("trf*.py")):
        module_name = module_path.stem
        rule_id = _rule_id_from_module_name(module_name)
        if rule_id is None:
            continue
        if rule_id in deprecated_rules:
            raise ValueError(f"Rule {rule_id} is marked deprecated but {module_name}.py still exists; delete it.")
        if rule_id not in rule_specs:
            raise ValueError(f"Missing rule spec for discovered module {module_name} ({rule_id}).")
        mod = importlib.import_module(f".{module_name}", package=__package__)
        check_fn = getattr(mod, "check", None)
        if not callable(check_fn):
            raise ValueError(f"Module {module_name} must define a check() function.")
        mod.RULE_ID = rule_id
        checks[rule_id] = check_fn

    missing_checks = sorted(set(rule_specs) - set(checks))
    if missing_checks:
        raise ValueError(f"Missing check module(s) for rule id(s): {', '.join(missing_checks)}")
    checks = dict(sorted(checks.items()))
    _apply_rule_module_state(rule_specs, checks)
    return checks


def _apply_rule_module_state(rule_specs: dict[str, dict], checks: dict[str, CheckFn]) -> None:
    """Push the spec fields a rule module reads as module globals onto that module.

    Every field is assigned whether or not the spec carries it: a module is imported once per process,
    so switching to another spec file -- or switching back off one, as `_using_rule_specs` does on exit
    -- has to overwrite what the previous file left behind, not only add to it.
    """
    for rule_id, check_fn in checks.items():
        mod = sys.modules[check_fn.__module__]
        spec = rule_specs.get(rule_id, {})
        if hasattr(mod, "CUTOFF_DATE"):
            mod.CUTOFF_DATE = spec.get("cutoff_date") or ""
        if hasattr(mod, "IGNORED_ATTRIBUTES"):
            mod.IGNORED_ATTRIBUTES = frozenset(spec.get("ignored_attributes", frozenset()))


def _is_rule_id_name(name: str) -> bool:
    return len(name) == 6 and name.startswith("TRF") and name[3:].isdigit()


def _refresh_rule_id_globals() -> None:
    for name in [name for name in globals() if _is_rule_id_name(name)]:
        if name not in TRF_RULE_CHECKS:
            del globals()[name]
    for rule_id in TRF_RULE_CHECKS:
        globals()[rule_id] = rule_id


def _rule_registry_snapshot() -> dict[str, object]:
    return {name: globals()[name] for name in _RULE_REGISTRY_GLOBALS}


def _activate_rule_registry(rule_specs_path: Path) -> None:
    rule_specs, deprecated_specs, rules_hash = _load_rule_specs(rule_specs_path)
    # The bundled tombstones always apply; an active file may add its own on top, and describes any
    # id the two share, since that is the copy the caller pointed us at.
    deprecated_specs = {**BUNDLED_DEPRECATED_TRF_RULE_SPECS, **deprecated_specs}
    deprecated_specs = {rule_id: deprecated_specs[rule_id] for rule_id in sorted(deprecated_specs)}
    deprecated_rules = frozenset(deprecated_specs) | BUNDLED_DEPRECATED_TRF_RULES
    rule_state = {
        "ACTIVE_RULE_SPECS_PATH": rule_specs_path,
        "RULE_SPECS_HASH": rules_hash,
        "TRF_RULE_SPECS": rule_specs,
        "TRF_RULES": {rule_id: spec["description"] for rule_id, spec in rule_specs.items()},
        "DEFAULT_ENABLED_TRF_RULES": {rule_id for rule_id, spec in rule_specs.items() if spec["default_enabled"]},
        "DEPRECATED_TRF_RULES": deprecated_rules,
        "DEPRECATED_TRF_RULE_SPECS": deprecated_specs,
        "TRF_MODEL_DIR_ALLOWLISTS": {
            rule_id: spec["allowlist_models"] for rule_id, spec in rule_specs.items() if spec["allowlist_models"]
        },
        "TRF_RULE_CHECKS": _build_rule_checks(rule_specs, deprecated_rules),
    }
    globals().update(rule_state)
    _refresh_rule_id_globals()


@contextmanager
def _using_rule_specs(rule_specs_path: Path):
    previous_state = _rule_registry_snapshot()
    _activate_rule_registry(rule_specs_path)
    try:
        yield
    finally:
        globals().update(previous_state)
        _refresh_rule_id_globals()
        # Restoring the globals is not enough: the rule modules still hold what the custom file wrote,
        # and they are shared process-wide.
        _apply_rule_module_state(TRF_RULE_SPECS, TRF_RULE_CHECKS)


_activate_rule_registry(DEFAULT_RULE_SPECS_PATH)


def analyze_file(file_path: Path, text: str, enabled_rules: set[str] | None = None) -> list[Violation]:
    if enabled_rules is None:
        enabled_rules = DEFAULT_ENABLED_TRF_RULES

    violations: list[Violation] = []
    source_lines = text.splitlines()
    tree = ast.parse(text, filename=str(file_path))

    for rule_id, check_fn in TRF_RULE_CHECKS.items():
        if rule_id in enabled_rules:
            for v in check_fn(tree, file_path, source_lines):
                violations.append(
                    Violation(
                        file_path=v.file_path,
                        line_number=v.line_number,
                        rule_id=rule_id,
                        message=v.message,
                    )
                )

    return [
        violation
        for violation in violations
        if not (
            violation.rule_id is not None and _is_rule_allowlisted_for_file(violation.rule_id, violation.file_path)
        )
    ]


def format_violation(violation: Violation) -> str:
    return colored_error_message(str(violation.file_path), violation.line_number, violation.message)


def emit_violation(violation: Violation, github_annotations: bool):
    if github_annotations:
        print(
            f"::error file={violation.file_path},line={violation.line_number}::{violation.message}",
            file=sys.stderr,
        )
        return

    print(format_violation(violation), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"mlinter {__version__}")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        metavar="PATH",
        help="Files or directories to check. A directory is searched recursively for model integration "
        f"files ({', '.join(ALL_PATTERNS)}); a file is checked as given. Use this to lint a standalone "
        "model repository, which does not mirror the transformers layout. Defaults to "
        f"{MODELS_ROOT} and {TESTS_ROOT} relative to the current directory.",
    )
    parser.add_argument(
        "--rules-toml",
        type=Path,
        default=DEFAULT_RULE_SPECS_PATH,
        help="Path to a rules TOML file. Defaults to the bundled mlinter/rules.toml.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only check changed model integration files compared to --base-ref, plus local worktree changes.",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base git ref used with --changed-only (default: origin/main).",
    )
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help="Emit GitHub Actions annotation format output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive progress animation.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the lint cache and re-check every file.",
    )
    parser.add_argument(
        "--enable-all-trf-rules",
        action="store_true",
        help="Enable all TRF rules (defaults already enable most).",
    )
    parser.add_argument(
        "--enable-rules",
        default="",
        help="Comma-separated TRF rule ids to enable in addition to defaults (e.g. TRF001,TRF002).",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List available TRF rules and exit.",
    )
    parser.add_argument(
        "--rule",
        default="",
        help="Show detailed docs for one rule id (e.g. TRF001) and exit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write all findings to FILE as JSON in addition to the normal output. "
        "The file is written even when there are no violations (empty findings list). "
        "Exit code is not affected. "
        "Used by the transformers CI to upload findings as an artifact and post "
        "them as inline review comments on the triggering PR.",
    )
    return parser.parse_args()


def should_show_progress(args: argparse.Namespace) -> bool:
    return (not args.no_progress) and (not args.github_annotations) and sys.stderr.isatty()


def resolve_enabled_rules(args: argparse.Namespace) -> set[str]:
    if args.enable_all_trf_rules:
        return _validate_rule_ids(set(TRF_RULES))

    enabled_rules = set(DEFAULT_ENABLED_TRF_RULES)
    if args.enable_rules.strip():
        enabled_rules.update(rule_id.strip() for rule_id in args.enable_rules.split(",") if rule_id.strip())
    return _validate_rule_ids(enabled_rules)


def format_rule_summary(rule_id: str) -> str:
    spec = TRF_RULE_SPECS[rule_id]
    default_label = "enabled" if spec["default_enabled"] else "disabled"
    return f"{rule_id}: {spec['description']} (default: {default_label})"


def format_rule_details(rule_id: str) -> str:
    spec = TRF_RULE_SPECS[rule_id]
    explanation = cast(dict[str, str], spec["explanation"])
    return "\n".join(
        [
            f"### {rule_id}",
            "",
            f"{explanation['what_it_does']} {explanation['why_bad']}",
            "",
            "```diff",
            explanation["diff"].strip(),
            "```",
        ]
    )


def render_rules_reference() -> str:
    return "\n\n".join(format_rule_details(rule_id) for rule_id in sorted(TRF_RULE_SPECS)) + "\n"


def maybe_handle_rule_docs_cli(args: argparse.Namespace) -> bool:
    if args.list_rules:
        for rule_id in sorted(TRF_RULE_SPECS):
            print(format_rule_summary(rule_id))
        return True

    if args.rule:
        rule_id = args.rule.strip().upper()
        _validate_rule_ids({rule_id})
        print(format_rule_details(rule_id))
        return True

    return False


def warn_about_search_paths(search_paths: list[Path], modeling_files: list[Path], warn_when_empty: bool) -> None:
    """Explain a run that checked nothing, so an empty run never reads as a clean one.

    Every rule gates on the file name, so a file whose name carries none of the known prefixes runs no
    rules at all and would otherwise be reported as `OK`. `warn_when_empty` is False under
    `--changed-only`, where finding no file means nothing changed rather than nothing to check.
    """
    for search_path in search_paths:
        if search_path.is_file() and not search_path.name.startswith(FILE_PREFIXES):
            print(
                f"Warning: {search_path} is not a model integration file "
                f"({', '.join(ALL_PATTERNS)}), so no rule applies to it.",
                file=sys.stderr,
            )
    if warn_when_empty and not modeling_files:
        print(
            f"Warning: no model integration file found in {', '.join(str(path) for path in search_paths)}.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    previous_state = _rule_registry_snapshot()
    try:
        _activate_rule_registry(args.rules_toml)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"Failed to load rule specs from {args.rules_toml}: {exc}", file=sys.stderr)
        return 2

    try:
        if maybe_handle_rule_docs_cli(args):
            return 0

        violations: list[Violation] = []
        enabled_rules = resolve_enabled_rules(args)
        search_paths = resolve_search_paths(args.paths)
        selected_paths = get_changed_modeling_files(args.base_ref, search_paths) if args.changed_only else None

        modeling_files = list(iter_modeling_files(selected_paths, search_paths))
        if search_paths is not None:
            warn_about_search_paths(search_paths, modeling_files, warn_when_empty=selected_paths is None)

        show_progress = should_show_progress(args)
        status_ctx = (
            CONSOLE.status(f"[bold blue]Checking modeling structure ({len(modeling_files)} files)...[/bold blue]")
            if show_progress
            else nullcontext()
        )

        use_cache = not args.no_cache
        cache = _load_cache() if use_cache else {}
        new_cache: dict[str, str] = {}

        with status_ctx:
            for file_path in modeling_files:
                try:
                    text = file_path.read_text(encoding="utf-8")
                    # Absolute: the cache is shared by every checkout, and a relative path such as
                    # `modeling_llada.py` names a different file in each standalone model repo.
                    file_key = str(file_path.resolve())
                    digest = _content_hash(text, enabled_rules, _find_companion_files(file_path))

                    if use_cache and cache.get(file_key) == digest:
                        new_cache[file_key] = digest
                        continue

                    file_violations = analyze_file(file_path, text, enabled_rules=enabled_rules)
                    violations.extend(file_violations)

                    if not file_violations:
                        new_cache[file_key] = digest
                except Exception as exc:
                    violations.append(
                        Violation(file_path=file_path, line_number=1, message=f"failed to parse ({exc}).")
                    )

        if use_cache:
            _save_cache(new_cache)

        violations = sorted(violations, key=lambda v: (str(v.file_path), v.line_number, v.message))

        if args.output_json is not None:
            # Exclude parse-error sentinels (rule_id=None) from the rules summary.
            rules_used = sorted({v.rule_id for v in violations if v.rule_id})
            payload = {
                # May include parse-error sentinels with rule=null (no associated rule).
                "findings": [
                    {"path": str(v.file_path), "line": v.line_number, "rule": v.rule_id, "message": v.message}
                    for v in violations
                ],
                "rules": {
                    rule: {
                        "description": TRF_RULE_SPECS[rule]["description"],
                        # Validated as a dict[str, str] by _load_rule_specs.
                        "why_bad": cast(dict[str, str], TRF_RULE_SPECS[rule]["explanation"])["why_bad"],
                        "diff": cast(dict[str, str], TRF_RULE_SPECS[rule]["explanation"])["diff"],
                    }
                    for rule in rules_used
                },
            }
            args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if len(violations) > 0:
            for violation in violations:
                emit_violation(violation, github_annotations=args.github_annotations)
            print(f"Found {len(violations)} modeling structure violation(s).", file=sys.stderr)
            return 1

        print("OK")
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        globals().update(previous_state)
        _refresh_rule_id_globals()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
