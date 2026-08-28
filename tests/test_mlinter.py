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

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlinter as public_api
from mlinter import _helpers as _helpers_mod
from mlinter import _version as _version_mod
from mlinter import mlinter
from tests.rule_test_utils import LICENSE_HEADER


def _write_custom_rules_toml(
    tmp_dir: Path, *, trf001_description: str | None = None, trf001_default_enabled: bool | None = None
) -> Path:
    text = mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8")
    if trf001_description is not None:
        text = text.replace(
            'description = "config_class on <Model>PreTrainedModel should match <Model>Config naming."',
            f'description = "{trf001_description}"',
            1,
        )
    if trf001_default_enabled is not None:
        replacement = "true" if trf001_default_enabled else "false"
        text = text.replace("default_enabled = true", f"default_enabled = {replacement}", 1)

    custom_rules_path = tmp_dir / "custom_rules.toml"
    custom_rules_path.write_text(text, encoding="utf-8")
    return custom_rules_path


def _unwrapped(text: str) -> str:
    """CLI output with rich's console wrapping collapsed, so assertions survive any terminal width."""
    return " ".join(text.split())


def _write_rules_toml_without_cutoffs(tmp_dir: Path) -> Path:
    """The bundled rule specs with every `cutoff_date` line dropped, as a project overriding them would."""
    lines = mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    custom_rules_path = tmp_dir / "no_cutoff_rules.toml"
    custom_rules_path.write_text(
        "".join(line for line in lines if not line.startswith("cutoff_date")), encoding="utf-8"
    )
    return custom_rules_path


def _module_cutoffs() -> dict[str, str]:
    """What each rule module currently holds in its `CUTOFF_DATE` global."""
    cutoffs = {}
    for rule_id, check_fn in mlinter.TRF_RULE_CHECKS.items():
        module = sys.modules[check_fn.__module__]
        if hasattr(module, "CUTOFF_DATE"):
            cutoffs[rule_id] = module.CUTOFF_DATE
    return cutoffs


def _write_rules_toml_with_extra(tmp_dir: Path, extra: str) -> Path:
    """The bundled rule specs plus `extra` appended, for exercising entries no module backs."""
    custom_rules_path = tmp_dir / "custom_rules.toml"
    custom_rules_path.write_text(mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8") + extra, encoding="utf-8")
    return custom_rules_path


class CheckModelingStructureTest(unittest.TestCase):
    # --- Utility tests ---

    def test_package_root_reexports_supported_api(self):
        self.assertIs(public_api.analyze_file, mlinter.analyze_file)
        self.assertIs(public_api.format_rule_details, mlinter.format_rule_details)
        self.assertIs(public_api.render_rules_reference, mlinter.render_rules_reference)
        self.assertIs(public_api.Violation, _helpers_mod.Violation)
        self.assertEqual(public_api.__version__, mlinter.__version__)
        self.assertIs(public_api.collect_class_bases, _helpers_mod._collect_class_bases)
        self.assertIs(public_api.has_rule_suppression, _helpers_mod._has_rule_suppression)
        self.assertIs(public_api.inherits_pretrained_model, _helpers_mod._inherits_pretrained_model)
        self.assertIs(public_api.model_dir_name, _helpers_mod._model_dir_name)
        self.assertIs(public_api.is_rule_allowlisted_for_file, mlinter._is_rule_allowlisted_for_file)
        self.assertEqual(public_api.TRF001, "TRF001")
        self.assertEqual(public_api.TRF015, "TRF015")
        self.assertEqual(public_api.TRF016, "TRF016")
        self.assertEqual(public_api.TRF017, "TRF017")
        self.assertEqual(public_api.TRF018, "TRF018")
        self.assertEqual(public_api.TRF019, "TRF019")
        self.assertEqual(public_api.TRF020, "TRF020")
        self.assertEqual(public_api.TRF021, "TRF021")
        self.assertEqual(public_api.TRF022, "TRF022")
        self.assertEqual(public_api.TRF023, "TRF023")
        self.assertEqual(public_api.TRF024, "TRF024")
        self.assertEqual(public_api.TRF025, "TRF025")
        self.assertEqual(public_api.TRF026, "TRF026")
        self.assertEqual(public_api.TRF027, "TRF027")
        self.assertEqual(public_api.TRF028, "TRF028")
        self.assertEqual(public_api.TRF029, "TRF029")
        self.assertEqual(public_api.TRF030, "TRF030")
        self.assertEqual(public_api.TRF031, "TRF031")
        self.assertEqual(public_api.TRF032, "TRF032")
        self.assertEqual(public_api.TRF033, "TRF033")
        self.assertEqual(public_api.TRF034, "TRF034")
        self.assertEqual(public_api.TRF035, "TRF035")
        self.assertEqual(public_api.TRF036, "TRF036")
        self.assertEqual(public_api.TRF037, "TRF037")
        self.assertEqual(public_api.TRF038, "TRF038")
        self.assertEqual(public_api.TRF039, "TRF039")
        self.assertEqual(public_api.TRF040, "TRF040")
        self.assertEqual(public_api.TRF041, "TRF041")
        self.assertEqual(public_api.TRF042, "TRF042")
        self.assertEqual(public_api.TRF043, "TRF043")
        self.assertEqual(public_api.TRF044, "TRF044")
        self.assertEqual(public_api.TRF045, "TRF045")
        self.assertEqual(public_api.TRF046, "TRF046")
        self.assertEqual(public_api.TRF047, "TRF047")
        self.assertEqual(public_api.TRF048, "TRF048")
        self.assertEqual(public_api.TRF049, "TRF049")
        self.assertEqual(public_api.TRF050, "TRF050")
        self.assertEqual(public_api.TRF051, "TRF051")
        self.assertEqual(public_api.TRF052, "TRF052")
        self.assertEqual(public_api.TRF053, "TRF053")
        self.assertEqual(public_api.TRF055, "TRF055")
        self.assertEqual(public_api.TRF056, "TRF056")
        self.assertEqual(public_api.TRF057, "TRF057")
        self.assertEqual(public_api.TRF058, "TRF058")
        self.assertEqual(public_api.TRF059, "TRF059")

    def test_package_root_all_lists_supported_api(self):
        self.assertIn("__version__", public_api.__all__)
        self.assertIn("analyze_file", public_api.__all__)
        self.assertIn("collect_class_bases", public_api.__all__)
        self.assertIn("model_dir_name", public_api.__all__)
        self.assertIn("render_rules_reference", public_api.__all__)
        self.assertIn("TRF001", public_api.__all__)
        self.assertIn("TRF015", public_api.__all__)
        self.assertIn("TRF016", public_api.__all__)
        self.assertIn("TRF017", public_api.__all__)
        self.assertIn("TRF018", public_api.__all__)
        self.assertIn("TRF019", public_api.__all__)
        self.assertIn("TRF020", public_api.__all__)
        self.assertIn("TRF021", public_api.__all__)
        self.assertIn("TRF022", public_api.__all__)
        self.assertIn("TRF023", public_api.__all__)
        self.assertIn("TRF024", public_api.__all__)
        self.assertIn("TRF025", public_api.__all__)
        self.assertIn("TRF026", public_api.__all__)
        self.assertIn("TRF027", public_api.__all__)
        self.assertIn("TRF028", public_api.__all__)
        self.assertIn("TRF029", public_api.__all__)
        self.assertIn("TRF030", public_api.__all__)
        self.assertIn("TRF031", public_api.__all__)
        self.assertIn("TRF032", public_api.__all__)
        self.assertIn("TRF033", public_api.__all__)
        self.assertIn("TRF034", public_api.__all__)
        self.assertIn("TRF035", public_api.__all__)
        self.assertIn("TRF036", public_api.__all__)
        self.assertIn("TRF037", public_api.__all__)
        self.assertIn("TRF038", public_api.__all__)
        self.assertIn("TRF039", public_api.__all__)
        self.assertIn("TRF040", public_api.__all__)
        self.assertIn("TRF041", public_api.__all__)
        self.assertIn("TRF042", public_api.__all__)
        self.assertIn("TRF043", public_api.__all__)
        self.assertIn("TRF044", public_api.__all__)
        self.assertIn("TRF045", public_api.__all__)
        self.assertIn("TRF046", public_api.__all__)
        self.assertIn("TRF047", public_api.__all__)
        self.assertIn("TRF048", public_api.__all__)
        self.assertIn("TRF049", public_api.__all__)
        self.assertIn("TRF050", public_api.__all__)
        self.assertIn("TRF051", public_api.__all__)
        self.assertIn("TRF052", public_api.__all__)
        self.assertIn("TRF053", public_api.__all__)
        self.assertIn("TRF055", public_api.__all__)
        self.assertIn("TRF056", public_api.__all__)
        self.assertIn("TRF057", public_api.__all__)
        self.assertIn("TRF058", public_api.__all__)
        self.assertIn("TRF059", public_api.__all__)
        self.assertNotIn("_collect_class_bases", public_api.__all__)
        self.assertNotIn("_rule_id", public_api.__all__)

    def test_mlinter_module_does_not_leak_rule_loop_variable(self):
        self.assertFalse(hasattr(mlinter, "_rule_id"))

    def test_version_helper_reads_git_hash_from_direct_url(self):
        dist = SimpleNamespace(
            read_text=lambda name: json.dumps(
                {
                    "url": "https://github.com/huggingface/transformers-mlinter",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "abcdef1234567890",
                    },
                }
            )
        )

        self.assertEqual(_version_mod._read_git_hash_from_direct_url(dist), "abcdef1")

    def test_version_helper_reads_git_hash_from_checkout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            (project_root / ".git").write_text("gitdir: /tmp/fake\n", encoding="utf-8")

            with (
                patch.object(_version_mod, "PROJECT_ROOT", project_root),
                patch.object(
                    _version_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        args=["git", "rev-parse", "--short", "HEAD"],
                        returncode=0,
                        stdout="deadbee\n",
                        stderr="",
                    ),
                ),
            ):
                self.assertEqual(_version_mod._read_git_hash_from_checkout(), "deadbee")

    def test_version_helper_resolve_version_prefers_direct_url_hash(self):
        dist = SimpleNamespace(
            version="9.9.9",
            read_text=lambda name: json.dumps(
                {
                    "url": "https://github.com/huggingface/transformers-mlinter",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "abcdef1234567890",
                    },
                }
            ),
        )

        with (
            patch.object(_version_mod, "_installed_distribution", return_value=dist),
            patch.object(_version_mod, "_read_git_hash_from_checkout", return_value="deadbee"),
        ):
            self.assertEqual(_version_mod._resolve_version(), "9.9.9+gabcdef1")

    def test_version_helper_resolve_version_falls_back_without_metadata_or_pyproject(self):
        with (
            patch.object(_version_mod, "_installed_distribution", return_value=None),
            patch.object(_version_mod, "_read_version_from_pyproject", return_value=None),
            patch.object(_version_mod, "_read_git_hash_from_checkout", return_value=None),
        ):
            self.assertEqual(_version_mod._resolve_version(), _version_mod.DEFAULT_BASE_VERSION)

    def test_parse_args_version_prints_version_and_exits(self):
        stdout = StringIO()
        with patch.object(mlinter.sys, "argv", ["mlinter", "--version"]), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as exc:
                mlinter.parse_args()

        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), f"mlinter {mlinter.__version__}\n")

    def test_parse_args_accepts_custom_rules_toml(self):
        custom_rules_path = Path("/tmp/custom_rules.toml")
        with patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path)]):
            args = mlinter.parse_args()

        self.assertEqual(args.rules_toml, custom_rules_path)

    def test_render_rules_reference_matches_rule_specs(self):
        rendered = public_api.render_rules_reference()
        self.assertEqual(rendered.count("### TRF"), len(public_api.TRF_RULE_SPECS))
        self.assertTrue(rendered.endswith("\n"))

    def test_main_uses_custom_rules_toml_for_rule_listing_and_restores_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(
                Path(tmp_dir),
                trf001_description="Custom config_class guidance.",
                trf001_default_enabled=False,
            )
            stdout = StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stdout(stdout),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("TRF001: Custom config_class guidance. (default: disabled)", rendered)
        self.assertIn(
            "config_class on <Model>PreTrainedModel should match <Model>Config naming.",
            mlinter.format_rule_summary("TRF001"),
        )

    def test_content_hash_changes_with_custom_rule_specs(self):
        source = "class FooPreTrainedModel(PreTrainedModel):\n    pass\n"
        default_digest = mlinter._content_hash(source, {mlinter.TRF001})

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(
                Path(tmp_dir),
                trf001_description="Custom config_class guidance.",
                trf001_default_enabled=False,
            )
            with mlinter._using_rule_specs(custom_rules_path):
                custom_digest = mlinter._content_hash(source, {mlinter.TRF001})

        self.assertNotEqual(default_digest, custom_digest)
        self.assertEqual(mlinter.ACTIVE_RULE_SPECS_PATH, mlinter.DEFAULT_RULE_SPECS_PATH)

    def test_rules_toml_omitting_cutoff_date_clears_the_bundled_one(self):
        """`--rules-toml` could add a cutoff but never remove one: the bundled value stayed in force."""
        bundled = _module_cutoffs()
        rules_with_a_cutoff = {rule_id for rule_id, cutoff in bundled.items() if cutoff}
        self.assertTrue(rules_with_a_cutoff, "expected the bundled specs to configure some cutoff dates")
        for rule_id in rules_with_a_cutoff:
            self.assertEqual(bundled[rule_id], mlinter.TRF_RULE_SPECS[rule_id]["cutoff_date"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_without_cutoffs(Path(tmp_dir))
            with mlinter._using_rule_specs(custom_rules_path):
                self.assertEqual(set(_module_cutoffs().values()), {""})

        # Leaving the custom specs behind puts the bundled dates back: rule modules are process-wide.
        self.assertEqual(_module_cutoffs(), bundled)

    def test_rules_toml_omitting_cutoff_date_really_checks_a_grandfathered_model(self):
        # TRF041 carries a cutoff, and a config-gated branch is the shortest thing that trips it.
        source = "def f(self):\n    if config.two_stage:\n        self.stage = Stage()\n"
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=date(2024, 1, 1)):
            # Grandfathered under the bundled specs...
            self.assertEqual(mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041}), [])
            with tempfile.TemporaryDirectory() as tmp_dir:
                custom_rules_path = _write_rules_toml_without_cutoffs(Path(tmp_dir))
                with mlinter._using_rule_specs(custom_rules_path):
                    # ... and checked once the active spec file drops the cutoff.
                    violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041})
                    self.assertEqual(len(violations), 1)
            self.assertEqual(mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041}), [])

    def test_main_rejects_custom_rules_toml_with_unsupported_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(Path(tmp_dir))
            custom_rules_path.write_text(
                custom_rules_path.read_text(encoding="utf-8").replace("version = 1", "version = 2", 1),
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("expected version 1", stderr.getvalue())

    def test_rules_toml_ignored_attributes_extend_a_rule_exempt_list(self):
        """A project keeping its own rules.toml can widen TRF041's exempt fields without a release."""
        source = "def f(self):\n    if config.two_stage:\n        self.stage = Stage()\n"
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        self.assertEqual(len(mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041})), 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = Path(tmp_dir) / "custom_rules.toml"
            custom_rules_path.write_text(
                mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8").replace(
                    "[rules.TRF041]", '[rules.TRF041]\nignored_attributes = ["config.two_stage"]', 1
                ),
                encoding="utf-8",
            )
            with mlinter._using_rule_specs(custom_rules_path):
                self.assertEqual(
                    mlinter.TRF_RULE_SPECS["TRF041"]["ignored_attributes"], frozenset({"config.two_stage"})
                )
                self.assertEqual(mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041}), [])

        # Leaving the custom specs behind restores the bundled exempt list.
        self.assertEqual(mlinter.TRF_RULE_SPECS["TRF041"]["ignored_attributes"], frozenset())
        self.assertEqual(len(mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF041})), 1)

    def test_rules_toml_rejects_non_list_ignored_attributes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = Path(tmp_dir) / "custom_rules.toml"
            custom_rules_path.write_text(
                mlinter.DEFAULT_RULE_SPECS_PATH.read_text(encoding="utf-8").replace(
                    "[rules.TRF041]", '[rules.TRF041]\nignored_attributes = "config.two_stage"', 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ignored_attributes must be list\\[str\\]"):
                with mlinter._using_rule_specs(custom_rules_path):
                    pass

    # --- Deprecated rules ---

    def test_deprecated_rule_is_ignored_by_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(
                Path(tmp_dir),
                '\n[rules.TRF999]\ndeprecated = true\ndescription = "Retired, kept as a tombstone."\n',
            )
            with mlinter._using_rule_specs(custom_rules_path):
                self.assertIn("TRF999", mlinter.DEPRECATED_TRF_RULES)
                self.assertNotIn("TRF999", mlinter.TRF_RULES)
                self.assertNotIn("TRF999", mlinter.TRF_RULE_SPECS)
                self.assertNotIn("TRF999", mlinter.TRF_RULE_CHECKS)
                self.assertNotIn("TRF999", mlinter.DEFAULT_ENABLED_TRF_RULES)
                self.assertFalse(hasattr(mlinter, "TRF999"))
                # The tombstone keeps its description so the docs site can still publish a page for it.
                self.assertEqual(
                    mlinter.DEPRECATED_TRF_RULE_SPECS["TRF999"], {"description": "Retired, kept as a tombstone."}
                )

        self.assertNotIn("TRF999", mlinter.DEPRECATED_TRF_RULES)
        self.assertNotIn("TRF999", mlinter.DEPRECATED_TRF_RULE_SPECS)

    def test_deprecated_rule_without_a_description_still_gets_a_tombstone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(Path(tmp_dir), "\n[rules.TRF999]\ndeprecated = true\n")
            with mlinter._using_rule_specs(custom_rules_path):
                self.assertEqual(
                    mlinter.DEPRECATED_TRF_RULE_SPECS["TRF999"],
                    {"description": "TRF999 was removed from mlinter."},
                )

    def test_deprecated_rule_rejects_an_empty_description(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(
                Path(tmp_dir), '\n[rules.TRF999]\ndeprecated = true\ndescription = "   "\n'
            )
            with self.assertRaises(ValueError) as ctx:
                with mlinter._using_rule_specs(custom_rules_path):
                    pass

        self.assertIn("description must be a non-empty string", str(ctx.exception))

    def test_bundled_deprecated_rules_are_fully_retired(self):
        # TRF054 was the first removal; the list only grows, so it doubles as the non-empty guard.
        self.assertIn("TRF054", mlinter.BUNDLED_DEPRECATED_TRF_RULES)
        for rule_id in mlinter.BUNDLED_DEPRECATED_TRF_RULES:
            self.assertIn(rule_id, public_api.DEPRECATED_TRF_RULES)
            self.assertNotIn(rule_id, public_api.TRF_RULES)
            self.assertNotIn(rule_id, public_api.TRF_RULE_CHECKS)
            self.assertNotIn(rule_id, public_api.__all__)
            self.assertFalse(hasattr(public_api, rule_id))
            module_path = mlinter.DEFAULT_RULE_SPECS_PATH.with_name(f"{rule_id.lower()}.py")
            self.assertFalse(module_path.exists(), f"{module_path.name} should have been deleted")
            # Retired but still documented: the docs site builds a page per entry here, so a reader who
            # meets the id in an old CI log finds out it is gone rather than hitting a 404.
            description = public_api.DEPRECATED_TRF_RULE_SPECS[rule_id]["description"]
            self.assertIsInstance(description, str)
            self.assertTrue(description.strip())

    def test_deprecated_rule_specs_is_public_and_disjoint_from_live_rules(self):
        self.assertIn("DEPRECATED_TRF_RULE_SPECS", public_api.__all__)
        self.assertEqual(
            set(public_api.DEPRECATED_TRF_RULE_SPECS) & set(public_api.TRF_RULE_SPECS),
            set(),
        )
        self.assertEqual(set(public_api.DEPRECATED_TRF_RULE_SPECS), set(public_api.DEPRECATED_TRF_RULES))

    def test_main_rejects_enabling_a_deprecated_rule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(
                Path(tmp_dir),
                '\n[rules.TRF999]\ndeprecated = true\ndescription = "Retired, kept as a tombstone."\n',
            )
            stderr = StringIO()
            with (
                patch.object(
                    mlinter.sys,
                    "argv",
                    ["mlinter", "--rules-toml", str(custom_rules_path), "--enable-rules", "TRF999"],
                ),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Deprecated rule id(s): TRF999", _unwrapped(stderr.getvalue()))

    def test_main_rejects_docs_request_for_a_deprecated_rule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(
                Path(tmp_dir),
                '\n[rules.TRF999]\ndeprecated = true\ndescription = "Retired, kept as a tombstone."\n',
            )
            stderr = StringIO()
            with (
                patch.object(
                    mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--rule", "TRF999"]
                ),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Deprecated rule id(s): TRF999", _unwrapped(stderr.getvalue()))

    def test_main_rejects_deprecated_rule_marked_default_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_rules_toml_with_extra(
                Path(tmp_dir), "\n[rules.TRF999]\ndeprecated = true\ndefault_enabled = true\n"
            )
            stderr = StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("deprecated rules cannot be enabled", _unwrapped(stderr.getvalue()))

    def test_main_rejects_rules_toml_that_still_activates_a_deprecated_rule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(Path(tmp_dir))
            stderr = StringIO()
            with (
                patch.object(mlinter, "BUNDLED_DEPRECATED_TRF_RULES", frozenset({"TRF001"})),
                patch.object(mlinter.sys, "argv", ["mlinter", "--rules-toml", str(custom_rules_path), "--list-rules"]),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Deprecated rule(s) still active", _unwrapped(stderr.getvalue()))
        self.assertIn("TRF001", _unwrapped(stderr.getvalue()))

    def test_deprecated_rule_with_a_surviving_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_rules_path = _write_custom_rules_toml(Path(tmp_dir), trf001_default_enabled=False)
            custom_rules_path.write_text(
                custom_rules_path.read_text(encoding="utf-8").replace(
                    "[rules.TRF001]\n", "[rules.TRF001]\ndeprecated = true\n", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as exc:
                with mlinter._using_rule_specs(custom_rules_path):
                    pass

        self.assertIn("trf001.py still exists", str(exc.exception))
        self.assertEqual(mlinter.ACTIVE_RULE_SPECS_PATH, mlinter.DEFAULT_RULE_SPECS_PATH)

    def test_analyze_file_allows_subscripted_class_bases(self):
        source = (
            LICENSE_HEADER
            + """
from collections import OrderedDict

class _LazyConfigMapping(OrderedDict[str, str]):
    pass
"""
        )
        file_path = Path("src/transformers/models/auto/configuration_auto.py")
        violations = mlinter.analyze_file(file_path, source)
        self.assertEqual(violations, [])

    def test_cache_path_uses_xdg_cache_home_on_linux(self):
        with (
            patch.object(mlinter.sys, "platform", "linux"),
            patch.dict(mlinter.os.environ, {"XDG_CACHE_HOME": "/tmp/mlinter-xdg-cache"}, clear=True),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/tmp/mlinter-xdg-cache") / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_cache_path_uses_library_caches_on_macos(self):
        with (
            patch.object(mlinter.sys, "platform", "darwin"),
            patch.object(mlinter.Path, "home", return_value=Path("/Users/tester")),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/Users/tester") / "Library" / "Caches" / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_cache_path_uses_localappdata_on_windows(self):
        with (
            patch.object(mlinter.sys, "platform", "win32"),
            patch.dict(mlinter.os.environ, {"LOCALAPPDATA": "/tmp/localappdata"}, clear=True),
        ):
            self.assertEqual(
                mlinter._cache_path(),
                Path("/tmp/localappdata") / "mlinter" / mlinter.CACHE_FILENAME,
            )

    def test_save_cache_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "nested" / "mlinter" / mlinter.CACHE_FILENAME

            with patch("mlinter.mlinter._cache_path", return_value=cache_path):
                mlinter._save_cache({"foo.py": "digest"})

            self.assertTrue(cache_path.exists())
            self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8")), {"foo.py": "digest"})

    @patch("mlinter.mlinter.subprocess.run")
    def test_get_changed_modeling_files_includes_configuration_files(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout=(
                    "src/transformers/models/foo/modeling_foo.py\n"
                    "src/transformers/models/foo/modular_foo.py\n"
                    "src/transformers/models/foo/configuration_foo.py\n"
                    "docs/source/en/index.md\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff", "--cached"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "ls-files"], returncode=0, stdout="", stderr=""),
        ]
        changed_files = mlinter.get_changed_modeling_files("origin/main")
        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/foo/modeling_foo.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
                Path("src/transformers/models/foo/configuration_foo.py"),
            },
        )

    @patch("mlinter.mlinter.subprocess.run")
    def test_get_changed_modeling_files_includes_uncommitted_worktree_changes(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout="src/transformers/models/helium/modeling_helium.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "diff", "--cached"],
                returncode=0,
                stdout="src/transformers/models/foo/modular_foo.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "ls-files"],
                returncode=0,
                stdout=("src/transformers/models/bar/modeling_bar.py\ndocs/source/en/index.md\n"),
                stderr="",
            ),
        ]

        changed_files = mlinter.get_changed_modeling_files("origin/main")

        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/helium/modeling_helium.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
                Path("src/transformers/models/bar/modeling_bar.py"),
            },
        )

    # --- Generated-file skipping ---

    def test_iter_modeling_files_skips_generated_files(self):
        banner = "# This file was automatically generated from src/transformers/models/foo/modular_foo.py.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            generated = model_dir / "modeling_foo.py"
            generated.write_text(banner + "class FooModel: ...\n", encoding="utf-8")
            handwritten = model_dir / "modeling_bar.py"
            handwritten.write_text("class BarModel: ...\n", encoding="utf-8")
            modular = model_dir / "modular_foo.py"
            modular.write_text("class FooModel: ...\n", encoding="utf-8")

            self.assertTrue(mlinter._is_generated_file(generated))
            self.assertFalse(mlinter._is_generated_file(handwritten))
            self.assertFalse(mlinter._is_generated_file(modular))

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                found = set(mlinter.iter_modeling_files())
            self.assertNotIn(generated, found)
            self.assertIn(handwritten, found)
            self.assertIn(modular, found)

    # --- generated-file filtering ---

    def test_is_generated_file_detects_banner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modeling_foo.py"
            path.write_text(
                "#                This file was automatically generated from foo/modular_foo.py.\n"
                "class FooModel:\n    pass\n",
                encoding="utf-8",
            )
            self.assertTrue(mlinter._is_generated_file(path))

    def test_is_generated_file_false_for_handwritten_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modular_foo.py"
            path.write_text("class FooModel:\n    pass\n", encoding="utf-8")
            self.assertFalse(mlinter._is_generated_file(path))

    def test_is_generated_file_false_for_missing_file(self):
        missing = Path("/nonexistent/modeling_foo.py")
        self.assertFalse(mlinter._is_generated_file(missing))

    def test_is_generated_file_only_reads_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modeling_foo.py"
            # Banner buried past the 1KB head is not treated as a generation marker.
            path.write_text(
                "x = 0\n" * 400 + f"# {mlinter.GENERATED_FILE_MARKER} foo/modular_foo.py\n",
                encoding="utf-8",
            )
            self.assertFalse(mlinter._is_generated_file(path))

    def test_iter_modeling_files_skips_generated_in_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            generated = model_dir / "modeling_foo.py"
            generated.write_text(
                f"# {mlinter.GENERATED_FILE_MARKER} foo/modular_foo.py\nclass FooModel:\n    pass\n",
                encoding="utf-8",
            )
            source = model_dir / "modular_foo.py"
            source.write_text("class FooModel:\n    pass\n", encoding="utf-8")

            yielded = list(mlinter.iter_modeling_files({generated, source}))

            self.assertEqual(yielded, [source])

    def test_iter_modeling_files_skips_generated_when_walking_models_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            generated = model_dir / "modeling_foo.py"
            generated.write_text(
                f"# {mlinter.GENERATED_FILE_MARKER} foo/modular_foo.py\nclass FooModel:\n    pass\n",
                encoding="utf-8",
            )
            source = model_dir / "modular_foo.py"
            source.write_text("class FooModel:\n    pass\n", encoding="utf-8")

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                yielded = set(mlinter.iter_modeling_files())

            self.assertEqual(yielded, {source})

    def test_iter_modeling_files_returns_processing_files(self):
        expected = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir)
            model_dir = models_root / "foo"
            model_dir.mkdir()
            filenames = [
                "modeling_foo.py",
                "processing_foo.py",
                "image_processing_foo.py",
                "video_processing_foo.py",
                "tokenization_foo.py",
                "generation_foo.py",
            ]
            for name in filenames:
                path = model_dir / name
                path.write_text("import torch", encoding="utf-8")
                expected.add(path)

            with patch.object(mlinter, "MODELS_ROOT", models_root):
                yielded = set(mlinter.iter_modeling_files())

            self.assertEqual(yielded, expected)

    # --- Linting a standalone model repository (search paths) ---

    @staticmethod
    def _write_standalone_model_repo(root: Path) -> Path:
        """A model repo laid out the way a Hub repository is: model files at the top level."""
        repo = root / "llada"
        repo.mkdir(parents=True)
        (repo / "configuration_llada.py").write_text("class LladaConfig:\n    pass\n", encoding="utf-8")
        (repo / "modeling_llada.py").write_text(
            "class LladaPreTrainedModel(PreTrainedModel):\n"
            "    pass\n"
            "\n"
            "class LladaModel(LladaPreTrainedModel):\n"
            "    def __init__(self, config):\n"
            "        super().__init__(config)\n",
            encoding="utf-8",
        )
        (repo / "sampling_helpers.py").write_text("def sample():\n    pass\n", encoding="utf-8")
        return repo

    def test_search_paths_discover_model_files_outside_the_transformers_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = self._write_standalone_model_repo(Path(tmp_dir))
            found = {path.name for path in mlinter.iter_modeling_files(search_paths=[repo])}
        self.assertEqual(found, {"configuration_llada.py", "modeling_llada.py"})

    def test_search_paths_skip_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            generated = repo / "modeling_foo.py"
            generated.write_text(
                f"# {_helpers_mod.GENERATED_FILE_MARKER} modular_foo.py\nclass FooModel: ...\n", encoding="utf-8"
            )
            (repo / "modular_foo.py").write_text("class FooModel: ...\n", encoding="utf-8")
            found = {path.name for path in mlinter.iter_modeling_files(search_paths=[repo])}
        self.assertEqual(found, {"modular_foo.py"})

    def test_search_paths_accept_an_explicit_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = self._write_standalone_model_repo(Path(tmp_dir))
            target = repo / "modeling_llada.py"
            found = list(mlinter.iter_modeling_files(search_paths=[target]))
        self.assertEqual(found, [target])

    def test_resolve_search_paths_rejects_a_missing_path(self):
        self.assertIsNone(mlinter.resolve_search_paths([]))
        with self.assertRaises(ValueError) as exc:
            mlinter.resolve_search_paths([Path("/nonexistent/model/repo")])
        self.assertIn("/nonexistent/model/repo", str(exc.exception))

    def test_changed_only_candidate_honours_search_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = self._write_standalone_model_repo(Path(tmp_dir))
            self.assertTrue(mlinter._is_modeling_candidate(repo / "modeling_llada.py", [repo]))
            self.assertFalse(mlinter._is_modeling_candidate(repo / "sampling_helpers.py", [repo]))
            self.assertFalse(
                mlinter._is_modeling_candidate(Path("src/transformers/models/foo/modeling_foo.py"), [repo])
            )

    def test_main_lints_a_standalone_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = self._write_standalone_model_repo(Path(tmp_dir))
            stdout, stderr = StringIO(), StringIO()
            with (
                patch.object(
                    mlinter.sys,
                    "argv",
                    ["mlinter", str(repo), "--no-cache", "--no-progress", "--enable-rules", "TRF013"],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 1)
        # The missing `post_init` call is exactly the kind of bug a Hub model repo silently ships.
        self.assertIn("does not call `self.post_init`", stderr.getvalue().replace("\n", ""))

    def test_main_warns_when_a_search_path_holds_no_model_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout, stderr = StringIO(), StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", tmp_dir, "--no-cache", "--no-progress"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("no model integration file found", stderr.getvalue().replace("\n", ""))

    def test_no_empty_warning_under_changed_only(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            mlinter.warn_about_search_paths([Path("/models/llada")], [], warn_when_empty=False)
        self.assertEqual(stderr.getvalue(), "")

    def test_main_warns_about_an_explicit_file_no_rule_applies_to(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            unrelated = Path(tmp_dir) / "model.py"
            unrelated.write_text("class LladaModel:\n    pass\n", encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with (
                patch.object(mlinter.sys, "argv", ["mlinter", str(unrelated), "--no-cache", "--no-progress"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = mlinter.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("is not a model integration file", stderr.getvalue().replace("\n", ""))

    def test_known_model_dirs_is_empty_outside_a_transformers_checkout(self):
        with patch.object(_helpers_mod, "MODELS_ROOT", Path("/nonexistent/src/transformers/models")):
            self.assertEqual(_helpers_mod._known_model_dirs(), set())
