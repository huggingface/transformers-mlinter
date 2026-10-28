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

import tempfile
import textwrap

from tests.rule_test_utils import Path, RuleTestCase, mlinter


class TRF055Test(RuleTestCase):
    """TRF055: PreTrainedModel `config` is a bare annotation matched by `__init__`; `config_class` is never used."""

    # --- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def _src(source: str) -> str:
        return textwrap.dedent(source).lstrip("\n")

    def _trf055(self, source: str, file_name: str = "modeling_foo.py") -> list:
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        violations = mlinter.analyze_file(file_path, self._src(source), enabled_rules={mlinter.TRF055})
        return [v for v in violations if v.rule_id == mlinter.TRF055]

    def _line(self, source: str, needle: str) -> int:
        """1-based line of the first line containing `needle`, so assertions survive fixture edits."""
        for i, line in enumerate(self._src(source).splitlines(), start=1):
            if needle in line:
                return i
        raise AssertionError(f"{needle!r} not found in fixture")

    def _assert_single(self, source: str, needle: str, *message_parts: str, **kwargs) -> None:
        found = self._trf055(source, **kwargs)
        self.assertEqual(len(found), 1, [v.message for v in found])
        self.assertEqual(found[0].line_number, self._line(source, needle))
        for part in message_parts:
            self.assertIn(part, found[0].message)

    # --- correct code ------------------------------------------------------------------------------

    def test_clean_cases(self):
        cases = {
            "base annotated": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig
            """,
            "subclass inherits config, no __init__": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig

                class FooModel(FooPreTrainedModel):
                    pass
            """,
            "subclass overrides config, matching __init__ (Gemma4 pattern)": """
                class Gemma4PreTrainedModel(PreTrainedModel):
                    config: Gemma4Config

                class Gemma4VisionModel(Gemma4PreTrainedModel):
                    config: Gemma4VisionConfig

                    def __init__(self, config: Gemma4VisionConfig):
                        super().__init__(config)
            """,
            "__init__ matches inherited config": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig

                class FooModel(FooPreTrainedModel):
                    def __init__(self, config: FooConfig):
                        super().__init__(config)
            """,
            "__init__ matches config inherited through several levels": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig

                class FooBase(FooPreTrainedModel):
                    pass

                class FooTaskFromBase(FooBase):
                    def __init__(self, config: FooConfig):
                        super().__init__(config)
            """,
            "__init__ with extra args and keyword-only config": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig

                class FooModel(FooPreTrainedModel):
                    def __init__(self, config: FooConfig, extra_kwargs: bool = False):
                        super().__init__(config)
            """,
            "string annotation matches": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: "FooConfig"

                class FooModel(FooPreTrainedModel):
                    def __init__(self, config: "FooConfig"):
                        super().__init__(config)
            """,
            "exotic __init__ annotation is not our business": """
                class FooPreTrainedModel(PreTrainedModel):
                    config: FooConfig

                class FooModel(FooPreTrainedModel):
                    def __init__(self, config: FooConfig | None = None):
                        super().__init__(config)
            """,
            "Not PreTrainedModel class can have config attribute": """
                class FooHelper:
                    config = FooConfig
            """,
            "Not PreTrainedModel class can omit annotation on __init__(config)": """
                class FooHelper:
                    def __init__(self, config):
                        self.config = config
            """,
        }
        for name, source in cases.items():
            with self.subTest(name):
                self.assertEqual([v.message for v in self._trf055(source)], [])

    # --- class body: config_class and config assignment ------------------------------------------

    def test_flags_config_class_assignment(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig
                config: FooConfig
        """
        self._assert_single(source, "config_class = FooConfig", "config_class", "config: <Model>Config")

    def test_flags_config_class_annotation(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class: FooConfig
                config: FooConfig
        """
        self._assert_single(source, "config_class: FooConfig", "config_class")

    def test_flags_config_class_annotation_with_value(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class: FooConfig = FooConfig
                config: FooConfig
        """
        self._assert_single(source, "config_class: FooConfig = FooConfig", "config_class")

    def test_flags_config_assignment(self):
        source = """
            class Gemma4PreTrainedModel(PreTrainedModel):
                config: Gemma4Config

            class Gemma4VisionModel(Gemma4PreTrainedModel):
                config = Gemma4VisionConfig
        """
        self._assert_single(source, "config = Gemma4VisionConfig", "Gemma4VisionModel", "config: Gemma4VisionConfig")

    def test_flags_config_assignment_with_dotted_name(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                config = configuration_foo.FooConfig
        """
        self._assert_single(source, "config = configuration_foo.FooConfig", "config: FooConfig")

    def test_flags_config_assignment_with_non_name_value(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                config = get_config()
        """
        self._assert_single(source, "config = get_config()", "config: <Model>Config")

    def test_flags_annotated_config_with_value(self):
        # `config: X = X` still creates a class attribute, so it is an assignment in disguise.
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig = FooConfig
        """
        self._assert_single(source, "config: FooConfig = FooConfig", "config: FooConfig")

    def test_reports_each_defect_on_its_own_line(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig
                config = FooConfig
        """
        found = self._trf055(source)
        print(found)
        self.assertEqual(
            sorted(v.line_number for v in found),
            [2, 3],
        )

    # --- the base <Model>PreTrainedModel must declare config -------------------------------------

    def test_flags_base_without_config_annotation(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                base_model_prefix = "model"
        """
        self._assert_single(source, "class FooPreTrainedModel", "FooPreTrainedModel", "config: <Model>Config")

    def test_flags_derived_pretrained_base_without_annotation(self):
        # A second *PreTrainedModel (e.g. a vision-specific base) must declare its own config.
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooVisionPreTrainedModel(FooPreTrainedModel):
                pass
        """
        self._assert_single(source, "class FooVisionPreTrainedModel", "FooVisionPreTrainedModel")

    def test_does_not_require_annotation_on_plain_subclass(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                def forward(self, x):
                    return x
        """
        self.assertEqual(self._trf055(source), [])

    # --- __init__ must annotate and agree with the effective config ------------------------------

    def test_flags_unannotated_init_config(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                def __init__(self, config):
                    super().__init__(config)
        """
        self._assert_single(source, "def __init__(self, config)", "must annotate", "config: FooConfig")

    def test_flags_init_mismatching_inherited_config(self):
        source = """
            class Gemma4PreTrainedModel(PreTrainedModel):
                config: Gemma4Config

            class Gemma4VisionModel(Gemma4PreTrainedModel):
                def __init__(self, config: Gemma4VisionConfig):
                    super().__init__(config)
        """
        self._assert_single(
            source,
            "config: Gemma4VisionConfig",
            "inherits",
            "config: Gemma4Config",
            "config: Gemma4VisionConfig",
        )

    def test_flags_init_mismatching_own_override(self):
        source = """
            class Gemma4PreTrainedModel(PreTrainedModel):
                config: Gemma4Config

            class Gemma4VisionModel(Gemma4PreTrainedModel):
                config: Gemma4VisionConfig

                def __init__(self, config: Gemma4TextConfig):
                    super().__init__(config)
        """
        self._assert_single(source, "config: Gemma4TextConfig", "overrides", "config: Gemma4VisionConfig")

    def test_flags_init_mismatching_config_inherited_through_chain(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooMid(FooPreTrainedModel):
                pass

            class FooLeaf(FooMid):
                def __init__(self, config: BarConfig):
                    super().__init__(config)
        """
        self._assert_single(source, "config: BarConfig", "inherits", "FooConfig", "BarConfig")

    def test_flags_init_config_with_nothing_declared_or_inherited(self):
        source = """
            class FooModel(PreTrainedModel):
                def __init__(self, config: FooConfig):
                    super().__init__(config)
        """
        self._assert_single(source, "config: FooConfig", "Declare `config: FooConfig`")

    # --- scope -----------------------------------------------------------------------------------

    def test_skips_files_other_than_modeling(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig

            class FooModel(FooPreTrainedModel):
                config = FooConfig

                def __init__(self, config):
                    super().__init__(config)
        """
        for file_name in ("modular_foo.py", "configuration_foo.py", "processing_foo.py", "foo_utils.py"):
            with self.subTest(file_name):
                self.assertEqual(self._trf055(source, file_name=file_name), [])

    def test_checks_modeling_files_with_extended_names(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig
        """
        self.assertTrue(self._trf055(source, file_name="modeling_foo_bar.py"))

    def _trf055_modular(self, modular: str, generated: str) -> list:
        """Lint `modular_foo.py` next to the `modeling_foo.py` generated from it."""
        banner = "#           This file was automatically generated from src/transformers/models/foo/modular_foo.py.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "modeling_foo.py").write_text(banner + self._src(generated), encoding="utf-8")
            modular_path = model_dir / "modular_foo.py"
            modular_path.write_text(self._src(modular), encoding="utf-8")
            violations = mlinter.analyze_file(modular_path, self._src(modular), enabled_rules={mlinter.TRF055})
        return [v for v in violations if v.rule_id == mlinter.TRF055]

    def test_modular_reports_generated_problem_on_its_class(self):
        modular = """
            from ..bar.modeling_bar import BarVisionEncoder

            class FooVisionEncoder(BarVisionEncoder):
                pass
        """
        generated = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooVisionEncoder(FooPreTrainedModel):
                def __init__(self, config: FooVisionConfig):
                    super().__init__(config)
        """
        found = self._trf055_modular(modular, generated)
        self.assertEqual(len(found), 1, [v.message for v in found])
        self.assertEqual(found[0].line_number, self._line(modular, "class FooVisionEncoder"))
        self.assertIn("generated modeling_foo.py:", found[0].message)
        self.assertIn("takes `config: FooVisionConfig` but the class inherits `config: FooConfig`", found[0].message)

    def test_modular_skips_classes_it_does_not_define(self):
        # `FooPreTrainedModel` was copied from another model by the converter: that model's files report it.
        modular = """
            class FooModel(BarModel):
                pass
        """
        generated = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig

            class FooModel(FooPreTrainedModel):
                config: FooConfig
        """
        self.assertEqual(self._trf055_modular(modular, generated), [])

    def test_modular_suppression_on_class_line(self):
        modular = """
            class FooPreTrainedModel(BarPreTrainedModel):  # trf-ignore: TRF055
                pass
        """
        generated = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig
        """
        self.assertEqual(self._trf055_modular(modular, generated), [])

    def test_skips_union_annotation_written_as_string(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooVisionBackbone(FooPreTrainedModel):
                def __init__(self, config: "FooConfig | FooVisionConfig"):
                    super().__init__(config)
        """
        self.assertEqual(self._trf055(source), [])

    # --- suppression -----------------------------------------------------------------------------

    def test_suppression_on_config_assignment(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                # trf-ignore: TRF055
                config = FooConfig
        """
        self.assertEqual(self._trf055(source), [])

    def test_suppression_on_config_class(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config_class = FooConfig  # trf-ignore: TRF055
                config: FooConfig
        """
        self.assertEqual(self._trf055(source), [])

    def test_suppression_on_missing_annotation_goes_on_class_line(self):
        source = """
            # trf-ignore: TRF055
            class FooPreTrainedModel(PreTrainedModel):
                pass
        """
        self.assertEqual(self._trf055(source), [])

    def test_suppression_on_init_config_argument(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                def __init__(
                    self,
                    config: BarConfig,  # trf-ignore: TRF055
                ):
                    super().__init__(config)
        """
        self.assertEqual(self._trf055(source), [])

    def test_suppression_on_unannotated_init(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                # trf-ignore: TRF055
                def __init__(self, config):
                    super().__init__(config)
        """
        self.assertEqual(self._trf055(source), [])

    def test_suppression_applies_only_to_its_own_construct(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooModel(FooPreTrainedModel):
                # trf-ignore: TRF055
                config = FooConfig

            class BarModel(FooPreTrainedModel):
                config = BarConfig
        """
        self._assert_single(source, "config = BarConfig", "BarModel")

    # --- a class built from a sub-config must declare it -----------------------------------------

    COMPOSITE_CONFIG = """
        class FooTextConfig(PreTrainedConfig):
            pass

        class FooVisionConfig(PreTrainedConfig):
            pass

        class FooConfig(PreTrainedConfig):
            sub_configs_defaults = {"text_config": FooTextConfig, "vision_config": FooVisionConfig}
    """

    def _trf055_in_model_dir(self, modeling: str, configuration: str = COMPOSITE_CONFIG, modular: str | None = None):
        """Lint `modeling_foo.py` (or `modular_foo.py`, generating `modeling_foo.py`) next to `configuration_foo.py`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "src" / "transformers" / "models" / "foo"
            model_dir.mkdir(parents=True)
            (model_dir / "configuration_foo.py").write_text(self._src(configuration), encoding="utf-8")
            banner = "# This file was automatically generated from src/transformers/models/foo/modular_foo.py.\n"
            (model_dir / "modeling_foo.py").write_text(
                (banner if modular is not None else "") + self._src(modeling), encoding="utf-8"
            )
            target = model_dir / ("modular_foo.py" if modular is not None else "modeling_foo.py")
            if modular is not None:
                target.write_text(self._src(modular), encoding="utf-8")
            violations = mlinter.analyze_file(
                target, target.read_text(encoding="utf-8"), enabled_rules={mlinter.TRF055}
            )
        return [v for v in violations if v.rule_id == mlinter.TRF055]

    INHERITS_COMPOSITE = """
        class FooPreTrainedModel(PreTrainedModel):
            config: FooConfig

        class FooTextModel(FooPreTrainedModel):
            def __init__(self, config: FooConfig):
                super().__init__(config)

        class FooModel(FooPreTrainedModel):
            def __init__(self, config: FooConfig):
                super().__init__(config)
                self.language_model = {call}
    """

    def test_flags_class_built_from_sub_config_that_declares_the_composite(self):
        for call in (
            "FooTextModel(config.text_config)",
            "FooTextModel(config=config.text_config)",
            "FooTextModel._from_config(config.text_config)",
        ):
            with self.subTest(call):
                source = self.INHERITS_COMPOSITE.format(call=call)
                found = self._trf055_in_model_dir(source)
                self.assertEqual(len(found), 1, [v.message for v in found])
                self.assertEqual(found[0].line_number, self._line(source, "class FooTextModel"))
                self.assertIn("built from `config.text_config` in `FooModel`", found[0].message)
                self.assertIn("declares `config: FooConfig`, the composite model's config", found[0].message)
                self.assertIn("Declare `config: FooTextConfig`", found[0].message)

    def test_flags_class_built_from_sub_config_that_declares_another_sub_config(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooTextModel(FooPreTrainedModel):
                config: FooVisionConfig

                def __init__(self, config: FooVisionConfig):
                    super().__init__(config)

            class FooModel(FooPreTrainedModel):
                def __init__(self, config: FooConfig):
                    super().__init__(config)
                    self.language_model = FooTextModel(config.text_config)
        """
        found = self._trf055_in_model_dir(source)
        self.assertEqual(len(found), 1, [v.message for v in found])
        self.assertIn("declares `config: FooVisionConfig`. Declare `config: FooTextConfig`", found[0].message)

    def test_accepts_class_declaring_its_sub_config(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooTextModel(FooPreTrainedModel):
                config: FooTextConfig

                def __init__(self, config: FooTextConfig):
                    super().__init__(config)

            class FooModel(FooPreTrainedModel):
                def __init__(self, config: FooConfig):
                    super().__init__(config)
                    self.language_model = FooTextModel._from_config(config.text_config)
        """
        self.assertEqual(self._trf055_in_model_dir(source), [])

    def test_skips_sub_config_typed_as_auto_config(self):
        configuration = """
            class FooConfig(PreTrainedConfig):
                sub_configs = {"text_config": AutoConfig}
        """
        source = self.INHERITS_COMPOSITE.format(call="FooTextModel(config.text_config)")
        self.assertEqual(self._trf055_in_model_dir(source, configuration=configuration), [])

    def test_without_sub_config_table_flags_only_the_composite(self):
        configuration = """
            class FooConfig(PreTrainedConfig):
                pass
        """
        source = self.INHERITS_COMPOSITE.format(call="FooTextModel(config.text_config)")
        found = self._trf055_in_model_dir(source, configuration=configuration)
        self.assertEqual(len(found), 1, [v.message for v in found])
        self.assertIn("Declare the config class of `config.text_config`", found[0].message)

    def test_ignores_non_pretrained_classes_built_from_sub_config(self):
        source = """
            class FooPreTrainedModel(PreTrainedModel):
                config: FooConfig

            class FooVisionTower(nn.Module):
                def __init__(self, config):
                    super().__init__()

            class FooModel(FooPreTrainedModel):
                def __init__(self, config: FooConfig):
                    super().__init__(config)
                    self.vision_tower = FooVisionTower(config.vision_config)
        """
        self.assertEqual(self._trf055_in_model_dir(source), [])

    def test_modular_pass_subclass_reported_on_its_class(self):
        # The shape that motivated the check: the modular class adds nothing, so it inherits the composite config
        # from `FooPreTrainedModel` and the converter renames the copied `__init__` annotation to match.
        modular = """
            from ..llama.modeling_llama import LlamaModel

            class FooTextModel(LlamaModel):
                pass
        """
        modeling = self.INHERITS_COMPOSITE.format(call="FooTextModel._from_config(config.text_config)")
        found = self._trf055_in_model_dir(modeling, modular=modular)
        self.assertEqual(len(found), 1, [v.message for v in found])
        self.assertEqual(found[0].line_number, self._line(modular, "class FooTextModel"))
        self.assertIn("generated modeling_foo.py:", found[0].message)

    def test_suppression_on_class_built_from_sub_config(self):
        source = self.INHERITS_COMPOSITE.format(call="FooTextModel(config.text_config)").replace(
            "class FooTextModel(FooPreTrainedModel):", "class FooTextModel(FooPreTrainedModel):  # trf-ignore: TRF055"
        )
        self.assertEqual(self._trf055_in_model_dir(source), [])
