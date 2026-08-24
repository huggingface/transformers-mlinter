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


from tests.rule_test_utils import Path, RuleTestCase, mlinter, tempfile


class TRF015Test(RuleTestCase):
    # --- TRF015: _tied_weights_keys requires tie_word_embeddings in config ---

    def test_trf015_valid_config_has_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_missing_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)

    def test_trf015_empty_tied_weights_keys_no_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_inherited_config_no_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(LlamaConfig):
    model_type = "foo"
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_main_composite_requires_top_level_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True

class FooConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForConditionalGeneration(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)

    def test_trf015_config_file_suffix_matching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo_audio.py").write_text(
                """
class FooAudioConfig(PreTrainedConfig):
    sample_rate: int = 16000
""",
                encoding="utf-8",
            )
            (model_dir / "configuration_foo_text.py").write_text(
                """
class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooTextPreTrainedModel(PreTrainedModel):
    pass

class FooTextForCausalLM(FooTextPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo_text.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_only_checks_target_config_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooVisionConfig(FooConfig):
    model_type = "foo_vision"

class FooConfig(PreTrainedConfig):
    model_type = "foo"
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForConditionalGeneration(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("tie_word_embeddings", trf015[0].message)
            self.assertIn("FooConfig", trf015[0].message)
            self.assertNotIn("FooVisionConfig", trf015[0].message)

    def test_trf015_resolves_inherited_config_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class FooConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}
    hidden_size: int = 768

class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooTextConfig

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(trf015, [])

    def test_trf015_resolves_inherited_config_annotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "configuration_foo.py").write_text(
                """
class CompositeConfig(PreTrainedConfig):
    sub_configs = {"text_config": FooTextConfig, "vision_config": AutoConfig}

class FooTextConfig(PreTrainedConfig):
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )

            modeling_source = """
class WrapperPreTrainedModel(PreTrainedModel):
    config: CompositeConfig

class FooMainModel(WrapperPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
"""
            file_path = model_dir / "modeling_foo.py"
            violations = mlinter.analyze_file(file_path, modeling_source, enabled_rules={mlinter.TRF015})
            trf015 = [v for v in violations if v.rule_id == mlinter.TRF015]
            self.assertEqual(len(trf015), 1)
            self.assertIn("CompositeConfig", trf015[0].message)

    def test_trf015_cache_invalidated_by_config_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            modeling_source = """
class FooPreTrainedModel(PreTrainedModel):
    pass

class FooForCausalLM(FooPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
"""
            modeling_path = model_dir / "modeling_foo.py"
            modeling_path.write_text(modeling_source, encoding="utf-8")

            config_path = model_dir / "configuration_foo.py"
            config_path.write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
""",
                encoding="utf-8",
            )
            digest_v1 = mlinter._content_hash(
                modeling_source,
                {mlinter.TRF015},
                mlinter._find_companion_files(modeling_path),
            )

            config_path.write_text(
                """
class FooConfig(PreTrainedConfig):
    hidden_size: int = 768
    tie_word_embeddings: bool = True
""",
                encoding="utf-8",
            )
            digest_v2 = mlinter._content_hash(
                modeling_source,
                {mlinter.TRF015},
                mlinter._find_companion_files(modeling_path),
            )

            self.assertNotEqual(digest_v1, digest_v2)
