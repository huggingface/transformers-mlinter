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


from tests.rule_test_utils import RuleTestCase, mlinter


class TRF041Test(RuleTestCase):
    # --- TRF041: config-gated branches need a # CODEPATH: note ---

    def test_trf041_flags_undocumented_config_branch(self):
        source = """
class FooLayer(nn.Module):
    def forward(self, hidden_states):
        if self.config.use_embedding_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states
"""
        violations = self._run(mlinter.TRF041, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("CODEPATH", violations[0].message)
        self.assertIn("use_embedding_norm", violations[0].message)

    def test_trf041_accepts_marker_above_and_inline(self):
        above = """
class FooLayer(nn.Module):
    def forward(self, hidden_states):
        # CODEPATH: only ESMC-6B ships pre-normalised embeddings.
        if self.config.use_embedding_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states
"""
        self.assertEqual(self._run(mlinter.TRF041, above), [])
        inline = """
class FooLayer(nn.Module):
    def forward(self, hidden_states):
        if self.config.use_embedding_norm:  # CODEPATH: 6B only
            hidden_states = self.norm(hidden_states)
        return hidden_states
"""
        self.assertEqual(self._run(mlinter.TRF041, inline), [])

    def test_trf041_accepts_marker_heading_a_comment_block(self):
        source = """
class FooLayer(nn.Module):
    def forward(self, hidden_states):
        # CODEPATH: the 6B checkpoint pre-normalises its embeddings.
        # The 300M and 600M checkpoints do not, so this stays optional.
        if self.config.use_embedding_norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states
"""
        self.assertEqual(self._run(mlinter.TRF041, source), [])

    def test_trf041_covers_elif_and_conditional_expressions(self):
        elif_source = """
def f(self, x):
    if self.config.a:
        return x
    elif self.config.b:
        return -x
    return 0
"""
        # Both the `if` and the `elif` need their own note.
        self.assertEqual(len(self._run(mlinter.TRF041, elif_source)), 2)
        ternary = """
class FooLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
"""
        violations = self._run(mlinter.TRF041, ternary)
        self.assertEqual(len(violations), 1)
        self.assertIn("conditional expression", violations[0].message)

    def test_trf041_flags_non_boolean_config_conditions(self):
        source = """
def f(self, x):
    if self.config.num_experts > 0:
        return x
    if self.config.backbone is not None:
        return -x
    return 0
"""
        self.assertEqual(len(self._run(mlinter.TRF041, source)), 2)

    def test_trf041_ignores_branches_not_touching_config(self):
        source = """
def f(self, x, use_cache=False):
    if use_cache:
        return x
    if x is None:
        return 0
    return -x
"""
        self.assertEqual(self._run(mlinter.TRF041, source), [])

    def test_trf041_respects_suppression_and_file_type(self):
        suppressed = """
def f(self, x):
    # trf-ignore: TRF041
    if self.config.a:
        return x
"""
        self.assertEqual(self._run(mlinter.TRF041, suppressed), [])
        config_file = "def f(self, x):\n    if self.config.a:\n        return x\n"
        self.assertEqual(self._run(mlinter.TRF041, config_file, file_name="configuration_foo.py"), [])

    def test_trf041_exempts_default_coalesce_but_not_real_none_forks(self):
        # `X if X is not None else fallback` yields the field when set and a default when not, so no
        # checkpoint diverges and there is no path to name. Accepted from either side, with or without
        # the `self.` prefix.
        for source in (
            "d = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout\n",
            "d = config.hidden_dropout if config.classifier_dropout is None else config.classifier_dropout\n",
            "d = self.config.cd if self.config.cd is not None else self.config.hd\n",
        ):
            self.assertEqual(self._run(mlinter.TRF041, source), [])

        # Merely mentioning None is not enough: these fork the graph and still owe a note.
        forks = (
            "m = VisionTower(config) if config.vision_config is not None else None\n",  # a whole extra tower
            "d = config.a if config.b is not None else config.c\n",  # tested field is not a result
            "d = config.a if config.a > 0 else config.b\n",  # not a None test at all
            "if config.vision_config is not None:\n    self.tower = VisionTower(config)\n",  # statement form
        )
        for source in forks:
            self.assertEqual(len(self._run(mlinter.TRF041, source)), 1, source)

    def test_trf041_exempts_framework_plumbing_fields_by_default(self):
        # None of these fork the graph: they select a loss, look up an activation, or answer a generic
        # PretrainedConfig question, in the same shape in every model that has the head or the feature.
        for field in (
            "problem_type",
            "hidden_act",
            "num_labels",
            "is_decoder",
            "is_encoder_decoder",
            "pad_token_id",
            "tie_word_embeddings",
            "_attn_implementation",
            "use_cache",
            "summary_use_proj",  # the whole summary_* family is exempt by prefix
        ):
            for source in (
                f"def f(self, x):\n    if self.config.{field}:\n        return x\n",
                f"def f(self, x):\n    return x if config.{field} == 'a' else -x\n",
            ):
                self.assertEqual(self._run(mlinter.TRF041, source), [], source)

        # A field whose name merely starts like an exempt one is not exempt.
        self.assertEqual(
            len(self._run(mlinter.TRF041, "def f():\n    if config.use_cache_router:\n        pass\n")), 1
        )

    def test_trf041_exempts_guard_branches(self):
        guards = (
            # a raise: one side aborts, so nothing diverges past the branch
            "def f(self):\n    if self.config.num_experts is None:\n        raise ValueError('need experts')\n",
            # a raise the guard builds a message for first
            (
                "def f(self):\n    if config.scale_embedding and config.two_stage:\n"
                "        msg = 'incompatible'\n        raise ValueError(msg)\n"
            ),
            # warnings and logging, including logger.warning_once
            "def f(self):\n    if config.two_stage:\n        logger.warning_once('deprecated')\n",
            "def f(self):\n    if config.two_stage:\n        warnings.warn('deprecated')\n",
            # an elif with no tail is a guard of its own
            "def f(self):\n    if x:\n        pass\n    elif config.two_stage:\n        raise ValueError('no')\n",
        )
        for source in guards:
            self.assertEqual(self._run(mlinter.TRF041, source), [], source)

        forks = (
            # an else means the branch really does pick between two paths
            (
                "def f(self):\n    if config.two_stage:\n        raise ValueError('no')\n"
                "    else:\n        self.stage = Stage()\n"
            ),
            # a body that does work beside the warning is a fork, warning or not
            "def f(self):\n    if config.two_stage:\n        logger.warning('slow')\n        self.stage = Stage()\n",
            # a conditional expression has no body to guard with
            "def f(self):\n    self.stage = Stage() if config.two_stage else None\n",
        )
        for source in forks:
            self.assertEqual(len(self._run(mlinter.TRF041, source)), 1, source)

    def test_trf041_file_scoped_ignore_directive(self):
        body = """
class M:
    def __init__(self, config):
        self.scale = 1.0 if config.scale_embedding else 2.0
        if self.config.auxiliary_loss:
            pass
        self.adapter = A if config.add_adapter else B
"""

        def flagged(source):
            messages = [v.message for v in self._run(mlinter.TRF041, source)]
            return sorted(m.split("`")[1].removeprefix("self.").removeprefix("config.") for m in messages)

        every_flag = ["add_adapter", "auxiliary_loss", "scale_embedding"]
        self.assertEqual(flagged(body), every_flag)

        # A module-level directive naming attributes exempts exactly those, comma- or space-separated.
        directive = "# trf-ignore: TRF041 config.auxiliary_loss, config.add_adapter\n"
        self.assertEqual(flagged(directive + body), ["scale_embedding"])

        # The bare field name is accepted too, since `self.config.x`, `config.x` and `x` are one field.
        self.assertEqual(flagged("# trf-ignore: TRF041 auxiliary_loss\n" + body), ["add_adapter", "scale_embedding"])

        # Prose after the attribute list is not parsed as further attribute names.
        trailing = "# trf-ignore: TRF041 config.auxiliary_loss - never set by a released checkpoint\n"
        self.assertEqual(flagged(trailing + body), ["add_adapter", "scale_embedding"])

        # A directive naming no attribute stays per-line only, so it never mutes the whole file.
        self.assertEqual(flagged("# trf-ignore: TRF041\n" + body), every_flag)

        # Another rule's directive, and an indented one, are both out of scope.
        self.assertEqual(flagged("# trf-ignore: TRF012 config.auxiliary_loss\n" + body), every_flag)
        self.assertEqual(flagged(body + "        # trf-ignore: TRF041 config.add_adapter\n"), every_flag)

    def test_trf041_file_scoped_ignore_does_not_carry_compound_conditions(self):
        """An exempt field in a compound condition must not exempt the live field beside it."""
        body = """
class M:
    def __init__(self, config):
        if config.auxiliary_loss and config.use_embedding_norm:
            pass
        self.act = A if config.add_adapter or config.two_stage else B
"""

        def flagged(source):
            messages = [v.message for v in self._run(mlinter.TRF041, source)]
            return sorted(m.split("`")[1].removeprefix("self.").removeprefix("config.") for m in messages)

        directive = "# trf-ignore: TRF041 auxiliary_loss, add_adapter\n"
        # The branch still reads a field nobody exempted, so it still has to name its checkpoints.
        self.assertEqual(flagged(directive + body), ["two_stage", "use_embedding_norm"])

        # Exempting every field in the condition is what silences it.
        every_field = "# trf-ignore: TRF041 auxiliary_loss, add_adapter, use_embedding_norm, two_stage\n"
        self.assertEqual(flagged(every_field + body), [])

        # A field exempt by default carries no weight either: `use_cache` beside a live field leaves the
        # live one reportable, and on its own it silences the branch.
        mixed = """
class M:
    def __init__(self, config):
        if config.use_cache and config.two_stage:
            pass
        if config.use_cache:
            pass
"""
        self.assertEqual(flagged(mixed), ["two_stage"])

        # A repeated field is reported once, not once per read.
        repeated = """
class M:
    def __init__(self, config):
        if config.two_stage and config.two_stage is not None:
            pass
"""
        self.assertEqual(flagged(repeated), ["two_stage"])
