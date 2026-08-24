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
    if self.config.num_labels > 0:
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

    def test_trf041_file_scoped_ignore_directive(self):
        body = """
class M:
    def __init__(self, config):
        self.scale = 1.0 if config.scale_embedding else 2.0
        if self.config.problem_type == "regression":
            pass
        self.act = A if config.hidden_act == "relu" else B
"""

        def flagged(source):
            messages = [v.message for v in self._run(mlinter.TRF041, source)]
            return sorted(m.split("`")[1].removeprefix("self.").removeprefix("config.") for m in messages)

        every_flag = ["hidden_act", "problem_type", "scale_embedding"]
        self.assertEqual(flagged(body), every_flag)

        # A module-level directive naming attributes exempts exactly those, comma- or space-separated.
        directive = "# trf-ignore: TRF041 config.problem_type, config.hidden_act\n"
        self.assertEqual(flagged(directive + body), ["scale_embedding"])

        # The bare field name is accepted too, since `self.config.x`, `config.x` and `x` are one field.
        self.assertEqual(flagged("# trf-ignore: TRF041 problem_type\n" + body), ["hidden_act", "scale_embedding"])

        # Prose after the attribute list is not parsed as further attribute names.
        trailing = "# trf-ignore: TRF041 config.problem_type - generic loss plumbing\n"
        self.assertEqual(flagged(trailing + body), ["hidden_act", "scale_embedding"])

        # A directive naming no attribute stays per-line only, so it never mutes the whole file.
        self.assertEqual(flagged("# trf-ignore: TRF041\n" + body), every_flag)

        # Another rule's directive, and an indented one, are both out of scope.
        self.assertEqual(flagged("# trf-ignore: TRF012 config.problem_type\n" + body), every_flag)
        self.assertEqual(flagged(body + "        # trf-ignore: TRF041 config.hidden_act\n"), every_flag)

    def test_trf041_file_scoped_ignore_does_not_carry_compound_conditions(self):
        """An exempt field in a compound condition must not exempt the live field beside it."""
        body = """
class M:
    def __init__(self, config):
        if config.problem_type and config.use_embedding_norm:
            pass
        self.act = A if config.hidden_act or config.use_cache else B
"""

        def flagged(source):
            messages = [v.message for v in self._run(mlinter.TRF041, source)]
            return sorted(m.split("`")[1].removeprefix("self.").removeprefix("config.") for m in messages)

        directive = "# trf-ignore: TRF041 problem_type, hidden_act\n"
        # The branch still reads a field nobody exempted, so it still has to name its checkpoints.
        self.assertEqual(flagged(directive + body), ["use_cache", "use_embedding_norm"])

        # Exempting every field in the condition is what silences it.
        every_field = "# trf-ignore: TRF041 problem_type, hidden_act, use_embedding_norm, use_cache\n"
        self.assertEqual(flagged(every_field + body), [])

        # A repeated field is reported once, not once per read.
        repeated = """
class M:
    def __init__(self, config):
        if config.use_cache and config.use_cache is not None:
            pass
"""
        self.assertEqual(flagged(repeated), ["use_cache"])
