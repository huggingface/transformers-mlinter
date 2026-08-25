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


class TRF029Test(RuleTestCase):
    # --- TRF029: config plus a redundant config field ---

    def test_trf029_flags_redundant_config_arguments(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
"""
        violations = self._run(mlinter.TRF029, source)
        self.assertEqual(len(violations), 1)
        for name in ("embed_dim", "num_heads", "dropout"):
            self.assertIn(name, violations[0].message)

    def test_trf029_accepts_config_only_and_layer_idx(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, layer_idx=None, device=None, **kwargs):
        super().__init__()
        self.embed_dim = config.hidden_size
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])

    def test_trf029_accepts_optional_config_field_overrides(self):
        # The MoE pattern: one MLP class serves the dense width and the expert width, and the config
        # stays the source of truth for callers that pass nothing.
        source = """
class FooMLP(LlamaMLP):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(config)
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])

        # Keyword-only, and after a `*`, is the same thing.
        kwonly = """
class FooMLP(nn.Module):
    def __init__(self, config, *, intermediate_size=None):
        super().__init__()
"""
        self.assertEqual(self._run(mlinter.TRF029, kwonly), [])

    def test_trf029_still_flags_required_and_hardcoded_config_fields(self):
        # A hardcoded default is a decision the signature made; only `None` means "read the config".
        hardcoded = """
class FooMLP(nn.Module):
    def __init__(self, config, intermediate_size=4096):
        super().__init__()
"""
        violations = self._run(mlinter.TRF029, hardcoded)
        self.assertEqual(len(violations), 1)
        self.assertIn("intermediate_size", violations[0].message)

        # An override beside a required field leaves the required one reportable, and only that one.
        mixed = """
class FooAttention(nn.Module):
    def __init__(self, config, num_heads, head_dim=None):
        super().__init__()
"""
        violations = self._run(mlinter.TRF029, mixed)
        self.assertEqual(len(violations), 1)
        self.assertIn("num_heads", violations[0].message)
        self.assertNotIn("head_dim", violations[0].message)

        # `config=None` still counts as taking a config, so the class stays in scope.
        optional_config = """
class FooAttention(nn.Module):
    def __init__(self, config=None, num_heads=8):
        super().__init__()
"""
        self.assertEqual(len(self._run(mlinter.TRF029, optional_config)), 1)

    def test_trf029_ignores_modules_without_config(self):
        source = """
class FooRotary(nn.Module):
    def __init__(self, head_dim, rope_theta):
        super().__init__()
        self.head_dim = head_dim
"""
        self.assertEqual(self._run(mlinter.TRF029, source), [])
