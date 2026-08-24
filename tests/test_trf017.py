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


class TRF017Test(RuleTestCase):
    # --- TRF017: @auto_docstring must be placed above @dataclass ---

    def test_trf017_flags_dataclass_above_auto_docstring(self):
        source = """
@dataclass
@auto_docstring
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(len(trf017), 1)
        self.assertIn("FooOutput", trf017[0].message)
        self.assertIn("@dataclass listed above @auto_docstring", trf017[0].message)

    def test_trf017_flags_dataclass_above_called_auto_docstring(self):
        source = '''
@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`FooForPreTraining`].
    """
)
class FooForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor = None
'''
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(len(trf017), 1)
        self.assertIn("FooForPreTrainingOutput", trf017[0].message)

    def test_trf017_allows_auto_docstring_above_dataclass(self):
        source = """
@auto_docstring
@dataclass
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(trf017, [])

    def test_trf017_allows_dataclass_only(self):
        source = """
@dataclass
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(trf017, [])

    def test_trf017_allows_auto_docstring_only(self):
        source = """
@auto_docstring
class FooModel(PreTrainedModel):
    pass
"""
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(trf017, [])

    def test_trf017_respects_inline_suppression(self):
        source = """
@dataclass  # trf-ignore: TRF017
@auto_docstring
class FooOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
"""
        trf017 = self._run(mlinter.TRF017, source)
        self.assertEqual(trf017, [])
