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

import tempfile  # noqa: F401 - re-exported for existing rule tests
import unittest
from datetime import date  # noqa: F401 - re-exported for existing rule tests
from pathlib import Path
from unittest.mock import patch  # noqa: F401 - re-exported for existing rule tests

from mlinter import _helpers as _helpers_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import mlinter
from mlinter import trf011 as _trf011_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf019 as _trf019_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf020 as _trf020_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf022 as _trf022_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf023 as _trf023_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf038 as _trf038_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf042 as _trf042_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf057 as _trf057_mod  # noqa: F401 - re-exported for existing rule tests
from mlinter import trf059 as _trf059_mod  # noqa: F401 - re-exported for existing rule tests


__all__ = [
    "LICENSE_HEADER",
    "RuleTestCase",
    "TEST_PP_PLAN_MODULES",
]


# The header every model file is expected to carry, verbatim from the library.
LICENSE_HEADER = """# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""


TEST_PP_PLAN_MODULES = {"foo": {"embed_tokens", "final_layer_norm", "layers", "norm"}}


class RuleTestCase(unittest.TestCase):
    def _run(self, rule, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={rule})
        return [v for v in violations if v.rule_id == rule]
