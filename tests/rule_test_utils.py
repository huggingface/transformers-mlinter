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
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from mlinter import _helpers as _helpers_mod
from mlinter import mlinter
from mlinter import trf011 as _trf011_mod
from mlinter import trf019 as _trf019_mod
from mlinter import trf020 as _trf020_mod
from mlinter import trf022 as _trf022_mod
from mlinter import trf023 as _trf023_mod
from mlinter import trf038 as _trf038_mod
from mlinter import trf042 as _trf042_mod
from mlinter import trf057 as _trf057_mod
from mlinter import trf059 as _trf059_mod


__all__ = [
    "LICENSE_HEADER",
    "Path",
    "RuleTestCase",
    "TEST_PP_PLAN_MODULES",
    "_helpers_mod",
    "_trf011_mod",
    "_trf019_mod",
    "_trf020_mod",
    "_trf022_mod",
    "_trf023_mod",
    "_trf038_mod",
    "_trf042_mod",
    "_trf057_mod",
    "_trf059_mod",
    "date",
    "mlinter",
    "patch",
    "tempfile",
    "unittest",
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
