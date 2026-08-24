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


from tests.rule_test_utils import LICENSE_HEADER, RuleTestCase, mlinter


class TRF028Test(RuleTestCase):
    # --- TRF028: Apache license header ---

    def test_trf028_flags_missing_header(self):
        violations = self._run(mlinter.TRF028, '"""PyTorch Foo model."""\n\nimport torch\n')
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].line_number, 1)
        self.assertIn("missing the license header", violations[0].message)

    def test_trf028_accepts_header(self):
        self.assertEqual(self._run(mlinter.TRF028, LICENSE_HEADER + "\nimport torch\n"), [])

    def test_trf028_accepts_header_below_the_generated_file_banner(self):
        banner = "#  🚨 This file was automatically generated from modular_foo.py.\n#  Do NOT edit it manually.\n"
        self.assertEqual(self._run(mlinter.TRF028, banner + LICENSE_HEADER + "\nimport torch\n"), [])

    def test_trf028_flags_truncated_header(self):
        # bitnet ships this: everything but the closing `limitations under the License.`
        truncated = LICENSE_HEADER.rsplit("\n", 2)[0] + "\n"
        violations = self._run(mlinter.TRF028, truncated + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("limitations under the license.", violations[0].message)

    def test_trf028_flags_header_mangled_by_search_and_replace(self):
        # tvp and bridgetower ship this: a stray `=` inserted before every comma.
        mangled = LICENSE_HEADER.replace(",", "=,")
        violations = self._run(mlinter.TRF028, mangled + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("incomplete license header", violations[0].message)

    def test_trf028_accepts_a_non_apache_license(self):
        # Not every model is Apache 2.0: blip is BSD-3-clause and sapiens2 carries Meta's own
        # license. Both spell out the same warranty paragraph, which is what the rule checks.
        bsd3 = LICENSE_HEADER.replace(
            'Apache License, Version 2.0 (the "License")', 'BSD-3-clause license (the "License")'
        ).replace("http://www.apache.org/licenses/LICENSE-2.0", "https://opensource.org/licenses/BSD-3-Clause")
        sapiens = """# Copyright 2026 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Sapiens2 License. You may obtain a copy of the License at
#
#     https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
        for name, header in (("bsd-3-clause", bsd3), ("sapiens2", sapiens)):
            with self.subTest(license=name):
                self.assertEqual(self._run(mlinter.TRF028, header + "\nimport torch\n"), [])

    def test_trf028_flags_a_header_that_never_names_a_license(self):
        # The warranty paragraph alone, with no "Licensed under the ..." line above it.
        headless = "\n".join(LICENSE_HEADER.splitlines()[4:]) + "\n"
        violations = self._run(mlinter.TRF028, headless + "\nimport torch\n")
        self.assertEqual(len(violations), 1)
        self.assertIn("does not state what license the file is under", violations[0].message)

    def test_trf028_accepts_any_copyright_attribution(self):
        # The year and the attributed team legitimately vary; only the boilerplate is checked.
        for line in (
            "# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.",
            "# Copyright 2019 The Google AI Language Team Authors.",
        ):
            source = line + "\n" + LICENSE_HEADER.split("\n", 1)[1] + "\nimport torch\n"
            with self.subTest(copyright=line):
                self.assertEqual(self._run(mlinter.TRF028, source), [])

    def test_trf028_respects_suppression(self):
        source = "# trf-ignore: TRF028\n\nimport torch\n"
        self.assertEqual(self._run(mlinter.TRF028, source), [])

    def test_trf028_ignores_unrelated_files(self):
        self.assertEqual(self._run(mlinter.TRF028, "import torch\n", file_name="tokenization_foo.py"), [])
