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


class TRF069Test(RuleTestCase):
    # --- TRF069: `_get_num_multimodal_tokens` must read its `**kwargs` through `self._merge_kwargs` ---

    def _trf069(self, source: str, file_name: str = "processing_foo.py") -> list:
        return self._run(mlinter.TRF069, source, file_name=file_name)

    def test_flags_raw_kwarg_reads_and_hand_offs(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
        size = kwargs["size"]
        patches = [self.image_processor.get_number_of_image_patches(h, w, kwargs) for h, w in image_sizes]
        return MultiModalData(num_image_tokens=[p // merge_size**2 for p in patches])
"""
        violations = self._trf069(source)
        self.assertEqual([v.line_number for v in violations], [4, 5, 6])
        self.assertIn("`_get_num_multimodal_tokens` uses `kwargs` directly", violations[0].message)

    def test_allows_merged_kwargs(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        images_kwargs = self.image_processor._merge_and_standardize_kwargs(
            **self._merge_kwargs(**kwargs)["images_kwargs"]
        )
        merged = self._merge_kwargs(FooProcessorKwargs, **kwargs)
        return MultiModalData(num_image_tokens=[images_kwargs["merge_size"]] * len(image_sizes))
"""
        self.assertEqual(self._trf069(source), [])

    def test_allows_delegating_to_parent_counter(self):
        source = """
class FooProcessor(BarProcessor):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        return super()._get_num_multimodal_tokens(image_sizes=image_sizes, **kwargs)
"""
        self.assertEqual(self._trf069(source), [])

    def test_allows_counter_without_kwargs_use(self):
        # Fixed-count processors such as PaliGemma accept `**kwargs` but never read them.
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        return MultiModalData(num_image_tokens=[self.image_seq_length] * len(image_sizes))

    def _other_method(self, **kwargs):
        return kwargs.get("size")
"""
        self.assertEqual(self._trf069(source), [])

    def test_flags_unpacking_into_other_calls_and_custom_kwargs_name(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **mm_kwargs):
        return self._count(image_sizes, **mm_kwargs)
"""
        violations = self._trf069(source)
        self.assertEqual([v.line_number for v in violations], [4])
        self.assertIn("uses `mm_kwargs` directly", violations[0].message)

    def test_reports_each_line_once(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        images_kwargs = {**kwargs, **kwargs.get("images_kwargs", {})}
        return MultiModalData(num_image_tokens=[images_kwargs["merge_size"]] * len(image_sizes))
"""
        self.assertEqual([v.line_number for v in self._trf069(source)], [4])

    def test_checks_modular_files_only_alongside_processing_files(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        return kwargs.get("size")
"""
        self.assertEqual(len(self._trf069(source, file_name="modular_foo.py")), 1)
        self.assertEqual(self._trf069(source, file_name="image_processing_foo.py"), [])

    def test_suppression_comment(self):
        source = """
class FooProcessor(ProcessorMixin):
    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        return kwargs.get("size")  # trf-ignore: TRF069
"""
        self.assertEqual(self._trf069(source), [])
