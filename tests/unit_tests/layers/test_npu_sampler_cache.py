import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import torch
from omni.layers import npu_sampler_cache

fake_vllm = types.ModuleType("vllm")
fake_vllm_utils = types.ModuleType("vllm.utils")
fake_vllm_utils.is_pin_memory_available = lambda: False
fake_vllm.utils = fake_vllm_utils
sys.modules.setdefault("vllm", fake_vllm)
sys.modules.setdefault("vllm.utils", fake_vllm_utils)
PenaltyCache = npu_sampler_cache.PenaltyCache
ProbCache = npu_sampler_cache.ProbCache
move_cached_tensors = npu_sampler_cache.move_cached_tensors


class DummyScheduledReq:
    def __init__(self, req_id, prompt_token_ids):
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids


class DummySamplingMetadata:
    def __init__(self, no_penalties=False):
        self.no_penalties = no_penalties


class TestMoveCachedTensors(TestCase):
    def test_swaps_values_using_last_row_as_buffer(self):
        tensor_one = torch.tensor([1, 2, 99])
        tensor_two = torch.tensor([3, 4, 88])

        move_cached_tensors([tensor_one, tensor_two], src=[1, 0])

        self.assertTrue(torch.equal(tensor_one, torch.tensor([2, 1, 1])))
        self.assertTrue(torch.equal(tensor_two, torch.tensor([3, 4, 88])))


class TestPenaltyCache(TestCase):
    def setUp(self):
        self.num_req = 2
        self.vocab_size = 6
        self.device = "cpu"
        self.cache = PenaltyCache(self.num_req, self.vocab_size, device=self.device)

    def test_permute_cached_reqs_reorders_cached_rows(self):
        self.cache.permute_cached_reqs([1, 2])

        original_prompt_masks = [
            torch.tensor([True, False, False, False, False, False]),
            torch.tensor([False, True, False, False, False, False]),
            torch.tensor([False, False, True, False, False, False]),
        ]
        for idx, mask in enumerate(original_prompt_masks):
            self.cache.prompt_mask[idx] = mask
            self.cache.output_mask[idx] = mask.clone()
            self.cache.output_bin_counts[idx] = mask.clone().to(dtype=torch.int64)

        self.cache.permute_cached_reqs([2, 1])

        self.assertEqual(self.cache.cached_req_ids, [2, 1])
        self.assertTrue(torch.equal(self.cache.prompt_mask[0], original_prompt_masks[1]))
        self.assertTrue(torch.equal(self.cache.prompt_mask[1], original_prompt_masks[0]))
        self.assertTrue(torch.equal(self.cache.prompt_mask[2], original_prompt_masks[0]))
        self.assertTrue(torch.equal(self.cache.output_mask[0], original_prompt_masks[0]))
        self.assertTrue(torch.equal(self.cache.output_bin_counts[1], original_prompt_masks[1].to(dtype=torch.int64)))

    def test_prepare_new_reqs_resets_and_sets_masks(self):
        cached_ids = [10, 20]
        self.cache.permute_cached_reqs(cached_ids)

        self.cache.output_bin_counts += 1
        self.cache.output_mask = torch.ones_like(self.cache.output_mask)

        scheduled = [
            DummyScheduledReq(cached_ids[0], [0, 2, 5]),
            DummyScheduledReq(cached_ids[1], [1, 3]),
        ]
        self.cache.prepare_new_reqs(scheduled)

        self.assertTrue(torch.equal(self.cache.output_bin_counts[0], torch.zeros(self.vocab_size, dtype=torch.int64)))
        self.assertTrue(torch.equal(self.cache.output_bin_counts[1], torch.zeros(self.vocab_size, dtype=torch.int64)))
        self.assertFalse(torch.any(self.cache.output_mask[0]))
        self.assertFalse(torch.any(self.cache.output_mask[1]))
        expected_first = torch.tensor([True, False, True, False, False, True])
        expected_second = torch.tensor([False, True, False, True, False, False])
        self.assertTrue(torch.equal(self.cache.prompt_mask[0], expected_first))
        self.assertTrue(torch.equal(self.cache.prompt_mask[1], expected_second))

    def test_prepare_new_reqs_raises_for_missing_req_id(self):
        self.cache.permute_cached_reqs([1, 2])

        with self.assertRaises(RuntimeError):
            self.cache.prepare_new_reqs([DummyScheduledReq(999, [0])])

    def test_do_penalty_flags_from_sampling_metadata(self):
        req_ids = [1, 2]
        frequency_penalties = torch.tensor([0.0, 0.5, 0.0])
        presence_penalties = torch.tensor([0.0, 0.0, 0.0])
        repetition_penalties = torch.tensor([1.1, 1.0, 1.0])
        input_batch = SimpleNamespace(
            req_ids=req_ids,
            frequency_penalties_cpu_tensor=frequency_penalties,
            presence_penalties_cpu_tensor=presence_penalties,
            repetition_penalties_cpu_tensor=repetition_penalties,
        )

        do_flags = self.cache.do_penalty_from_samplinng_metadata(input_batch)

        self.assertEqual(do_flags, (True, False, True))

    def test_update_and_prepare_cache_with_penalties(self):
        req_ids = [3, 4]
        sampling_metadata = DummySamplingMetadata(no_penalties=False)
        input_batch = SimpleNamespace(
            req_ids=req_ids,
            frequency_penalties_cpu_tensor=torch.tensor([0.0, 0.0]),
            presence_penalties_cpu_tensor=torch.tensor([0.0, 1.0]),
            repetition_penalties_cpu_tensor=torch.tensor([1.5, 1.0]),
        )
        scheduled = [DummyScheduledReq(req_id, [1, 2]) for req_id in req_ids]

        self.cache.prepare_cache(scheduled, req_ids, sampling_metadata, input_batch)

        self.assertTrue(self.cache.do_penalties)
        self.assertEqual(self.cache.cached_req_ids, req_ids)
        self.assertTrue(self.cache.do_presence_penalties)
        self.assertTrue(self.cache.do_repetition_penalties)
        self.assertFalse(self.cache.do_frequency_penalties)
        self.assertFalse(torch.any(self.cache.output_mask[0]))
        self.assertFalse(torch.any(self.cache.output_mask[1]))

    def test_prepare_cache_skips_when_no_penalties(self):
        req_ids = [5, 6]
        sampling_metadata = DummySamplingMetadata(no_penalties=True)
        input_batch = SimpleNamespace(
            req_ids=req_ids,
            frequency_penalties_cpu_tensor=torch.tensor([0.0, 0.0]),
            presence_penalties_cpu_tensor=torch.tensor([0.0, 0.0]),
            repetition_penalties_cpu_tensor=torch.tensor([1.0, 1.0]),
        )

        self.cache.prepare_cache([], req_ids, sampling_metadata, input_batch)

        self.assertFalse(self.cache.do_penalties)
        self.assertIsNone(self.cache.cached_req_ids)

    def test_save_token_ids_and_revert_rejected_tokens(self):
        req_ids = [7, 8]
        sampling_metadata = DummySamplingMetadata(no_penalties=False)
        input_batch = SimpleNamespace(
            req_ids=req_ids,
            frequency_penalties_cpu_tensor=torch.tensor([0.0, 0.0]),
            presence_penalties_cpu_tensor=torch.tensor([0.0, 0.0]),
            repetition_penalties_cpu_tensor=torch.tensor([1.0, 1.0]),
        )
        scheduled = [DummyScheduledReq(req_id, [0]) for req_id in req_ids]
        self.cache.prepare_cache(scheduled, req_ids, sampling_metadata, input_batch)

        sampled_tokens = torch.tensor([1, 3])
        self.cache.save_token_ids(sampled_tokens)

        flat_counts = self.cache.output_bin_counts.view(-1)
        self.assertEqual(flat_counts[self.cache.start_indices[0] + 1].item(), 1)
        self.assertEqual(flat_counts[self.cache.start_indices[1] + 3].item(), 1)
        self.assertTrue(self.cache.output_mask.view(-1)[self.cache.start_indices[0] + 1])
        self.assertTrue(self.cache.output_mask.view(-1)[self.cache.start_indices[1] + 3])

        accepted_mask = torch.tensor([True, False])
        self.cache.revert_rejected_tokens(accepted_mask, sampled_tokens)

        flat_counts = self.cache.output_bin_counts.view(-1)
        self.assertEqual(flat_counts[self.cache.start_indices[0] + 1].item(), 1)
        self.assertEqual(flat_counts[self.cache.start_indices[1] + 3].item(), 0)
        self.assertTrue(self.cache.output_mask.view(-1)[self.cache.start_indices[0] + 1])
        self.assertFalse(self.cache.output_mask.view(-1)[self.cache.start_indices[1] + 3])


class TestProbCache(TestCase):
    def setUp(self):
        self.num_req = 2
        self.num_tokens = 3
        self.vocab_size = 5
        self.cache = ProbCache(self.num_req, self.num_tokens, self.vocab_size, device="cpu")

    def test_permute_cached_reqs_moves_probabilities(self):
        self.cache.permute_cached_reqs([1, 2])
        self.cache.topk_spec_token_probs[0] = torch.full((self.num_tokens, self.vocab_size), 1.0)
        self.cache.topk_spec_token_probs[1] = torch.full((self.num_tokens, self.vocab_size), 2.0)
        self.cache.computed[0] = True

        self.cache.permute_cached_reqs([2, 1])

        self.assertEqual(self.cache.cached_req_ids, [2, 1])
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[0], torch.full((self.num_tokens, self.vocab_size), 2.0)))
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[1], torch.full((self.num_tokens, self.vocab_size), 1.0)))
        self.assertTrue(self.cache.computed[0].item())

    def test_prepare_new_reqs_resets_state(self):
        cached_ids = [3, 4]
        self.cache.permute_cached_reqs(cached_ids)
        self.cache.topk_spec_token_probs += 1.0
        self.cache.computed = torch.ones_like(self.cache.computed)

        scheduled = [DummyScheduledReq(req_id, []) for req_id in cached_ids]
        self.cache.prepare_new_reqs(scheduled)

        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[0], torch.zeros((self.num_tokens, self.vocab_size))))
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[1], torch.zeros((self.num_tokens, self.vocab_size))))
        self.assertFalse(self.cache.computed[0].item())
        self.assertFalse(self.cache.computed[1].item())

    def test_update_cached_probs_marks_computed(self):
        probs = torch.full((2, self.vocab_size), 0.25)
        self.cache.update_cached_probs(idx=1, topk_spec_token_probs=probs)

        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[0, 1], probs[0]))
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[1, 1], probs[1]))
        self.assertTrue(self.cache.computed[0].item())
        self.assertTrue(self.cache.computed[1].item())
        self.assertFalse(self.cache.computed[2].item())

    def test_prepare_cache_calls_helpers(self):
        req_ids = [7, 8]
        scheduled = [DummyScheduledReq(req_id, []) for req_id in req_ids]
        input_batch = SimpleNamespace(req_ids=req_ids)

        self.cache.prepare_cache(scheduled, req_ids, DummySamplingMetadata(), input_batch)

        self.assertEqual(self.cache.cached_req_ids, req_ids)
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[0], torch.zeros((self.num_tokens, self.vocab_size))))
        self.assertTrue(torch.equal(self.cache.topk_spec_token_probs[1], torch.zeros((self.num_tokens, self.vocab_size))))
        self.assertFalse(self.cache.computed[0].item())
        self.assertFalse(self.cache.computed[1].item())