import pytest
import torch
import torch_npu
from types import SimpleNamespace

import omni.layers.sampler as sampler_mod
from omni.layers.sampler import (
    _apply_penalties,
    _get_logprobs_adapter,
    _multinomial,
    _apply_top_k_top_p_faster,
    _modify_greedy_probs_inplace,
    _check_top_ks,
    _get_greedy_token,
    _need_log_probs,
    _apply_penalties_v1,
    gather_tokens,
    AscendSampler,
    AscendSamplerV1,
    AscendTopKTopPSamplerV1,
    RejectionSampler,
    SimpleSampler,
    SparseRejectionSampler,
    apply_top_k_only,
    apply_top_k_top_p,
    random_choice,
    random_sample,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadataV1
from vllm.sequence import Logprob
from .distributed_test_common import parse_ascend_devices

FIRST_DIE = parse_ascend_devices()[0]
@pytest.fixture(scope="module")
def npu_device():
    device = torch.device(f"npu:{FIRST_DIE}")
    torch.npu.set_device(device)
    return device


def _scatter_back(sorted_logits: torch.Tensor, sorted_idx: torch.Tensor, template: torch.Tensor):
    full = torch.full_like(template, float("-inf"))
    full.scatter_(1, sorted_idx, sorted_logits)
    return full


def _ensure_bin_counts():
    if hasattr(sampler_mod, "_get_bin_counts_and_mask"):
        return sampler_mod._get_bin_counts_and_mask

    def _get_bin_counts_and_mask(tokens: torch.Tensor, vocab_size: int, num_seqs: int):
        tokens = tokens.view(num_seqs, -1)
        valid = tokens >= 0
        rows = torch.arange(num_seqs, device=tokens.device).unsqueeze(1).expand_as(tokens)
        valid_rows = rows[valid]
        valid_tokens = tokens[valid]
        counts = torch.zeros(num_seqs, vocab_size, device=tokens.device, dtype=torch.float32)
        if valid_tokens.numel():
            counts.index_put_((valid_rows, valid_tokens), torch.ones_like(valid_tokens, dtype=torch.float32), accumulate=True)
        mask = counts > 0
        return counts, mask

    sampler_mod._get_bin_counts_and_mask = _get_bin_counts_and_mask
    return sampler_mod._get_bin_counts_and_mask


def test_modify_greedy_probs_inplace_sets_one_hot(npu_device):
    logprobs = torch.randn(2, 4, device=npu_device, dtype=torch.float32)
    probs = torch.ones_like(logprobs)
    sample_indices = torch.tensor([0, 1], device=npu_device, dtype=torch.long)
    greedy_samples = torch.tensor([3, 0], device=npu_device, dtype=torch.long)

    _modify_greedy_probs_inplace(logprobs, probs, sample_indices, greedy_samples)

    expected = torch.zeros_like(probs)
    expected[0, 3] = 1.0
    expected[1, 0] = 1.0
    assert torch.allclose(probs, expected)


def test_apply_top_k_only_masks_logits(npu_device):
    logits = torch.tensor(
        [
            [1.0, 0.5, 2.5, -0.5],
            [0.0, -0.2, 0.3, 4.0],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    k = torch.tensor([2, 3], device=npu_device, dtype=torch.long)

    masked = apply_top_k_only(logits.clone(), k.clone())

    expected = torch.tensor(
        [
            [1.0, float("-inf"), 2.5, float("-inf")],
            [0.0, float("-inf"), 0.3, 4.0],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    assert torch.equal(torch.isinf(masked), torch.isinf(expected))
    assert torch.allclose(
        masked.masked_fill(torch.isinf(masked), 0.0),
        expected.masked_fill(torch.isinf(expected), 0.0),
    )


def test_apply_top_k_top_p_faster_matches_reference(npu_device):
    logits = torch.tensor(
        [
            [2.0, 1.5, 0.2, -0.5],
            [0.5, 0.1, -0.2, 3.0],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    top_ps = torch.tensor([0.7, 0.6], device=npu_device, dtype=torch.float32)
    top_ks = torch.tensor([3, 4], device=npu_device, dtype=torch.int64)

    fast_logits, fast_idx, _ = _apply_top_k_top_p_faster(logits.clone(), top_ps, top_ks)
    fast_full = _scatter_back(fast_logits, fast_idx, logits)

    ref_sorted, ref_idx = apply_top_k_top_p(logits.clone(), top_ks, top_ps)
    ref_full = _scatter_back(ref_sorted, ref_idx, logits)

    assert torch.allclose(fast_full.cpu(), ref_full.cpu(), equal_nan=True)


def test_random_sample_and_choice_return_expected_indices(npu_device):
    stream = torch_npu.npu.Stream()
    probs = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    idx = torch.tensor(
        [
            [5, 6, 7],
            [9, 10, 11],
        ],
        device=npu_device,
        dtype=torch.int64,
    )
    generators = {1: torch.Generator(device="npu").manual_seed(42)}

    sampled = random_sample(probs.clone(), idx, generators, stream)
    choice = random_choice(probs.clone(), generators, stream)

    assert torch.equal(sampled.cpu(), torch.tensor([7, 10]))
    assert torch.equal(choice.cpu(), torch.tensor([2, 1]))


def test_apply_penalties_matches_manual(npu_device):
    get_counts = _ensure_bin_counts()
    logits = torch.tensor(
        [
            [1.5, -0.2, 0.8, 0.0, -1.0],
            [0.6, 0.4, -0.5, 0.9, 0.1],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    prompt_tokens = torch.tensor([[0, 1, 4], [2, 3, 2]], device=npu_device, dtype=torch.long)
    output_tokens = torch.tensor([[1, 4], [3, 2]], device=npu_device, dtype=torch.long)
    presence = torch.tensor([0.2, 0.3], device=npu_device, dtype=torch.float32)
    frequency = torch.tensor([0.4, 0.1], device=npu_device, dtype=torch.float32)
    repetition = torch.tensor([1.5, 1.2], device=npu_device, dtype=torch.float32)

    result = _apply_penalties(
        logits.clone(),
        prompt_tokens,
        output_tokens,
        presence.clone(),
        frequency.clone(),
        repetition.clone(),
        True,
        True,
        True,
    )

    vocab_size = logits.shape[1]
    num_seqs = logits.shape[0]
    _, prompt_mask = get_counts(prompt_tokens, vocab_size, num_seqs)
    output_counts, output_mask = get_counts(output_tokens, vocab_size, num_seqs)

    expected = logits.clone()
    rep = (repetition - 1).unsqueeze(1).repeat(1, vocab_size)
    rep = rep * (prompt_mask | output_mask) + 1
    expected = torch.where(expected > 0, expected / rep, expected * rep)
    expected -= frequency.unsqueeze(1) * output_counts
    expected -= presence.unsqueeze(1) * output_mask

    assert torch.allclose(result, expected, atol=1e-6)


def test_multinomial_seeded_reproducible(npu_device):
    probs = torch.tensor(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.1, 0.3],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    num_samples = 2
    k = num_samples
    seeds = {
        0: torch.Generator(device="npu").manual_seed(10),
        1: torch.Generator(device="npu").manual_seed(20),
    }

    out = _multinomial(probs.clone(), num_samples, k, seeds)

    # Manual reproduction of the algorithm for determinism check
    seeds_expected = {
        0: torch.Generator(device="npu").manual_seed(10),
        1: torch.Generator(device="npu").manual_seed(20),
    }
    q = torch.empty_like(probs).repeat_interleave(num_samples, dim=0)
    start = 0
    for idx in range(len(q) // k):
        end = start + k
        gen = seeds_expected[idx]
        q[start:end].exponential_(1.0, generator=gen)
        start = end
    q.add_(sampler_mod.FP32_EPS)
    expected = probs.repeat_interleave(num_samples, dim=0).div_(q).argmax(dim=1).view(-1, num_samples)

    assert torch.equal(out.cpu(), expected.cpu())


def test_get_logprobs_adapter_handles_greedy_and_rank(npu_device):
    logprobs = torch.tensor(
        [[-1.0, -0.5, -2.0], [-0.1, -0.3, -0.9]],
        device=npu_device,
        dtype=torch.float32,
    )
    sample_results = [[(1, None)], [(0, None)]]

    prompt_lp, sample_lp = _get_logprobs_adapter(
        need_log_probs=False,
        fully_greedy_mode=False,
        slice_indexes=None,
        logprobs=logprobs,
        sampling_metadata=None,
        sample_results=sample_results,
    )

    assert prompt_lp == [None, None]
    assert isinstance(sample_lp[0][0][1], Logprob)
    assert sample_lp[0][0][1].logprob == pytest.approx(logprobs[0, 1].item())
    assert sample_lp[1][0][0].logprob == pytest.approx(logprobs[1, 0].item())


def test_get_logprobs_adapter_greedy_mode(npu_device):
    logprobs = torch.zeros(2, 3, device=npu_device, dtype=torch.float32)
    sample_results = [[(2, None)], [(1, None)]]

    prompt_lp, sample_lp = _get_logprobs_adapter(
        need_log_probs=False,
        fully_greedy_mode=True,
        slice_indexes=None,
        logprobs=logprobs,
        sampling_metadata=None,
        sample_results=sample_results,
    )

    for entry, expected_id in zip(sample_lp, [2, 1]):
        lp = entry[0][expected_id]
        assert isinstance(lp, Logprob)
        assert lp.logprob == 0
        assert lp.rank == 1


def test_rejection_sampler_get_accepted_matches_ratios(npu_device):
    class DummyRejection(RejectionSampler):
        def __init__(self):
            pass

    sampler = DummyRejection()
    sampler._create_uniform_samples = lambda seeded, bs, k_minus_one, device: torch.tensor(
        [[0.2, 0.5, 0.8]], device=device, dtype=torch.float32
    )

    target_probs = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]]],
        device=npu_device,
        dtype=torch.float32,
    )
    draft_probs = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]]],
        device=npu_device,
        dtype=torch.float32,
    )
    draft_token_ids = torch.tensor([[1, 0, 1]], device=npu_device, dtype=torch.long)

    accepted = sampler._get_accepted(target_probs, draft_probs, draft_token_ids, seeded_seqs={})

    dt_ids = draft_token_ids.view(1, 3, 1)
    sel_draft = torch.gather(draft_probs, -1, dt_ids).view(1, 3)
    sel_target = torch.gather(target_probs, -1, dt_ids).view(1, 3)
    ratios = sel_target / sel_draft
    ratios.clamp_max_(1)
    expected = sampler._create_uniform_samples({}, 1, 2, npu_device) < ratios

    assert torch.equal(accepted, expected)


def test_ascend_sampler_get_probs_passthrough_without_softmax(npu_device):
    sampler = AscendSampler()
    sampler.include_gpu_probs_tensor = False
    logits = torch.randn(2, 3, device=npu_device, dtype=torch.float32)

    logprobs, probs = sampler.get_probs_and_logprobs(logits.clone(), not_need_softmax=True)

    assert torch.allclose(logprobs, logits)
    assert torch.allclose(probs, logits)


def test_ascend_topk_topp_sampler_v1_python_fallback_matches_reference(npu_device, monkeypatch):
    monkeypatch.setenv("OMNI_DISABLE_NPU_TOP_K_TOP_P_SAMPLE", "1")
    sampler = AscendTopKTopPSamplerV1()
    logits = torch.tensor([[2.0, 0.1, 3.0, -1.0]], device=npu_device, dtype=torch.float32)
    k = torch.tensor([2], device=npu_device, dtype=torch.int32)
    p = torch.tensor([0.8], device=npu_device, dtype=torch.float32)
    generators = {0: torch.Generator(device="npu").manual_seed(7)}

    out = sampler.forward_native(logits.clone(), generators, k, p)

    masked_logits, idx = apply_top_k_top_p(logits.clone(), k.to(torch.long), p)
    expected = random_sample(masked_logits.softmax(dim=-1, dtype=torch.float32), idx, generators, sampler.dsa_stream)
    assert torch.equal(out.cpu(), expected.cpu())


def test_ascend_sampler_v1_apply_penalties_passthrough(npu_device):
    from omni.accelerators.reasoning_compression.config import ThinkCompressDict
    ThinkCompressDict.reasoner_early_think_stopping_enabled = 0
    sampler = AscendSamplerV1()
    sampler.penalty_cache = None
    logits = torch.randn(2, 4, device=npu_device, dtype=torch.float32)
    sampling_metadata = SamplingMetadataV1(
        temperature=torch.tensor([1.0, 1.0], device=npu_device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(2, device=npu_device),
        presence_penalties=torch.zeros(2, device=npu_device),
        repetition_penalties=torch.ones(2, device=npu_device),
        output_token_ids=[[], []],
        min_tokens={},
        logit_bias=[None, None],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        seq_data=[]
    )

    out = sampler.apply_penalties(logits.clone(), sampling_metadata)

    assert torch.allclose(out, logits)


def test_simple_sampler_prefill_returns_argmax(npu_device):
    class DummyMainSampler:
        def __init__(self):
            self.topk_topp_sampler = None

    sampler = SimpleSampler(main_sampler=DummyMainSampler())
    logits = torch.tensor([[0.1, 2.0, -1.0]], device=npu_device, dtype=torch.float32)
    logits_indices = torch.tensor([0], device=npu_device, dtype=torch.int32)
    input_ids = torch.tensor([1], device=npu_device, dtype=torch.int32)
    sampling_metadata = SamplingMetadataV1(
        temperature=torch.tensor([1.0], device=npu_device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device=npu_device),
        presence_penalties=torch.zeros(1, device=npu_device),
        repetition_penalties=torch.ones(1, device=npu_device),
        output_token_ids=[[]],
        min_tokens={},
        logit_bias=[None],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        seq_data=[]
    )

    sampler_output, forward_tokens, last_accepted_index, accepted_num = sampler(
        input_ids=input_ids,
        logits=logits,
        logits_indices=logits_indices,
        sampling_metadata=sampling_metadata,
        num_decodes=0,
        num_prefills=1,
    )

    assert torch.equal(forward_tokens.view(-1).cpu(), torch.tensor([1]))
    assert torch.equal(sampler_output.sampled_token_ids.cpu(), torch.tensor([[1]], dtype=torch.int32))
    assert torch.equal(last_accepted_index.cpu(), torch.tensor([0], dtype=torch.int32))
    assert accepted_num == 0


def test_sparse_rejection_sampler_resample_when_uncomputed(npu_device, monkeypatch):
    def _fake_random_choice(probs: torch.Tensor, _generators, _stream):
        return probs.argmax(dim=-1)

    monkeypatch.setattr(sampler_mod, "random_choice", _fake_random_choice)

    prob_cache = SimpleNamespace(
        topk_spec_token_ids=torch.tensor([[[2]]], device=npu_device, dtype=torch.int64),
        topk_spec_token_probs=torch.tensor([[[0.6]]], device=npu_device, dtype=torch.float32),
        selected_indices=torch.tensor([[0]], device=npu_device, dtype=torch.int64),
        computed=torch.tensor([False], device=npu_device, dtype=torch.bool),
    )

    class DummyMainSampler:
        def __init__(self):
            self.penalty_cache = None
            self.prob_cache = prob_cache
            self.topk_topp_sampler = SimpleNamespace(dsa_stream=None)

        def __call__(self, logits, sampling_metadata, update_penalty=True):
            tokens = logits.argmax(dim=-1).to(torch.int64)
            return sampler_mod.SamplerOutputV1(sampled_token_ids=tokens.unsqueeze(-1), logprobs_tensors=None)

    main_sampler = DummyMainSampler()
    sampler = SparseRejectionSampler(main_sampler=main_sampler, topk=1, max_num_tokens=2)
    original_reject = sampler._reject_sampling

    def _reject_sampling_fp32(generators, target_probs, topk_ids, topk_probs):
        return original_reject(
            generators,
            target_probs.float(),
            topk_ids,
            topk_probs.float(),
        )

    sampler._reject_sampling = _reject_sampling_fp32
    logits = torch.tensor(
        [
            [0.2, 0.3, 1.5, -0.1],
            [0.1, 1.2, 0.0, -0.4],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    logits_indices = torch.tensor([0, 1], device=npu_device, dtype=torch.int64)
    input_ids = torch.tensor([5, 6], device=npu_device, dtype=torch.int64)
    sampling_metadata = SamplingMetadataV1(
        temperature=torch.tensor([1.0], device=npu_device),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        generators={0: torch.Generator(device="npu").manual_seed(12)},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device=npu_device),
        presence_penalties=torch.zeros(1, device=npu_device),
        repetition_penalties=torch.ones(1, device=npu_device),
        output_token_ids=[[]],
        min_tokens={},
        logit_bias=[None],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        seq_data=[]
    )

    sampler_output, forward_tokens, last_accepted_index, accepted_num = sampler(
        input_ids=input_ids,
        logits=logits,
        logits_indices=logits_indices,
        sampling_metadata=sampling_metadata,
        num_decodes=1,
        num_prefills=0,
    )

    recovered_probs = torch.nn.functional.softmax(logits[0].half(), dim=-1).float()
    draft_probs = torch.zeros_like(recovered_probs)
    draft_probs[prob_cache.topk_spec_token_ids.view(-1)] = prob_cache.topk_spec_token_probs.view(-1)
    expected_token = (recovered_probs - draft_probs).argmax().item()

    assert torch.equal(last_accepted_index.cpu(), torch.tensor([0], dtype=torch.int64))
    assert sampler_output.sampled_token_ids.shape == (1, 2)
    assert sampler_output.sampled_token_ids[0, 0].item() == expected_token
    assert sampler_output.sampled_token_ids[0, 1].item() == -1
    assert torch.equal(accepted_num.cpu(), torch.tensor([0], dtype=torch.int64))


def test_gather_tokens_and_get_greedy_token_update_outputs(npu_device):
    prob_indexes = torch.tensor(
        [[4, 5], [6, 7], [8, 9]],
        device=npu_device,
        dtype=torch.long,
    )
    selection = torch.tensor([1, 0, 1], device=npu_device, dtype=torch.long)
    gathered = gather_tokens(selection.clone(), prob_indexes, torch.tensor([0, 1, 2], device=npu_device))
    assert torch.equal(gathered.cpu(), torch.tensor([5, 6, 9]))

    logprobs = torch.tensor(
        [[0.1, 0.9], [0.2, 1.5], [-0.5, -0.1]],
        device=npu_device,
        dtype=torch.float32,
    )
    probs = torch.ones(3, 2, device=npu_device, dtype=torch.float32)
    sample_indices = torch.tensor([1], device=npu_device, dtype=torch.long)
    sampled_token_ids = torch.full((prob_indexes.shape[0], 1), -1, device=npu_device, dtype=torch.long)

    greedy_samples = _get_greedy_token(
        probs=probs,
        logprobs=logprobs,
        prob_indexes=prob_indexes,
        long_sample_indices=sample_indices,
        include_gpu_probs_tensor=True,
        modify_greedy_probs=True,
        sampled_token_ids_tensor=sampled_token_ids,
    )

    assert greedy_samples.tolist() == [7]
    assert sampled_token_ids[1, 0].item() == 7
    assert torch.allclose(probs[sample_indices], torch.tensor([[0.0, 1.0]], device=npu_device))
    assert torch.equal(probs[0], torch.ones(2, device=npu_device))


def test_check_top_ks_handles_empty_and_passthrough(npu_device):
    assert not _check_top_ks(None, True)
    empty = SimpleNamespace(top_ks=torch.tensor([], device=npu_device))
    assert not _check_top_ks(empty, True)
    populated = SimpleNamespace(top_ks=torch.tensor([1, 2], device=npu_device))
    assert _check_top_ks(populated, True)
    assert not _check_top_ks(populated, False)


def test_need_log_probs_respects_sampling_params():
    params = SimpleNamespace(prompt_logprobs=None, logprobs=None, n=1)
    seq_group = SimpleNamespace(is_prompt=False, sampling_params=params)
    metadata = SimpleNamespace(seq_groups=[seq_group])

    assert not _need_log_probs(metadata, include_gpu_probs_tensor=False)

    params.n = 2
    assert _need_log_probs(metadata, include_gpu_probs_tensor=False)

    params.n = 1
    params.logprobs = 1
    assert _need_log_probs(metadata, include_gpu_probs_tensor=False)

    params.logprobs = None
    assert _need_log_probs(metadata, include_gpu_probs_tensor=True)


def test_apply_penalties_v1_matches_manual(npu_device):
    get_counts = _ensure_bin_counts()
    logits = torch.tensor(
        [[1.0, -0.5, 0.3], [0.6, 0.2, -0.4]],
        device=npu_device,
        dtype=torch.float32,
    )
    prompt_tokens = torch.tensor([[0, 1], [2, 0]], device=npu_device, dtype=torch.long)
    output_tokens = torch.tensor([[2], [1]], device=npu_device, dtype=torch.long)
    vocab_size = logits.shape[1]
    num_seqs = logits.shape[0]
    _, prompt_mask = get_counts(prompt_tokens, vocab_size, num_seqs)
    output_counts, output_mask = get_counts(output_tokens, vocab_size, num_seqs)
    presence = torch.tensor([0.1, 0.2], device=npu_device, dtype=torch.float32)
    frequency = torch.tensor([0.3, 0.4], device=npu_device, dtype=torch.float32)
    repetition = torch.tensor([1.5, 1.1], device=npu_device, dtype=torch.float32)

    result = _apply_penalties_v1(
        logits.clone(),
        prompt_mask,
        output_mask,
        output_counts,
        presence.clone(),
        frequency.clone(),
        repetition.clone(),
        True,
        True,
        True,
    )

    expected = logits.clone()
    rep_penalties = (repetition - 1).unsqueeze(1).repeat(1, vocab_size)
    rep_penalties = rep_penalties * (prompt_mask | output_mask) + 1
    expected = torch.where(expected > 0, expected / rep_penalties, expected * rep_penalties)
    expected -= frequency.unsqueeze(1) * output_counts
    expected -= presence.unsqueeze(1) * output_mask

    assert torch.allclose(result, expected, atol=1e-6)


def test_rejection_sampler_get_recovered_probs_normalizes(npu_device):
    sampler = SimpleNamespace(_smallest_positive_value=torch.tensor(1e-6, device=npu_device))
    target_probs = torch.tensor(
        [[[0.6, 0.4], [0.2, 0.8]]],
        device=npu_device,
        dtype=torch.float32,
    )
    draft_probs = torch.tensor(
        [[[0.1, 0.1], [0.05, 0.05]]],
        device=npu_device,
        dtype=torch.float32,
    )

    recovered = RejectionSampler._get_recovered_probs(sampler, target_probs.clone(), draft_probs.clone())

    assert torch.all(recovered >= 0)
    assert torch.allclose(recovered.sum(dim=-1), torch.ones_like(recovered.sum(dim=-1)))


def test_rejection_sampler_create_output_assigns_bonus_and_metrics(npu_device):
    sampler = SimpleNamespace(
        _num_bonus_tokens=1,
        int64_neg_one=torch.tensor(-1, device=npu_device, dtype=torch.int64),
        cached_indices=None,
        cached_k_tensor=None,
        cached_k=sampler_mod.UNINITIALIZED_CACHED_K_NUM,
        token_id_dtype=torch.int64,
        enable_spec_metric=True,
        num_accepted_tokens=0,
        num_emitted_tokens=0,
        num_draft_tokens=0,
    )
    accepted = torch.tensor([[True, False, False]], device=npu_device)
    substitute_token_ids = torch.tensor([[9, 8, 7]], device=npu_device, dtype=torch.int64)
    draft_token_ids = torch.tensor([[1, 2, 3]], device=npu_device, dtype=torch.int64)
    bonus_token_ids = torch.tensor([5], device=npu_device, dtype=torch.int64)

    output = RejectionSampler._create_output(sampler, accepted, substitute_token_ids, draft_token_ids, bonus_token_ids)

    assert torch.equal(output[:, :3].cpu(), torch.tensor([[1, 8, -1]], dtype=torch.int64))
    assert output.shape[1] == 4
    assert sampler.num_accepted_tokens == 1
    assert sampler.num_draft_tokens == 3
    assert sampler.num_emitted_tokens == 2


def test_simple_sampler_decode_accepts_tokens(npu_device):
    class DummyMainSampler:
        def __init__(self):
            self.topk_topp_sampler = None
            self.penalty_cache = None

    sampler = SimpleSampler(main_sampler=DummyMainSampler())
    logits = torch.tensor(
        [
            [0.0, 1.0, -0.5],
            [1.5, 0.2, -0.1],
        ],
        device=npu_device,
        dtype=torch.float32,
    )
    logits_indices = torch.tensor([0, 1], device=npu_device, dtype=torch.int32)
    input_ids = torch.tensor([5, 1], device=npu_device, dtype=torch.int32)
    sampling_metadata = SamplingMetadataV1(
        temperature=torch.tensor([1.0], device=npu_device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device=npu_device),
        presence_penalties=torch.zeros(1, device=npu_device),
        repetition_penalties=torch.ones(1, device=npu_device),
        output_token_ids=[[]],
        min_tokens={},
        logit_bias=[None],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        seq_data=[]
    )

    sampler_output, forward_tokens, last_accepted_index, accepted_num = sampler(
        input_ids=input_ids,
        logits=logits,
        logits_indices=logits_indices,
        sampling_metadata=sampling_metadata,
        num_decodes=1,
        num_prefills=0,
    )

    assert torch.equal(forward_tokens.cpu(), torch.tensor([[1, 0]], dtype=torch.int32))
    assert torch.equal(sampler_output.sampled_token_ids.cpu(), torch.tensor([[1, 0]], dtype=torch.int32))
    assert torch.equal(last_accepted_index.cpu(), torch.tensor([1], dtype=torch.int32))
    assert torch.equal(accepted_num.cpu(), torch.tensor([1], dtype=torch.int32))
