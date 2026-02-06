import types
import multiprocessing as mp
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Parameter

pytestmark = [
    pytest.mark.npu_2cards_4dies,
    pytest.mark.npu,
]

torch_npu = pytest.importorskip("torch_npu")


def _npu_count() -> int:
    try:
        return int(torch_npu.npu.device_count())
    except Exception:
        return 0


class _DummyGroup:
    def __init__(self, world_size: int, rank_in_group: int):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _FusedMoEStub:
    @staticmethod
    def make_expert_params_mapping(*args, **kwargs):
        return []


def _clear_tensor_attr(t: torch.Tensor, name: str) -> None:
    if hasattr(t, name):
        try:
            delattr(t, name)
        except Exception:
            pass


def _worker_tp2(rank: int, world_size: int, queue: "mp.Queue"):
    pangu_mod = None
    parallel_state_mod = None
    orig_get_tp_group = None
    orig_parallel_get_tp_group = None
    orig_FusedMoE = None
    orig_is_pp_missing_parameter = None
    orig_logger = None

    try:
        import omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2 as pangu_mod
        import vllm.distributed.parallel_state as parallel_state_mod

        torch_npu.npu.set_device(rank)
        npu_device = torch.device(f"npu:{rank}")

        orig_get_tp_group = pangu_mod.get_tp_group
        orig_parallel_get_tp_group = parallel_state_mod.get_tp_group
        orig_FusedMoE = pangu_mod.FusedMoE
        orig_is_pp_missing_parameter = pangu_mod.is_pp_missing_parameter
        orig_logger = pangu_mod.logger

        pangu_mod.get_tp_group = lambda: _DummyGroup(world_size=world_size, rank_in_group=rank)
        parallel_state_mod.get_tp_group = lambda: _DummyGroup(world_size=world_size, rank_in_group=rank)
        pangu_mod.FusedMoE = _FusedMoEStub
        pangu_mod.is_pp_missing_parameter = lambda name, self: False
        pangu_mod.logger = SimpleNamespace(warning_once=Mock())

        model = pangu_mod.PanguProMoEV2ForCausalLM.__new__(pangu_mod.PanguProMoEV2ForCausalLM)
        model.config = types.SimpleNamespace(num_experts=0)
        model.model = types.SimpleNamespace(end_layer=1)

        key_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
        value_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
        kv_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
        sink_key = Parameter(torch.zeros((1, 2), device=npu_device, dtype=torch.float16))
        sink_value = Parameter(torch.zeros((1, 2), device=npu_device, dtype=torch.float16))

        params = {
            "model.layers.0.self_attn.attn.key_antiquant_scale": key_scale,
            "model.layers.0.self_attn.attn.value_antiquant_scale": value_scale,
            "model.layers.0.self_attn.attn.kv_scale": kv_scale,
            "model.layers.0.self_attn.param_sink_key": sink_key,
            "model.layers.0.self_attn.param_sink_value": sink_value,
        }

        def _named_parameters():
            yield from params.items()

        model.named_parameters = _named_parameters  # type: ignore[assignment]

        # =========================================
        # First load
        # =========================================
        w_k_full = torch.tensor([11.0, 22.0], device=npu_device, dtype=torch.float16)
        w_v_full = torch.tensor([33.0, 44.0], device=npu_device, dtype=torch.float16)
        w_kv = torch.tensor([55.0], device=npu_device, dtype=torch.float16)

        w_sink_k_full = torch.tensor([[1.0, 2.0],
                                      [3.0, 4.0]], device=npu_device, dtype=torch.float16)
        w_sink_v_full = torch.tensor([[10.0, 20.0],
                                      [30.0, 40.0]], device=npu_device, dtype=torch.float16)

        weights = [
            ("model.layers.0.self_attn.k_proj.kv_cache_scale", w_k_full),
            ("model.layers.0.self_attn.v_proj.kv_cache_scale", w_v_full),
            ("model.layers.0.self_attn.kv_scale", w_kv),
            ("model.layers.0.self_attn.param_sink_key", w_sink_k_full),
            ("model.layers.0.self_attn.param_sink_value", w_sink_v_full),
        ]

        loaded = pangu_mod.PanguProMoEV2ForCausalLM.load_weights(model, weights)
        if hasattr(torch_npu, "synchronize"):
            torch_npu.synchronize()

        assert "model.layers.0.self_attn.attn.key_antiquant_scale" in loaded
        assert "model.layers.0.self_attn.attn.value_antiquant_scale" in loaded
        assert "model.layers.0.self_attn.attn.kv_scale" in loaded
        assert "model.layers.0.self_attn.param_sink_key" in loaded
        assert "model.layers.0.self_attn.param_sink_value" in loaded

        expected_k = torch.tensor([11.0], device=npu_device, dtype=torch.float16) if rank == 0 \
            else torch.tensor([22.0], device=npu_device, dtype=torch.float16)
        expected_v = torch.tensor([33.0], device=npu_device, dtype=torch.float16) if rank == 0 \
            else torch.tensor([44.0], device=npu_device, dtype=torch.float16)

        assert torch.equal(key_scale.data, expected_k)
        assert torch.equal(value_scale.data, expected_v)
        assert torch.equal(kv_scale.data, w_kv)

        expected_sink_k = w_sink_k_full[0:1, :] if rank == 0 else w_sink_k_full[1:2, :]
        expected_sink_v = w_sink_v_full[0:1, :] if rank == 0 else w_sink_v_full[1:2, :]

        assert torch.equal(sink_key.data, expected_sink_k)
        assert torch.equal(sink_value.data, expected_sink_v)

        # Save ptrs
        key_ptr0 = key_scale.data_ptr()
        value_ptr0 = value_scale.data_ptr()
        kv_ptr0 = kv_scale.data_ptr()
        sink_k_ptr0 = sink_key.data_ptr()
        sink_v_ptr0 = sink_value.data_ptr()

        # Clear non-idempotent metadata before reload
        _clear_tensor_attr(key_scale, "is_2_dims")
        _clear_tensor_attr(value_scale, "is_2_dims")
        _clear_tensor_attr(kv_scale, "is_2_dims")
        _clear_tensor_attr(sink_key, "is_2_dims")
        _clear_tensor_attr(sink_value, "is_2_dims")

        # =========================================
        # Second load (reload new weights, same structure)
        # =========================================
        w_k_full2 = torch.tensor([111.0, 222.0], device=npu_device, dtype=torch.float16)
        w_v_full2 = torch.tensor([333.0, 444.0], device=npu_device, dtype=torch.float16)
        w_kv2 = torch.tensor([555.0], device=npu_device, dtype=torch.float16)

        w_sink_k_full2 = torch.tensor([[101.0, 102.0],
                                       [103.0, 104.0]], device=npu_device, dtype=torch.float16)
        w_sink_v_full2 = torch.tensor([[201.0, 202.0],
                                       [203.0, 204.0]], device=npu_device, dtype=torch.float16)

        weights2 = [
            ("model.layers.0.self_attn.k_proj.kv_cache_scale", w_k_full2),
            ("model.layers.0.self_attn.v_proj.kv_cache_scale", w_v_full2),
            ("model.layers.0.self_attn.kv_scale", w_kv2),
            ("model.layers.0.self_attn.param_sink_key", w_sink_k_full2),
            ("model.layers.0.self_attn.param_sink_value", w_sink_v_full2),
        ]

        loaded2 = pangu_mod.PanguProMoEV2ForCausalLM.load_weights(model, weights2)
        if hasattr(torch_npu, "synchronize"):
            torch_npu.synchronize()

        assert "model.layers.0.self_attn.attn.key_antiquant_scale" in loaded2
        assert "model.layers.0.self_attn.attn.value_antiquant_scale" in loaded2
        assert "model.layers.0.self_attn.attn.kv_scale" in loaded2
        assert "model.layers.0.self_attn.param_sink_key" in loaded2
        assert "model.layers.0.self_attn.param_sink_value" in loaded2

        expected_k2 = torch.tensor([111.0], device=npu_device, dtype=torch.float16) if rank == 0 \
            else torch.tensor([222.0], device=npu_device, dtype=torch.float16)
        expected_v2 = torch.tensor([333.0], device=npu_device, dtype=torch.float16) if rank == 0 \
            else torch.tensor([444.0], device=npu_device, dtype=torch.float16)

        assert torch.equal(key_scale.data, expected_k2)
        assert torch.equal(value_scale.data, expected_v2)
        assert torch.equal(kv_scale.data, w_kv2)

        expected_sink_k2 = w_sink_k_full2[0:1, :] if rank == 0 else w_sink_k_full2[1:2, :]
        expected_sink_v2 = w_sink_v_full2[0:1, :] if rank == 0 else w_sink_v_full2[1:2, :]

        assert torch.equal(sink_key.data, expected_sink_k2)
        assert torch.equal(sink_value.data, expected_sink_v2)

        # data_ptr must remain stable (in-place reload)
        assert key_scale.data_ptr() == key_ptr0
        assert value_scale.data_ptr() == value_ptr0
        assert kv_scale.data_ptr() == kv_ptr0
        assert sink_key.data_ptr() == sink_k_ptr0
        assert sink_value.data_ptr() == sink_v_ptr0

        queue.put(("ok", rank))

    except Exception as exc:
        queue.put(("err", rank, repr(exc)))
        raise
    finally:
        if pangu_mod is not None and orig_get_tp_group is not None:
            pangu_mod.get_tp_group = orig_get_tp_group
        if parallel_state_mod is not None and orig_parallel_get_tp_group is not None:
            parallel_state_mod.get_tp_group = orig_parallel_get_tp_group
        if pangu_mod is not None and orig_FusedMoE is not None:
            pangu_mod.FusedMoE = orig_FusedMoE
        if pangu_mod is not None and orig_is_pp_missing_parameter is not None:
            pangu_mod.is_pp_missing_parameter = orig_is_pp_missing_parameter
        if pangu_mod is not None and orig_logger is not None:
            pangu_mod.logger = orig_logger


@pytest.mark.npu
def test_load_weights_on_npu_real_loader_tp2():
    if _npu_count() < 2:
        pytest.skip("Need >=2 NPUs for TP=2 real sharding test")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    world_size = 2

    for r in range(world_size):
        p = ctx.Process(target=_worker_tp2, args=(r, world_size, q))
        p.start()
        procs.append(p)

    results = [q.get() for _ in range(world_size)]
    for p in procs:
        p.join()

    assert all(x[0] == "ok" for x in results), f"Worker failures: {results}"
