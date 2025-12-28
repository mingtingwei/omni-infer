import os
import sys
import types
import contextlib
import pytest
import unittest
from unittest import mock
from unittest.mock import Mock, patch, MagicMock
import importlib

import torch
from torch import nn
from torch.nn import Parameter

# NOTE: Do not import npu_worker at module import time; import lazily in helpers/tests to avoid
# import-time side effects during pytest collection and to minimize cross-test interference.


MODULE = "omni.adaptors.vllm.worker.npu_worker"

_MISSING = object()

def _get_npu_worker_module():
    """Lazy import to avoid import-time side effects during pytest collection."""
    return importlib.import_module(MODULE)

_ISOLATED_SYS_MODULE_KEYS = (
    "omni.adaptors.vllm.token_recovery.ha_server",
    "omni.adaptors.vllm.npu_mem_pool",
)



def _sn(**kwargs):
    return types.SimpleNamespace(**kwargs)


@contextlib.contextmanager
def _dummy_cm():
    yield


class _FakeTensor:
    """A minimal scalar-like stand-in for torch.tensor in graph-cache tests.

    Supports:
      - dtype attribute
      - != comparison returning bool
      - ordering (for min())
      - int() conversion
    """

    def __init__(self, value, dtype=None, device=None):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        self.value = int(value)
        self.dtype = dtype
        self.device = device

    def __ne__(self, other):
        ov = other.value if isinstance(other, _FakeTensor) else int(other)
        return self.value != ov

    def __lt__(self, other):
        ov = other.value if isinstance(other, _FakeTensor) else int(other)
        return self.value < ov

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f"_FakeTensor({self.value})"


class _IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors


class _ModelRunnerOutput:
    pass


class TestNPUWorker(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Snapshot global state that can easily leak across tests.
        self._environ_snapshot = os.environ.copy()
        self._cuda_get_dev_props_snapshot = getattr(getattr(torch, "cuda", None), "get_device_properties", _MISSING)
        self._sys_modules_snapshot = {k: sys.modules.get(k, _MISSING) for k in _ISOLATED_SYS_MODULE_KEYS}

    def tearDown(self):
        # Restore global state to avoid polluting other test modules.
        try:
            os.environ.clear()
            os.environ.update(self._environ_snapshot)

            for k, v in self._sys_modules_snapshot.items():
                if v is _MISSING:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

            if self._cuda_get_dev_props_snapshot is not _MISSING:
                try:
                    torch.cuda.get_device_properties = self._cuda_get_dev_props_snapshot
                except Exception:
                    # Some torch builds may not allow assignment here; ignore defensively.
                    pass
        finally:
            super().tearDown()

    # ---------------- helpers ----------------

    def _make_vllm_config(
        self,
        *,
        cache_dtype="auto",
        model_dtype=torch.bfloat16,
        use_mla=False,
        kv_role=None,
    ):
        hf_cfg = _sn(kv_lora_rank=4, qk_rope_head_dim=8, num_hidden_layers=2)
        model_cfg = _sn(
            dtype=model_dtype,
            trust_remote_code=False,
            disable_cascade_attn=False,
            seed=123,
            use_mla=use_mla,
            hf_config=hf_cfg,
        )
        cache_cfg = _sn(cache_dtype=cache_dtype, block_size=16, gpu_memory_utilization=0.5)
        parallel_cfg = _sn(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            disable_custom_all_reduce=False,
        )
        scheduler_cfg = _sn(enable_chunked_prefill=False, max_num_seqs=1)
        device_cfg = _sn(device=_sn(type="npu"))
        kv_cfg = None if kv_role is None else _sn(kv_role=kv_role)
        npu_compilation_cfg = _sn(level=0, use_ge_graph_cached=False)
        additional_config = {}

        return _sn(
            model_config=model_cfg,
            cache_config=cache_cfg,
            parallel_config=parallel_cfg,
            scheduler_config=scheduler_cfg,
            device_config=device_cfg,
            kv_transfer_config=kv_cfg,
            additional_config=additional_config,
            npu_compilation_config=npu_compilation_cfg,
        )

    def _make_worker_stub(self, vllm_config=None, *, is_driver_worker=True):
        if vllm_config is None:
            vllm_config = self._make_vllm_config()

        NPUWorker = _get_npu_worker_module().NPUWorker
        w = NPUWorker.__new__(NPUWorker)
        w.vllm_config = vllm_config
        w.model_config = vllm_config.model_config
        w.cache_config = vllm_config.cache_config
        w.parallel_config = vllm_config.parallel_config
        w.scheduler_config = vllm_config.scheduler_config
        w.device_config = vllm_config.device_config

        w.local_rank = 0
        w.rank = 0
        w.distributed_init_method = "tcp://127.0.0.1:23456"
        w.is_driver_worker = is_driver_worker

        # for _compute_kv_cache_bytes
        w.init_npu_memory = 0

        # for execute_model profiler gates (keep inert by default)
        w.profile_finished = True
        w.profile_already_start = False
        w._use_token = True
        w.profiler_token_threshold = 10**12
        w.profiler_stop_step = 10**12
        w._profile_start_requested = False
        w.profiler = Mock()
        w.enable_token_recover = False


        return w

    # ---------------- tests ----------------

    def test_npuworker_init_sets_cache_dtype_auto_or_explicit(self):
        NPUWorker = _get_npu_worker_module().NPUWorker

        def fake_workerbase_init(
            self,
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=False,
            **kwargs,
        ):
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.parallel_config = vllm_config.parallel_config
            self.scheduler_config = vllm_config.scheduler_config
            self.device_config = vllm_config.device_config
            self.local_rank = local_rank
            self.rank = rank
            self.distributed_init_method = distributed_init_method
            self.is_driver_worker = is_driver_worker
        with patch(f"{MODULE}.WorkerBase.__init__", new=fake_workerbase_init), \
             patch(
             f"{MODULE}.envs",
             _sn(
                 VLLM_LOGGING_CONFIG_PATH=None,
                 VLLM_USE_RAY_SPMD_WORKER=False,
                 VLLM_ENABLE_V1_MULTIPROCESSING=False,
                 VLLM_TORCH_PROFILER_DIR=None,
             ),
         ),  \
             patch.object(NPUWorker, "_init_graph_options", lambda self: setattr(self, "enable_torchair_graph_mode", False)), \
             patch(f"{MODULE}.get_device_properties", Mock()), \
             patch(f"{MODULE}.STR_DTYPE_TO_TORCH_DTYPE", {"float16": torch.float16, "bfloat16": torch.bfloat16}):

            cfg = self._make_vllm_config(cache_dtype="auto", model_dtype=torch.bfloat16)
            w = NPUWorker(cfg, local_rank=0, rank=0, distributed_init_method="tcp://x")
            self.assertEqual(w.cache_dtype, cfg.model_config.dtype)

            cfg2 = self._make_vllm_config(cache_dtype="float16", model_dtype=torch.bfloat16)
            w2 = NPUWorker(cfg2, local_rank=0, rank=0, distributed_init_method="tcp://x")
            self.assertEqual(w2.cache_dtype, torch.float16)
    def test_npuworker_init_enables_token_recover_only_when_ha_and_kv_consumer(self):
        NPUWorker = _get_npu_worker_module().NPUWorker

        def fake_workerbase_init(
            self,
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=False,
            **kwargs,
        ):
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.parallel_config = vllm_config.parallel_config
            self.scheduler_config = vllm_config.scheduler_config
            self.device_config = vllm_config.device_config
            self.local_rank = local_rank
            self.rank = rank
            self.distributed_init_method = distributed_init_method
            self.is_driver_worker = is_driver_worker

        with patch(f"{MODULE}.WorkerBase.__init__", new=fake_workerbase_init), \
            patch(
                f"{MODULE}.envs",
                _sn(
                    VLLM_LOGGING_CONFIG_PATH=None,
                    VLLM_USE_RAY_SPMD_WORKER=False,
                    VLLM_ENABLE_V1_MULTIPROCESSING=False,
                    VLLM_TORCH_PROFILER_DIR=None,
                ),
            ), \
            patch.object(NPUWorker, "_init_graph_options", lambda self: None), \
            patch(f"{MODULE}.get_device_properties", Mock()), \
            patch(f"{MODULE}.ENV", _sn(use_ha=True, ha_port=12345)), \
            patch(f"{MODULE}.STR_DTYPE_TO_TORCH_DTYPE", {"float16": torch.float16, "bfloat16": torch.bfloat16}):

            cfg = self._make_vllm_config(cache_dtype="auto", kv_role="kv_consumer")
            w = NPUWorker(cfg, local_rank=0, rank=0, distributed_init_method="tcp://x")
            self.assertTrue(w.enable_token_recover)

            cfg2 = self._make_vllm_config(cache_dtype="auto", kv_role="kv_producer")
            w2 = NPUWorker(cfg2, local_rank=0, rank=0, distributed_init_method="tcp://x")
            self.assertFalse(w2.enable_token_recover)

            with patch(f"{MODULE}.ENV", _sn(use_ha=False, ha_port=12345)):
                cfg3 = self._make_vllm_config(cache_dtype="auto", kv_role="kv_consumer")
                w3 = NPUWorker(cfg3, local_rank=0, rank=0, distributed_init_method="tcp://x")
                self.assertFalse(w3.enable_token_recover)

            cfg4 = self._make_vllm_config(cache_dtype="auto", kv_role=None)
            w4 = NPUWorker(cfg4, local_rank=0, rank=0, distributed_init_method="tcp://x")
            self.assertFalse(w4.enable_token_recover)
    def test_sleep_returns_early_when_sleep_mode_unavailable(self):
        w = self._make_worker_stub()
        with patch(f"{MODULE}.NPUPlatform.is_sleep_mode_available", return_value=False), \
             patch(f"{MODULE}.logger") as mlogger:
            w.sleep(level=1)
            mlogger.error.assert_called()

    def test_wake_up_returns_early_when_sleep_mode_unavailable(self):
        w = self._make_worker_stub()
        with patch(f"{MODULE}.NPUPlatform.is_sleep_mode_available", return_value=False), \
             patch(f"{MODULE}.logger") as mlogger:
            w.wake_up(tags=["weights"])
            mlogger.error.assert_called()

    def test_init_device_uses_cpu_when_no_npu_mock_set(self):
        current_platform = _get_npu_worker_module().current_platform

        w = self._make_worker_stub()
        w.device_config = _sn(device=_sn(type="npu"))
        w.local_rank = 3
        w.model_config.seed = 1

        with patch.dict(os.environ, {"NO_NPU_MOCK": "1"}), \
             patch.object(current_platform, "device_type", "npu"), \
             patch(f"{MODULE}.NPUPlatform") as mplat, \
             patch.object(w, "_init_omni_placement_configs", Mock()), \
             patch.object(w, "_init_model_best_practice_configs", Mock()), \
             patch.object(w, "_init_worker_distributed_environment", Mock()), \
             patch(f"{MODULE}.set_random_seed") as mseed, \
             patch(f"{MODULE}.NPUModelRunner") as mrunner, \
             patch.object(w, "_init_profiler", return_value=Mock()):
            w.init_device()
            self.assertEqual(w.device.type, "cpu")
            mplat.set_device.assert_not_called()
            mseed.assert_called_once_with(w.model_config.seed)
            mrunner.assert_called_once()

    def test_init_device_raises_on_unsupported_device_type(self):
        current_platform = _get_npu_worker_module().current_platform

        w = self._make_worker_stub()
        w.device_config = _sn(device=_sn(type="cuda"))
        with patch.object(current_platform, "device_type", "npu"):
            with self.assertRaises(RuntimeError):
                w.init_device()

    def test_init_device_creates_model_runner_and_profiler_and_starts_ha_server_on_rank0_when_enabled(self):
        current_platform = _get_npu_worker_module().current_platform

        w = self._make_worker_stub()
        w.device_config = _sn(device=_sn(type="npu"))
        w.local_rank = 0
        w.enable_token_recover = True

        fake_world_group = _sn(rank=0)

        ha_server_mod = types.ModuleType("omni.adaptors.vllm.token_recovery.ha_server")
        ha_server_mod.start_server = Mock()

        def _fake_torch_device(spec):
            # avoid requiring real "npu" device support in pure-CPU CI
            spec = str(spec)
            if spec.startswith("cpu"):
                return torch.device("cpu")
            return _sn(type=spec.split(":")[0])

        with patch.dict(sys.modules, {"omni.adaptors.vllm.token_recovery.ha_server": ha_server_mod}), \
             patch.dict(os.environ, {"NO_NPU_MOCK": "0"}), \
             patch.object(current_platform, "device_type", "npu"), \
             patch(f"{MODULE}.torch.device", side_effect=_fake_torch_device), \
             patch(f"{MODULE}.NPUPlatform") as mplat, \
             patch.object(w, "_init_omni_placement_configs", Mock()), \
             patch.object(w, "_init_model_best_practice_configs", Mock()), \
             patch.object(w, "_init_worker_distributed_environment", Mock()), \
             patch(f"{MODULE}.set_random_seed") as mseed, \
             patch(f"{MODULE}.NPUModelRunner") as mrunner, \
             patch.object(w, "_init_profiler", return_value=Mock()) as mprof, \
             patch(f"{MODULE}.get_world_group", return_value=fake_world_group), \
             patch(f"{MODULE}.ENV", _sn(ha_port=4567, use_ha=True)):

            mplat.mem_get_info.return_value = (1000, 2000)

            w.init_device()
            self.assertEqual(w.device.type, "npu")
            mrunner.assert_called_once_with(w.vllm_config, w.device)
            mprof.assert_called_once()
            ha_server_mod.start_server.assert_called_once_with(4567)
            mseed.assert_called_once()

    def test_page_size_bytes_coef_depends_on_use_mla(self):
        cfg = self._make_vllm_config(use_mla=False)
        w = self._make_worker_stub(cfg)
        bytes_no_mla = w.page_size_bytes()

        cfg2 = self._make_vllm_config(use_mla=True)
        w2 = self._make_worker_stub(cfg2)
        bytes_mla = w2.page_size_bytes()

        self.assertEqual(bytes_no_mla, 2 * bytes_mla)

    def test_determine_available_memory_short_circuits_when_no_npu_mock(self):
        w = self._make_worker_stub()
        w.enable_torchair_graph_mode = False
        with patch.dict(os.environ, {"NO_NPU_MOCK": "1"}), \
             patch.object(w, "_compute_kv_cache_bytes", side_effect=AssertionError("should not be called")):
            self.assertEqual(w.determine_available_memory(), 100000000)

    def test_determine_available_memory_non_graph_mode_returns_compute_and_may_trigger_omni_placement_planner(self):
        w = self._make_worker_stub()
        w.enable_torchair_graph_mode = False
        w.model_runner = _sn(planner=_sn(start_dynamic_optimize_expert_load_balance=Mock()))

        with patch.dict(os.environ, {"NO_NPU_MOCK": "0"}), \
             patch.object(w, "_compute_kv_cache_bytes", return_value=123), \
             patch(f"{MODULE}.clear_var") as mclear, \
             patch(f"{MODULE}.model_extra_config", _sn(task_config=_sn(enable_omni_placement=True))):
            out = w.determine_available_memory()
            self.assertEqual(out, 123)
            mclear.assert_called()
            w.model_runner.planner.start_dynamic_optimize_expert_load_balance.assert_called_once()

    def test_determine_available_memory_graph_cache_hit_validates_all_ranks_consistency(self):
        w = self._make_worker_stub()
        w.enable_torchair_graph_mode = True
        w.use_cached_npu_graph = True
        fake_world_group = _sn(world_size=2, cpu_group=object())

        def fake_all_gather(out_list, in_tensor, group=None):
            out_list[0] = _FakeTensor(int(in_tensor))
            out_list[1] = _FakeTensor(int(in_tensor))

        with patch.dict(os.environ, {"NO_NPU_MOCK": "0"}), \
             patch.object(w, "_compute_kv_cache_bytes", return_value=111), \
             patch(f"{MODULE}.get_world_group", return_value=fake_world_group), \
             patch(f"{MODULE}.check_torchair_cache_exists", return_value=True), \
             patch(f"{MODULE}.check_block_num_cache_exist", return_value=True), \
             patch(f"{MODULE}.read_block_num_from_file", return_value=999), \
             patch(f"{MODULE}.torch.tensor", side_effect=lambda v, dtype=None, device=None: _FakeTensor(v, dtype=dtype, device=device)), \
             patch(f"{MODULE}.dist.all_gather", side_effect=fake_all_gather), \
             patch(f"{MODULE}.torch.distributed.get_rank", return_value=0), \
             patch(f"{MODULE}.clear_var") as mclear:
            out = w.determine_available_memory()
            self.assertEqual(out, 999)
            mclear.assert_called()

    def test_determine_available_memory_graph_cache_miss_gathers_min_and_writes_cache_file(self):
        w = self._make_worker_stub()
        w.enable_torchair_graph_mode = True
        w.use_cached_npu_graph = True
        fake_world_group = _sn(world_size=2, cpu_group=object())

        def fake_all_gather(out_list, in_tensor, group=None):
            out_list[0] = _FakeTensor(120)
            out_list[1] = _FakeTensor(80)

        with patch.dict(os.environ, {"NO_NPU_MOCK": "0"}), \
             patch.object(w, "_compute_kv_cache_bytes", return_value=120), \
             patch(f"{MODULE}.get_world_group", return_value=fake_world_group), \
             patch(f"{MODULE}.check_torchair_cache_exists", return_value=False), \
             patch(f"{MODULE}.check_block_num_cache_exist", return_value=False), \
             patch(f"{MODULE}.delete_torchair_cache_file") as mdel, \
             patch(f"{MODULE}.write_block_num_to_file") as mwrite, \
             patch(f"{MODULE}.torch.tensor", side_effect=lambda v, dtype=None, device=None: _FakeTensor(v, dtype=dtype, device=device)), \
             patch(f"{MODULE}.dist.all_gather", side_effect=fake_all_gather), \
             patch(f"{MODULE}.torch.distributed.get_rank", return_value=0), \
             patch(f"{MODULE}.clear_var") as mclear:
            out = w.determine_available_memory()
            self.assertEqual(out, 80)
            mdel.assert_called_once()
            mwrite.assert_called_once_with(0, 80)
            mclear.assert_called()

    def test_determine_available_memory_graph_cache_hit_raises_when_read_block_num_invalid(self):
        w = self._make_worker_stub()
        w.enable_torchair_graph_mode = True
        w.use_cached_npu_graph = True
        fake_world_group = _sn(world_size=1, cpu_group=object())

        with patch.dict(os.environ, {"NO_NPU_MOCK": "0"}), \
             patch.object(w, "_compute_kv_cache_bytes", return_value=111), \
             patch(f"{MODULE}.get_world_group", return_value=fake_world_group), \
             patch(f"{MODULE}.check_torchair_cache_exists", return_value=True), \
             patch(f"{MODULE}.check_block_num_cache_exist", return_value=True), \
             patch(f"{MODULE}.torch.distributed.get_rank", return_value=0), \
             patch(f"{MODULE}.read_block_num_from_file", return_value=-1):
            with self.assertRaises(RuntimeError):
                w.determine_available_memory()

    def test_compute_kv_cache_bytes_raises_when_peak_memory_non_positive(self):
        w = self._make_worker_stub()
        w.init_npu_memory = 1000
        w.cache_config.gpu_memory_utilization = 0.5
        w.model_runner = _sn(profile_run=Mock())

        with patch(f"{MODULE}.NPUPlatform.empty_cache"), \
             patch(f"{MODULE}.NPUPlatform.mem_get_info", return_value=(1000, 2000)), \
             patch("gc.collect"):
            with self.assertRaises(RuntimeError):
                w._compute_kv_cache_bytes()

    def test_compute_kv_cache_bytes_returns_non_negative_int(self):
        w = self._make_worker_stub()
        w.init_npu_memory = 1000
        w.cache_config.gpu_memory_utilization = 0.5
        w.model_runner = _sn(profile_run=Mock())

        with patch(f"{MODULE}.NPUPlatform.empty_cache"), \
             patch(f"{MODULE}.NPUPlatform.mem_get_info", return_value=(700, 2000)), \
             patch("gc.collect"):
            out = w._compute_kv_cache_bytes()
            self.assertIsInstance(out, int)
            self.assertGreaterEqual(out, 0)

    def test_execute_model_returns_output_only_for_driver_worker(self):
        w = self._make_worker_stub(is_driver_worker=False)
        w.execute_model_wrapper = Mock(return_value=_ModelRunnerOutput())

        with patch(f"{MODULE}.envs", _sn(VLLM_TORCH_PROFILER_DIR=None)):
            out = w.execute_model(scheduler_output=Mock())
            self.assertIsNone(out)

        w2 = self._make_worker_stub(is_driver_worker=True)
        expected = _ModelRunnerOutput()
        w2.execute_model_wrapper = Mock(return_value=expected)

        with patch(f"{MODULE}.envs", _sn(VLLM_TORCH_PROFILER_DIR=None)):
            out2 = w2.execute_model(scheduler_output=Mock())
            self.assertIs(out2, expected)

    def test_execute_model_token_recompute_triggers_fallback_and_second_execute(self):
        w = self._make_worker_stub(is_driver_worker=True)
        w.execute_model_wrapper = Mock(side_effect=[{"dummy": 1}, _ModelRunnerOutput()])
        w.model_runner = _sn(recompute_fallback=Mock())

        with patch(f"{MODULE}.envs", _sn(VLLM_TORCH_PROFILER_DIR=None)), \
             patch(f"{MODULE}.is_token_recompute", return_value=True):
            out = w.execute_model(scheduler_output=Mock())
            self.assertIsInstance(out, _ModelRunnerOutput)
            self.assertEqual(w.execute_model_wrapper.call_count, 2)
            w.model_runner.recompute_fallback.assert_called_once()

    def test_execute_model_wrapper_pipeline_non_last_rank_sends_and_returns_none(self):
        w = self._make_worker_stub()
        pp_group = _sn(
            is_first_rank=True,
            is_last_rank=False,
            recv_tensor_dict=Mock(return_value={"x": 1}),
            send_tensor_dict=Mock(),
        )
        w.model_runner = _sn(execute_model=Mock(return_value=_IntermediateTensors({"t": 1})))

        worker_cls = type(w)
        func = getattr(worker_cls.execute_model_wrapper, "__wrapped__", worker_cls.execute_model_wrapper)

        with patch(f"{MODULE}.get_pp_group", return_value=pp_group), \
             patch(f"{MODULE}.IntermediateTensors", _IntermediateTensors), \
             patch(f"{MODULE}.ModelRunnerOutput", _ModelRunnerOutput):
            out = func(w, Mock())
            self.assertIsNone(out)
            pp_group.send_tensor_dict.assert_called_once_with({"t": 1}, all_gather_group=None)

    def test_execute_model_wrapper_pipeline_last_rank_returns_model_runner_output(self):
        w = self._make_worker_stub()
        pp_group = _sn(
            is_first_rank=True,
            is_last_rank=True,
            recv_tensor_dict=Mock(return_value={"x": 1}),
            send_tensor_dict=Mock(),
        )
        expected = _ModelRunnerOutput()
        w.model_runner = _sn(execute_model=Mock(return_value=expected))

        worker_cls = type(w)
        func = getattr(worker_cls.execute_model_wrapper, "__wrapped__", worker_cls.execute_model_wrapper)

        with patch(f"{MODULE}.get_pp_group", return_value=pp_group), \
             patch(f"{MODULE}.IntermediateTensors", _IntermediateTensors), \
             patch(f"{MODULE}.ModelRunnerOutput", _ModelRunnerOutput):
            out = func(w, Mock())
            self.assertIs(out, expected)
            pp_group.send_tensor_dict.assert_not_called()

    def test_load_model_uses_memory_pool_context_when_sleep_mode_available(self):
        w = self._make_worker_stub()
        w.enable_token_recover = False
        w.model_runner = _sn(load_model=Mock())

        mem_pool_mod = types.ModuleType("omni.adaptors.vllm.npu_mem_pool")
        allocator = _sn(
            get_current_usage=Mock(return_value=0),
            use_memory_pool=Mock(return_value=_dummy_cm()),
        )
        mem_pool_mod.NpuMemAllocator = _sn(get_instance=Mock(return_value=allocator))

        with patch.dict(sys.modules, {"omni.adaptors.vllm.npu_mem_pool": mem_pool_mod}), \
             patch(f"{MODULE}.NPUPlatform.is_sleep_mode_available", return_value=True):
            w.load_model()
            allocator.use_memory_pool.assert_called_once_with(tag="weights")
            w.model_runner.load_model.assert_called_once()

    def test_initialize_from_config_selects_omni_kv_cache_or_normal_kv_cache(self):
        w = self._make_worker_stub()
        w.model_runner = _sn(
            initialize_omni_kv_cache=Mock(),
            initialize_kv_cache=Mock(),
        )

        with patch(f"{MODULE}.NPUPlatform.is_sleep_mode_available", return_value=False):
            with patch(f"{MODULE}.model_extra_config", _sn(operator_opt_config=_sn(use_omni_cache=True))):
                w.initialize_from_config(kv_cache_config=Mock())
                w.model_runner.initialize_omni_kv_cache.assert_called_once()
                w.model_runner.initialize_kv_cache.assert_not_called()

            w.model_runner.initialize_omni_kv_cache.reset_mock()
            w.model_runner.initialize_kv_cache.reset_mock()

            with patch(f"{MODULE}.model_extra_config", _sn(operator_opt_config=_sn(use_omni_cache=False))):
                w.initialize_from_config(kv_cache_config=Mock())
                w.model_runner.initialize_kv_cache.assert_called_once()
                w.model_runner.initialize_omni_kv_cache.assert_not_called()

    def test_init_worker_distributed_environment_invokes_dist_init_and_model_parallel_and_kv_transfer_init(self):
        w = self._make_worker_stub()
        w.parallel_config = _sn(
            world_size=8,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            disable_custom_all_reduce=False,
        )
        w.rank = 3
        w.local_rank = 1
        w.distributed_init_method = "tcp://127.0.0.1:9999"

        with patch(f"{MODULE}.set_custom_all_reduce") as mset, \
             patch(f"{MODULE}.init_distributed_environment") as minit, \
             patch(f"{MODULE}.ensure_model_parallel_initialized") as mensure_mp, \
             patch(f"{MODULE}.ensure_kv_transfer_initialized") as mensure_kv:
            w._init_worker_distributed_environment()
            mset.assert_called_once_with(True)
            minit.assert_called_once_with(8, 3, "tcp://127.0.0.1:9999", 1, "hccl")
            mensure_mp.assert_called_once_with(2, 1)
            mensure_kv.assert_called_once_with(w.vllm_config)


if __name__ == "__main__":
    unittest.main()
