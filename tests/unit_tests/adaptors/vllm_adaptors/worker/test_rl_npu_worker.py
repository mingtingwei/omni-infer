import types
from types import SimpleNamespace
import pytest
from unittest.mock import Mock
import torch
from omni.adaptors.vllm.worker.npu_worker import RLNPUWorker
from omni.adaptors.vllm.platform import ConfigUpdater

torch_npu = pytest.importorskip("torch_npu")
import omni.adaptors.vllm.worker.npu_worker as npu_worker_module
from omni.models.config_loader.loader import model_extra_config


class DummyKVCacheConfig:
    def __init__(self) -> None:
        self.batch_size = 2
        self.num_heads = 2
        self.seq_len = 6
        self.head_dim = 128
        self.num_layers = 2

class FakeModel:
    def __init__(self, device):
        self.device = device

    def to(self, device):
        self.device = device
        return self

@pytest.fixture
def fake_platform(monkeypatch):
    class FakePlatform:
        @staticmethod
        def is_sleep_mode_available():
            return True

        @staticmethod
        def mem_get_info():
            # free, total
            return (1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)

        @staticmethod
        def empty_cache():
            return None

    monkeypatch.setattr(npu_worker_module, "NPUPlatform", FakePlatform)
    return FakePlatform


@pytest.fixture
def dummy_worker():
    w = object.__new__(RLNPUWorker)
    w.device = "npu"    
    w.kv_cache_config = DummyKVCacheConfig()
    return w


def test_rlnpuworker_has_expected_api(dummy_worker):
    assert hasattr(dummy_worker, "sleep")
    assert hasattr(dummy_worker, "wake_up")
    assert hasattr(dummy_worker, "load_model")
    assert hasattr(dummy_worker, "initialize_from_config")


def test_sleep_and_wake_up(fake_platform, monkeypatch, dummy_worker):
    # Ensure calling sleep/wake_up on RLNPUWorker with patched platform/allocator does not raise.
    fake_tp = SimpleNamespace(all_reduce=lambda x:None)
    fake_dp = SimpleNamespace(all_reducce=lambda x:None, world_size=1)

    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_dp_group", lambda: fake_dp)

    kv_config = dummy_worker.kv_cache_config
    kv_caches = [[torch.zeros((kv_config.batch_size, kv_config.num_heads, kv_config.seq_len, kv_config.head_dim), 
                dtype=torch.float32, device=dummy_worker.device)
                for _ in range(2)] for _ in range(kv_config.num_layers)]
    
    called = {"unregister_kv_caches": False, "reregister_kv_caches": False}
    def fake_unregister_kv_caches():
        called["unregister_kv_caches"] = True

    def fake_reregister_kv_caches():
        called["reregister_kv_caches"] = True

    dummy_worker.model_runner = SimpleNamespace(   
        unregister_kv_caches = fake_unregister_kv_caches,
        reregister_kv_caches = fake_reregister_kv_caches,     
        kv_caches = kv_caches,
        model=FakeModel(dummy_worker.device),
    )

    id_kv_caches_before_sleep = id(dummy_worker.model_runner.kv_caches)
    dummy_worker.sleep(level=1)
    assert dummy_worker.kv_nbytes is not None
    assert dummy_worker.kv_nbytes[0][0] ==  \
        kv_config.batch_size * kv_config.num_heads * kv_config.seq_len * kv_config.head_dim * torch.float32.itemsize
    assert called["unregister_kv_caches"]
    assert dummy_worker.model_runner.kv_caches[0][0].untyped_storage().nbytes() == 0
    assert dummy_worker.model_runner.model.device == "cpu"

    dummy_worker.wake_up(tags=["weights"])
    dummy_worker.wake_up(tags=["kv_cache"])
    id_kv_caches_after_wake_up = id(dummy_worker.model_runner.kv_caches)
    assert id_kv_caches_before_sleep == id_kv_caches_after_wake_up
    assert dummy_worker.model_runner.kv_caches[0][0].untyped_storage().nbytes() == \
        dummy_worker.kv_nbytes[0][0]
    assert called["reregister_kv_caches"]
    assert dummy_worker.model_runner.model.device == "npu"

def test_sleep_and_wake_up_with_drafter(fake_platform, monkeypatch, dummy_worker):
    # Ensure calling sleep/wake_up on RLNPUWorker with patched platform/allocator does not raise.
    fake_tp = SimpleNamespace(all_reduce=lambda x:None)
    fake_dp = SimpleNamespace(all_reduce=lambda x:None, world_size=1)

    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_dp_group", lambda: fake_dp)

    kv_config = dummy_worker.kv_cache_config
    kv_caches = [[torch.zeros((kv_config.batch_size, kv_config.num_heads, kv_config.seq_len, kv_config.head_dim), 
                dtype=torch.float32, device=dummy_worker.device)
                for _ in range(2)] for _ in range(kv_config.num_layers)]
    
    called = {"unregister_kv_caches": False, "reregister_kv_caches": False}
    def fake_unregister_kv_caches():
        called["unregister_kv_caches"] = True

    def fake_reregister_kv_caches():
        called["reregister_kv_caches"] = True

    dummy_worker.model_runner = SimpleNamespace(   
        unregister_kv_caches= fake_unregister_kv_caches,
        reregister_kv_caches = fake_reregister_kv_caches,     
        kv_caches = kv_caches,
        model=FakeModel(dummy_worker.device),
    )

    drafter = SimpleNamespace(model=FakeModel(dummy_worker.device))
    dummy_worker.model_runner.drafter = drafter
   
    dummy_worker.sleep(level=1)

    assert dummy_worker.model_runner.model.device == "cpu"
    assert dummy_worker.model_runner.drafter.model.device == "cpu"

    dummy_worker.wake_up(tags=["weights"])

    assert dummy_worker.model_runner.model.device == "npu"
    assert dummy_worker.model_runner.drafter.model.device == "npu"

def test_wake_up_dp_world_size(fake_platform, monkeypatch, dummy_worker):
    # Ensure calling sleep/wake_up on RLNPUWorker with patched platform/allocator does not raise.
    called = {"dp_all_reduce": False, "unregister_kv_caches": False, "reregister_kv_caches": False}

    def fake_all_reduce(x):
        called["dp_all_reduce"] = True

    fake_tp = SimpleNamespace(all_reduce=lambda x:None)
    fake_dp = SimpleNamespace(all_reduce=fake_all_reduce, world_size=1)

    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: fake_tp)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_dp_group", lambda: fake_dp)

    kv_config = dummy_worker.kv_cache_config
    kv_caches = [[torch.zeros((kv_config.batch_size, kv_config.num_heads, kv_config.seq_len, kv_config.head_dim), 
                dtype=torch.float32, device=dummy_worker.device)
                for _ in range(2)] for _ in range(kv_config.num_layers)]
    
    def fake_unregister_kv_caches():
        called["unregister_kv_caches"] = True

    def fake_reregister_kv_caches():
        called["reregister_kv_caches"] = True
    dummy_worker.model_runner = SimpleNamespace(   
        unregister_kv_caches = fake_unregister_kv_caches,
        reregister_kv_caches = fake_reregister_kv_caches,     
        kv_caches = kv_caches,
        model=FakeModel(dummy_worker.device),
    )

    dummy_worker.sleep(level=1)
    dummy_worker.wake_up(tags=["kv_cache"])
    assert not called["dp_all_reduce"]

    monkeypatch.setattr("vllm.distributed.parallel_state.get_dp_group", lambda: fake_dp)
    fake_dp.world_size = 2
    dummy_worker.sleep(level=1)
    dummy_worker.wake_up(tags=["kv_cache"])
    assert called["dp_all_reduce"]

def test_initialize_from_config_is_callable(fake_platform, monkeypatch, dummy_worker):
    # initialize_from_config should be callable and not raise with a simple config object
    kv_cfg = dummy_worker.kv_cache_config

    called = {"omni": False, "kv": False}

    def fake_initialize_omni_kv_cache(kv_cfg):
        called["omni"] = True
        called["kv"] = False
    def fake_initialize_kv_cache(kv_cfg):
        called["kv"] = True
        called["omni"] = False

    dummy_worker.model_runner = SimpleNamespace()
    dummy_worker.model_runner.initialize_omni_kv_cache = fake_initialize_omni_kv_cache
    dummy_worker.model_runner.initialize_kv_cache = fake_initialize_kv_cache

    # monkeypatch.setattr("model_extra_config.operator_opt_config.use_omni_cache", True)
    model_extra_config.operator_opt_config.use_omni_cache = True
    dummy_worker.initialize_from_config(kv_cfg)
    assert called["omni"] and not called["kv"]

    # monkeypatch.setattr("model_extra_config.operator_opt_config.use_omni_cache", False)
    model_extra_config.operator_opt_config.use_omni_cache = False
    dummy_worker.initialize_from_config(kv_cfg)
    assert called["kv"] and not called["omni"]


def test_load_model_is_callable(monkeypatch, dummy_worker):
    # If model_runner has a load_model that would set a flag, RLNPUWorker.load_model should be safe to call.
    called = {"load": False, "start_server": False}

    def fake_load():
        called["load"] = True

    def fake_start_server(dummy_worker):
        called["start_server"] = True

    monkeypatch.setattr("omni.adaptors.vllm.token_recovery.ha_worker_server.start_server", fake_start_server)
    dummy_worker.model_runner = SimpleNamespace(load_model=fake_load)
    dummy_worker.enable_token_recover = False
    dummy_worker.load_model()

    assert called["load"] and not called["start_server"]

    dummy_worker.enable_token_recover = True
    dummy_worker.load_model()
    
    assert called["start_server"]

def test_update_parallel_config_rl_service_mode(monkeypatch):
    monkeypatch.setenv("RL_SERVICE_MODE", "1")
    parallel_config = SimpleNamespace(worker_cls="auto")
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    ConfigUpdater._update_parallel_config(vllm_config)

    assert vllm_config.parallel_config.worker_cls == "omni.adaptors.vllm.worker.npu_worker.RLNPUWorker"

def test_update_parallel_config_auto_worker_cls(monkeypatch):
    parallel_config = SimpleNamespace(worker_cls="auto")
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    ConfigUpdater._update_parallel_config(vllm_config)

    assert vllm_config.parallel_config.worker_cls == "omni.adaptors.vllm.worker.npu_worker.NPUWorker"