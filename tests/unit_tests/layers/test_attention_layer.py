import importlib
import sys
from pathlib import Path
import types
from types import SimpleNamespace

import pytest
import torch


def _mkpkg(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # mark as package so dotted imports work
    return m


def _mkmod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture()
def attention_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root))  # avoid sys.path pollution

    # Force re-import with our stubs (so module-level imports are rebound).
    sys.modules.pop("omni.layers.attention.layer", None)

    # -----------------------------
    # Build a minimal fake vllm tree
    # -----------------------------
    vllm_pkg = _mkpkg("vllm")

    # vllm.envs constants
    envs_mod = _mkmod(
        "vllm.envs",
        Q_SCALE_CONSTANT=2.0,
        K_SCALE_CONSTANT=3.0,
        V_SCALE_CONSTANT=4.0,
    )

    # vllm.logger
    logger_mod = _mkmod("vllm.logger", logger=None)

    # vllm.attention (+ selector + layer)
    attention_pkg = _mkpkg("vllm.attention")
    attention_pkg.AttentionType = SimpleNamespace(DECODER="decoder")

    class FakeImpl:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeBackend:
        accept_output_buffer = True

        def __init__(self, name="fake_backend"):
            self._name = name

        def get_impl_cls(self):
            return FakeImpl

        def get_name(self):
            return self._name

    def backend_name_to_enum(name):
        return f"enum:{name}"

    def get_attn_backend(*args, **kwargs):
        return FakeBackend()

    selector_mod = _mkmod(
        "vllm.attention.selector",
        backend_name_to_enum=backend_name_to_enum,
        get_attn_backend=get_attn_backend,
    )

    class AttentionBase:
        def __init__(self, *args, **kwargs):
            pass

    attn_layer_mod = _mkmod("vllm.attention.layer", Attention=AttentionBase)

    # vllm.config
    class CacheConfig:
        def __init__(
            self,
            cache_dtype="auto",
            block_size=16,
            is_attention_free=False,
            sliding_window=None,
            calculate_kv_scales=False,
        ):
            self.cache_dtype = cache_dtype
            self.block_size = block_size
            self.is_attention_free = is_attention_free
            self.sliding_window = sliding_window
            self.calculate_kv_scales = calculate_kv_scales

    shared_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )

    config_mod = _mkmod(
        "vllm.config",
        CacheConfig=CacheConfig,
        get_current_vllm_config=lambda: shared_config,
    )

    # vllm.model_executor.layers.linear
    class UnquantizedLinearMethod:
        pass

    # vllm.model_executor.layers.quantization.base_config
    class QuantizationConfig:
        def __init__(self, quant_method=None):
            self._quant_method = quant_method

        def get_quant_method(self, *_, **__):
            return self._quant_method

    model_executor_pkg = _mkpkg("vllm.model_executor")
    layers_pkg = _mkpkg("vllm.model_executor.layers")
    quant_pkg = _mkpkg("vllm.model_executor.layers.quantization")

    linear_mod = _mkmod(
        "vllm.model_executor.layers.linear",
        UnquantizedLinearMethod=UnquantizedLinearMethod,
    )

    base_config_mod = _mkmod(
        "vllm.model_executor.layers.quantization.base_config",
        QuantizationConfig=QuantizationConfig,
    )

    # vllm.platforms
    class BackendHelper:
        device_type = "cpu"

        def is_cuda_alike(self):
            return False

        def is_cpu(self):
            return False

    platforms_mod = _mkmod(
        "vllm.platforms",
        _Backend=object,
        current_platform=BackendHelper(),
    )

    # -----------------------------
    # Wire parent->child attributes
    # (IMPORTANT for `import vllm.envs as envs`)
    # -----------------------------
    vllm_pkg.envs = envs_mod
    vllm_pkg.logger = logger_mod
    vllm_pkg.attention = attention_pkg
    vllm_pkg.config = config_mod
    vllm_pkg.model_executor = model_executor_pkg
    vllm_pkg.platforms = platforms_mod

    attention_pkg.selector = selector_mod
    attention_pkg.layer = attn_layer_mod

    model_executor_pkg.layers = layers_pkg
    layers_pkg.linear = linear_mod
    layers_pkg.quantization = quant_pkg
    quant_pkg.base_config = base_config_mod

    # -----------------------------
    # Register in sys.modules
    # -----------------------------
    for m in [
        vllm_pkg,
        envs_mod,
        logger_mod,
        attention_pkg,
        selector_mod,
        attn_layer_mod,
        config_mod,
        model_executor_pkg,
        layers_pkg,
        linear_mod,
        quant_pkg,
        base_config_mod,
        platforms_mod,
    ]:
        monkeypatch.setitem(sys.modules, m.__name__, m)

    # Stub adaptor chain to bypass heavy imports
    adaptor_utils = SimpleNamespace(get_attr_by_names=lambda obj, *names: None)
    monkeypatch.setitem(sys.modules, "omni.adaptors.vllm.utils", adaptor_utils)
    monkeypatch.setitem(
        sys.modules,
        "omni.adaptors.vllm.patches.pangu_patch",
        SimpleNamespace(patch_pangu=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "omni.adaptors.vllm.patches.model_patch",
        SimpleNamespace(model_patch=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "omni.adaptors.vllm.patches",
        SimpleNamespace(model_patch=None),
    )

    # Import module under test
    module = importlib.import_module("omni.layers.attention.layer")

    # Force the module under test to use our env constants at runtime.
    monkeypatch.setattr(module, "envs", envs_mod, raising=False)

    # -----------------------------
    # Extra stubs exposed for tests
    # -----------------------------
    class QuantMethod:
        def __init__(self):
            self.called_with = None

        def create_weights(self, module_obj, total_num_kv_heads, head_size):
            self.called_with = (module_obj, total_num_kv_heads, head_size)

    # Expose to tests exactly as they expect
    module.FakeImpl = FakeImpl
    module.QuantMethod = QuantMethod
    module.QuantizationConfig = QuantizationConfig
    module.UnquantizedLinearMethod = UnquantizedLinearMethod

    return module


@pytest.fixture(autouse=True)
def clear_forward_context(attention_module):
    cfg = attention_module.get_current_vllm_config()
    cfg.compilation_config.static_forward_context.clear()
    yield
    cfg.compilation_config.static_forward_context.clear()


def test_attention_init_c8_sliding_window_and_defaults(attention_module):
    cache_config = attention_module.CacheConfig(sliding_window=8, cache_dtype="fp8")
    module_cfg = attention_module.get_current_vllm_config()
    layer = attention_module.Attention()

    attention_module.attention_init_c8(
        layer,
        num_heads=4,
        head_size=16,
        scale=1.0,
        cache_config=cache_config,
    )

    assert layer.sliding_window == 8
    assert layer.kv_cache_dtype == "fp8"
    assert layer.k_range.item() == 3.0
    assert len(layer.kv_cache) == module_cfg.parallel_config.pipeline_parallel_size


def test_attention_init_c8_per_layer_window_overrides(attention_module):
    cache_config = attention_module.CacheConfig(sliding_window=4)
    layer = attention_module.Attention()
    attention_module.attention_init_c8(
        layer,
        num_heads=2,
        head_size=8,
        scale=1.0,
        cache_config=cache_config,
        per_layer_sliding_window=10,
    )
    assert layer.sliding_window == 10


def test_attention_init_c8_quant_method_invocation(attention_module):
    quant_method = attention_module.QuantMethod()
    quant_config = attention_module.QuantizationConfig(quant_method=quant_method)
    layer = attention_module.Attention()

    attention_module.attention_init_c8(
        layer,
        num_heads=1,
        head_size=4,
        scale=1.0,
        quant_config=quant_config,
        total_num_kv_heads=5,
    )

    assert quant_method.called_with[0] is layer
    assert quant_method.called_with[1] == 5
    assert quant_method.called_with[2] == 4


def test_attention_init_c8_backend_and_flags(attention_module):
    layer = attention_module.Attention()
    attention_module.attention_init_c8(
        layer,
        num_heads=1,
        head_size=4,
        scale=1.0,
        attn_type=attention_module.AttentionType.DECODER,
    )
    assert layer.impl.__class__.__name__ == "FakeImpl"
    assert layer.backend == "enum:fake_backend"
    assert layer.use_output is True
    assert layer.use_direct_call is True


def test_attention_init_c8_raises_on_duplicate_prefix(attention_module):
    cfg = attention_module.get_current_vllm_config()
    cfg.compilation_config.static_forward_context["dup"] = object()
    layer = attention_module.Attention()
    with pytest.raises(ValueError):
        attention_module.attention_init_c8(
            layer,
            num_heads=1,
            head_size=4,
            scale=1.0,
            prefix="dup",
        )
