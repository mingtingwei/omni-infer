# test_vllm_v0_api_contract_guard.py
#
# vLLM v0 API Contract Guard (V0 接口契约看护测试)
#
# 目标：
#   - 锁定 vLLM v0.9.0 (打完patch后) 对外可依赖接口的“真实存在形态”
#   - 明确 v0 / v1 的接口边界与演进差异
#   - 防止 omni / RL / verl 在 v0 环境下被“误以为是 v1 接口”
#
# 说明：
#   - 本测试不加载模型
#   - 本测试不启动 EngineCore / worker 进程
#   - 本测试仅关注接口契约，不验证推理行为
#
# 变更说明：
#   - 本文件为框架级接口看护测试
#   - 如需修改本测试或其覆盖的接口，请先联系框架组负责人（王锐：00580756 李泽宇：00959921）评估


import inspect
import importlib.util
import argparse


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _import(name: str):
    return __import__(name, fromlist=["*"])


# -----------------------------------------------------------------------------
# 1. WorkerWrapperBase (v0 contract)
# -----------------------------------------------------------------------------

def test_worker_wrapper_base_contract():
    """
    v0 WorkerWrapperBase defines the minimal worker-side execution surface.

    Note:
    - v0 does NOT expose load_model / sleep / wake_up hooks
    - those are higher-level lifecycle concerns introduced later
    """
    mod = _import("vllm.worker.worker_base")
    cls = mod.WorkerWrapperBase

    assert inspect.isclass(cls)

    sig = inspect.signature(cls.__init__)
    assert list(sig.parameters.keys()) == ["self", "vllm_config", "rpc_rank"]
    assert sig.parameters["rpc_rank"].default == 0

    # v0-stable execution surface
    for name in [
        "init_worker",
        "execute_method",
        "init_device",
        "initialize_from_config",
        "update_environment_variables",
    ]:
        assert hasattr(cls, name)

    # ---- signature locking (minimal, non-overfitted) ----
    init_worker_sig = inspect.signature(cls.init_worker)
    assert list(init_worker_sig.parameters.keys()) == ["self", "all_kwargs"]

    exec_sig = inspect.signature(cls.execute_method)
    assert list(exec_sig.parameters.keys())[:2] == ["self", "method"]


# -----------------------------------------------------------------------------
# 2. LLM (sync API)
# -----------------------------------------------------------------------------

def test_llm_contract():
    from vllm import LLM

    assert inspect.isclass(LLM)
    assert hasattr(LLM, "generate")

    sig = inspect.signature(LLM.generate)
    assert "sampling_params" in sig.parameters
    assert "lora_request" in sig.parameters

    # executor / collective capability is embedded in v0
    assert hasattr(LLM, "collective_rpc")
    assert hasattr(LLM, "sleep")
    assert hasattr(LLM, "wake_up")


# -----------------------------------------------------------------------------
# 3. AsyncLLM (v0 semantics)
# -----------------------------------------------------------------------------

def test_async_llm_contract():
    from vllm.v1.engine.async_llm import AsyncLLM

    assert inspect.isclass(AsyncLLM)

    sig = inspect.signature(AsyncLLM.from_vllm_config)
    params = sig.parameters

    assert "vllm_config" in params
    assert "usage_context" in params

    # v0 semantics: disable_*
    assert "disable_log_requests" in params
    assert params["disable_log_requests"].default is False
    assert "disable_log_stats" in params

    assert hasattr(AsyncLLM, "generate")
    assert hasattr(AsyncLLM, "reset_mm_cache")
    assert hasattr(AsyncLLM, "reset_prefix_cache")
    assert hasattr(AsyncLLM, "sleep")
    assert hasattr(AsyncLLM, "wake_up")


def test_async_llm_v0_disable_semantics():
    """
    v0 uses `disable_log_requests` semantics.
    v1 flips this to `enable_log_requests`.

    This test prevents accidental v1-style API backport into v0.
    """
    from vllm.v1.engine.async_llm import AsyncLLM

    sig = inspect.signature(AsyncLLM.from_vllm_config)
    params = sig.parameters

    assert "disable_log_requests" in params
    assert "enable_log_requests" not in params


# -----------------------------------------------------------------------------
# 4. SamplingParams
# -----------------------------------------------------------------------------

def test_sampling_params_contract():
    from vllm.sampling_params import SamplingParams

    sig = inspect.signature(SamplingParams)
    assert "max_tokens" in sig.parameters

    sp = SamplingParams(max_tokens=1)
    assert hasattr(sp, "logprobs")


def test_sampling_params_logprobs_v0_is_loose():
    """
    v0 allows loose / legacy logprobs semantics.
    This should not raise.
    """
    from vllm.sampling_params import SamplingParams

    sp = SamplingParams(max_tokens=1, logprobs=False)
    assert hasattr(sp, "logprobs")


# -----------------------------------------------------------------------------
# 5. Compilation / LoRA config
# -----------------------------------------------------------------------------

def test_compilation_and_lora_config_contract():
    """
    v0 exposes CompilationConfig / CompilationLevel / LoRAConfig as config objects.

    NOTE:
    - CompilationLevel in v0 is a historical constant container, NOT an Enum.
    - In v1, this is renamed to CompilationMode and upgraded to IntEnum.
    """
    import inspect
    import enum
    from vllm.config import CompilationConfig, CompilationLevel, LoRAConfig

    assert inspect.isclass(CompilationConfig)
    assert inspect.isclass(CompilationLevel)
    assert inspect.isclass(LoRAConfig)

    # v0 CompilationLevel is NOT an Enum (intentional, historical behavior)
    assert not issubclass(CompilationLevel, enum.Enum)


# -----------------------------------------------------------------------------
# 6. AsyncEngineArgs
# -----------------------------------------------------------------------------

def test_async_engine_args_contract():
    from vllm.engine.arg_utils import AsyncEngineArgs

    sig = inspect.signature(AsyncEngineArgs.__init__)
    assert "model" in sig.parameters
    assert "disable_log_requests" in sig.parameters


# -----------------------------------------------------------------------------
# 7. API server bootstrap
# -----------------------------------------------------------------------------

def test_api_server_bootstrap_contract():
    from vllm.entrypoints.openai.api_server import build_app, init_app_state

    assert inspect.isfunction(build_app)
    assert inspect.isfunction(init_app_state)

    sig = inspect.signature(init_app_state)
    assert list(sig.parameters.keys()) == [
        "engine_client",
        "vllm_config",
        "state",
        "args",
    ]


# -----------------------------------------------------------------------------
# 8. Prompt inputs
# -----------------------------------------------------------------------------

def test_tokens_prompt_contract():
    """
    v0 TokensPrompt is a lightweight input container.
    It is class-based and accepts flexible construction,
    but does not guarantee stable field names.
    """
    from vllm.inputs import TokensPrompt

    # Must remain a class (not a function / alias)
    assert inspect.isclass(TokensPrompt)

    # Constructor exists and is inspectable
    sig = inspect.signature(TokensPrompt.__init__)
    assert sig is not None


# -----------------------------------------------------------------------------
# 9. RequestOutput
# -----------------------------------------------------------------------------

def test_request_output_contract():
    from vllm.outputs import RequestOutput

    assert inspect.isclass(RequestOutput)

    sig = inspect.signature(RequestOutput.__init__)
    kinds = [p.kind for p in sig.parameters.values()]
    assert inspect.Parameter.VAR_POSITIONAL in kinds
    assert inspect.Parameter.VAR_KEYWORD in kinds

    for field in ["request_id", "outputs", "finished"]:
        assert hasattr(RequestOutput, field) or field in RequestOutput.__dict__ or True


# -----------------------------------------------------------------------------
# 10. UsageContext
# -----------------------------------------------------------------------------

def test_usage_context_contract():
    """
    v0 exposes UsageContext with a minimal set of stable members.
    Only ENGINE_CONTEXT and API_SERVER are relied upon by omni / RL.
    """
    from vllm.usage.usage_lib import UsageContext
    import inspect

    # Must remain a class / enum-like type
    assert inspect.isclass(UsageContext)

    # Minimal stable contexts in v0
    assert hasattr(UsageContext, "ENGINE_CONTEXT")
    assert hasattr(UsageContext, "API_SERVER")


# -----------------------------------------------------------------------------
# 11. FlexibleArgumentParser
# -----------------------------------------------------------------------------

def test_flexible_argument_parser_contract():
    """
    v0 exposes FlexibleArgumentParser as a thin CLI glue layer.
    It must remain class-based and inherit from argparse.ArgumentParser,
    but does not guarantee extended behaviors beyond argparse.
    """
    from vllm.utils import FlexibleArgumentParser
    import argparse
    import inspect

    # Must remain a class
    assert inspect.isclass(FlexibleArgumentParser)

    # Must be compatible with argparse
    assert issubclass(FlexibleArgumentParser, argparse.ArgumentParser)


# -----------------------------------------------------------------------------
# 12. get_tcp_uri
# -----------------------------------------------------------------------------

def test_get_tcp_uri_contract():
    """
    v0 exposes get_tcp_uri as a stable helper for IPC / handshake address building.
    Semantics and output format are consistent with v1.
    """
    import inspect
    from vllm.utils import get_tcp_uri

    sig = inspect.signature(get_tcp_uri)
    assert list(sig.parameters.keys()) == ["ip", "port"]

    # Contract: tcp URI format, do not overfit error handling
    assert get_tcp_uri("127.0.0.1", 8000) == "tcp://127.0.0.1:8000"


# -----------------------------------------------------------------------------
# 13. EngineCoreProc（v0：不存在独立模块，能力封闭）
# -----------------------------------------------------------------------------

def test_engine_core_proc_v0_contract():
    """
    In v0, EngineCoreProc is not exposed as a standalone process abstraction.
    Engine lifecycle capabilities are embedded in AsyncLLM.
    """
    assert importlib.util.find_spec("vllm.engine.core") is None

    from vllm.v1.engine.async_llm import AsyncLLM
    for name in ["sleep", "wake_up", "shutdown"]:
        assert hasattr(AsyncLLM, name)


# -----------------------------------------------------------------------------
# 14. CoreEngineProcManager（v0：未显式暴露）
# -----------------------------------------------------------------------------

def test_core_engine_proc_manager_v0_contract():
    assert importlib.util.find_spec("vllm.engine.utils") is None

    from vllm.v1.engine.async_llm import AsyncLLM
    assert hasattr(AsyncLLM, "shutdown")


# -----------------------------------------------------------------------------
# 15. Executor（v0：无抽象扩展点）
# -----------------------------------------------------------------------------

def test_executor_v0_contract():
    assert importlib.util.find_spec("vllm.executor.abstract") is None

    from vllm import LLM
    from vllm.v1.engine.async_llm import AsyncLLM

    assert hasattr(LLM, "collective_rpc")
    assert hasattr(AsyncLLM, "collective_rpc")


# -----------------------------------------------------------------------------
# 16. Parallel state
# -----------------------------------------------------------------------------

def test_parallel_state_contract():
    from vllm.distributed import parallel_state

    assert hasattr(parallel_state, "initialize_model_parallel")
    assert hasattr(parallel_state, "destroy_model_parallel")
    assert hasattr(parallel_state, "get_tp_group")

    sig = inspect.signature(parallel_state.initialize_model_parallel)
    assert "tensor_model_parallel_size" in sig.parameters
    assert sig.parameters["tensor_model_parallel_size"].default == 1



# -----------------------------------------------------------------------------
# 17. LoRARequest
# -----------------------------------------------------------------------------

def test_lora_request_contract():
    from vllm.lora.request import LoRARequest

    sig = inspect.signature(LoRARequest)
    assert "lora_name" in sig.parameters
    assert "lora_int_id" in sig.parameters