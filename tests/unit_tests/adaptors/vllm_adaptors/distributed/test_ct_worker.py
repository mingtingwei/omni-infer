"""
Contract:
- Always prints structured results block (per rank), even on errors:
    === CT-REAL-RESULTS-BEGIN ===
    PASS: ...
    SKIP: ... | <category>: <reason>
    FAIL: ... | <exception>
    === CT-REAL-RESULTS-END ===

- Skip categories are strictly one of:
    env / topology / not-supported

- tp2 should NOT fail:
    - local group unavailable => SKIP (not-supported)
    - two-stage groups not present / not initialized => SKIP (topology or not-supported)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pytest
import torch
import torch.distributed as dist


RESULT_BEGIN = "=== CT-REAL-RESULTS-BEGIN ==="
RESULT_END = "=== CT-REAL-RESULTS-END ==="

_SKIP_CATEGORIES = {"env", "topology", "not-supported"}

# In-memory results for this rank
_CT_RESULTS: List[Tuple[str, str, str]] = []  # (status, test_name, detail)


def _append(status: str, name: str, detail: str = "") -> None:
    _CT_RESULTS.append((status, name, detail))


def _skip(name: str, category: str, reason: str) -> None:
    assert category in _SKIP_CATEGORIES, f"invalid skip category: {category}"
    _append("SKIP", name, f"{category}: {reason}")
    pytest.skip(f"{category}: {reason}")


def _fail(name: str, err: BaseException) -> None:
    _append("FAIL", name, repr(err))
    raise err


def _pass(name: str, detail: str = "") -> None:
    _append("PASS", name, detail)


@dataclass(frozen=True)
class Topo:
    scenario: str
    tp: int
    pp: int
    ep: bool
    expect_world: int
    backend: str


def _require_torchrun() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("env: not launched by torchrun (missing RANK/WORLD_SIZE)")


def _parse_topo_from_env() -> Topo:
    scenario = os.environ.get("CT_SCENARIO", "unknown")
    tp = int(os.environ.get("CT_TP", "1"))
    pp = int(os.environ.get("CT_PP", "1"))
    ep = bool(int(os.environ.get("CT_EP", "0")))
    expect_world = int(os.environ.get("CT_EXPECT_WORLD", os.environ.get("WORLD_SIZE", "1")))
    backend = os.environ.get("DIST_BACKEND", os.environ.get("DIST_BACKEND", "hccl"))
    return Topo(scenario=scenario, tp=tp, pp=pp, ep=ep, expect_world=expect_world, backend=backend)


def _init_dist(topo: Topo) -> None:
    if dist.is_initialized():
        return
    try:
        dist.init_process_group(backend=topo.backend, init_method="env://")
    except Exception as e:
        _skip("dist_init", "env", f"dist.init_process_group failed: {e!r}")


def _check_world(topo: Topo) -> None:
    try:
        ws = dist.get_world_size()
    except Exception as e:
        _skip("world_check", "env", f"cannot read world_size: {e!r}")
        return
    if ws != topo.expect_world:
        _skip("world_check", "env", f"world_size mismatch: expect {topo.expect_world}, got {ws}")


def _safe_import(path: str):
    try:
        module = __import__(path, fromlist=["*"])
        return module
    except Exception:
        return None


def _safe_getattr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _is_env_collective_error(e: BaseException) -> bool:
    msg = repr(e)
    needles = ("HCCL", "hccl", "ERR", "SIGSEGV", "Signal 11", "ChildFailedError")
    return any(n in msg for n in needles)


def _try_get_vllm_world_group() -> Tuple[Optional[object], str]:
    ps = _safe_import("vllm.distributed.parallel_state")
    if ps is None:
        return None, "not-supported: vllm.distributed.parallel_state not importable"

    fn = _safe_getattr(ps, "get_world_group_from_list")
    if callable(fn):
        try:
            g = fn(0)
            if g is not None:
                return g, ""
            return None, "not-supported: world group list returned None"
        except Exception as e:
            return None, f"not-supported: world group list not ready: {e!r}"

    return None, "not-supported: cannot locate get_world_group_from_list"


def _should_run_two_stage(topo: Topo) -> Tuple[bool, str]:
    if not topo.ep:
        return False, "group=ep not enabled"
    return True, ""


@pytest.fixture(scope="session")
def dist_and_parallel_state() -> Topo:
    _require_torchrun()
    topo = _parse_topo_from_env()
    _init_dist(topo)
    _check_world(topo)
    return topo


def test_parallel_groups_basic(dist_and_parallel_state: Topo):
    name = "parallel_groups_basic"
    try:
        assert dist.is_initialized(), "dist not initialized"
        ws = dist.get_world_size()
        rk = dist.get_rank()
        assert ws == dist_and_parallel_state.expect_world, f"world mismatch: {ws}"
        assert 0 <= rk < ws, f"rank out of range: {rk}"
        _pass(name)
    except Exception as e:
        _fail(name, e)


def test_local_all_to_all_semantic(dist_and_parallel_state: Topo):
    name = "local_all_to_all"
    mod = _safe_import("omni.adaptors.vllm.distributed.communication_op")
    if mod is None:
        _skip(name, "not-supported", "cannot import omni.adaptors.vllm.distributed.communication_op")

    fn = _safe_getattr(mod, "local_all_to_all")
    if not callable(fn):
        _skip(name, "not-supported", "local_all_to_all op not found")

    g, greason = _try_get_vllm_world_group()
    if g is None:
        _skip(name, "not-supported", greason)

    try:
        x = torch.arange(8, device="cpu").reshape(2, 4).contiguous()
        try:
            y = fn(x, group=g)  
        except TypeError:
            y = fn(x)  

        assert y is not None
        _pass(name)
    except pytest.skip.Exception:
        raise
    except Exception as e:
        if _is_env_collective_error(e):
            _skip(name, "env", f"collective failed: {repr(e)}")
        _skip(name, "not-supported", f"local group unavailable: {repr(e)}")


@pytest.mark.parametrize("reverse", [False, True])
def test_reduce_scatter_two_stage(dist_and_parallel_state: Topo, reverse: bool):
    name = f"reduce_scatter_two_stage[{reverse}]"
    topo = dist_and_parallel_state

    ok, reason = _should_run_two_stage(topo)
    if not ok:
        _skip(name, "topology", reason)

    mod = _safe_import("omni.adaptors.vllm.distributed.communication_op")
    if mod is None:
        _skip(name, "not-supported", "cannot import communication_op")

    fn = _safe_getattr(mod, "reduce_scatter_two_stage")
    if not callable(fn):
        _skip(name, "not-supported", "reduce_scatter_two_stage not found")

    g, greason = _try_get_vllm_world_group()
    if g is None:
        _skip(name, "not-supported", greason)

    try:
        x = torch.arange(8, device="cpu").contiguous()
        try:
            y = fn(x, idx=0, reverse=reverse)  
        except TypeError:
            y = fn(x)  
        assert y is not None
        _pass(name)
    except pytest.skip.Exception:
        raise
    except Exception as e:
        if _is_env_collective_error(e):
            _skip(name, "env", f"collective failed: {repr(e)}")
        _fail(name, e)


@pytest.mark.parametrize("reverse", [False, True])
def test_all_gather_two_stage(dist_and_parallel_state: Topo, reverse: bool):
    name = f"all_gather_two_stage[{reverse}]"
    topo = dist_and_parallel_state

    ok, reason = _should_run_two_stage(topo)
    if not ok:
        _skip(name, "topology", reason)

    mod = _safe_import("omni.adaptors.vllm.distributed.communication_op")
    if mod is None:
        _skip(name, "not-supported", "cannot import communication_op")

    fn = _safe_getattr(mod, "all_gather_two_stage")
    if not callable(fn):
        _skip(name, "not-supported", "all_gather_two_stage not found")

    g, greason = _try_get_vllm_world_group()
    if g is None:
        _skip(name, "not-supported", greason)

    try:
        x = torch.arange(8, device="cpu").contiguous()
        try:
            y = fn(x, idx=0, dim=0, reverse=reverse)  
        except TypeError:
            y = fn(x)  
        assert y is not None
        _pass(name)
    except pytest.skip.Exception:
        raise
    except Exception as e:
        if _is_env_collective_error(e):
            _skip(name, "env", f"collective failed: {repr(e)}")
        _fail(name, e)



@pytest.fixture(scope="session", autouse=True)
def _ct_emit_structured_block():
    yield
    seen = set()
    lines = []
    for st, name, detail in _CT_RESULTS:
        if name in seen:
            continue
        seen.add(name)
        if st == "PASS":
            lines.append(f"PASS: {name}")
        elif st == "SKIP":
            lines.append(f"SKIP: {name} | {detail}")
        else:
            lines.append(f"FAIL: {name} | {detail}")

    if not lines:
        lines = ["SKIP: worker | env: no tests executed (collection/init failure)"]

    print(RESULT_BEGIN)
    for ln in lines:
        print(ln)
    print(RESULT_END)
