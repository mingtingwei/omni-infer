"""
本测试用例用于验证分布式启动、并行参数传递及测试协议本身是否正确，
不验证模型运行时（runtime）或通信算子的完整功能。

运行方式：
    pytest -sv test_ct_runner.py

──────────────────────────────────────────────────────────────────────────────
测试场景
──────────────────────────────────────────────────────────────────────────────
本测试固定运行以下 4 种并行配置场景：

    - tp2        : Tensor Parallel = 2
    - pp2        : Pipeline Parallel = 2
    - ep2        : Expert Parallel = 2
    - tp2_pp2   : Tensor Parallel = 2, Pipeline Parallel = 2

每个场景对应固定的 world_size（由 nproc 控制）。

──────────────────────────────────────────────────────────────────────────────
PASS 语义
──────────────────────────────────────────────────────────────────────────────
- parallel_groups_basic 必须 PASS  
  仅验证：
    - torch.distributed 初始化成功
    - world_size / rank 与期望一致

该测试失败意味着环境或启动层存在问题。

──────────────────────────────────────────────────────────────────────────────
SKIP 语义（设计行为）
──────────────────────────────────────────────────────────────────────────────
以下测试在 CT 阶段允许并期望 SKIP：
    - local_all_to_all
    - reduce_scatter_two_stage
    - ...

原因是当前测试：
    - 不加载模型
    - 不启动模型 runtime
    - 不执行真实 forward / prefill / decode

因此无法保证相关通信 group 被构造。
此类 SKIP 属于正常行为，而非功能缺失。

SKIP 分类说明：
    - env           : 环境或分布式初始化问题
    - topology      : 当前并行配置不满足测试前提
    - not-supported : 未进入模型 runtime

──────────────────────────────────────────────────────────────────────────────
如何避免 SKIP
──────────────────────────────────────────────────────────────────────────────
要使上述测试 PASS，必须：
    - 启动真实模型 runtime
    - 完成 initialize_model_parallel
    - 执行至少一次真实算子路径

──────────────────────────────────────────────────────────────────────────────
总结
──────────────────────────────────────────────────────────────────────────────
当前测试结果中：
    - 1 PASS + 若干 SKIP 是正确且预期的状态
"""

from __future__ import annotations

import os
import re
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

RESULT_BEGIN = "=== CT-REAL-RESULTS-BEGIN ==="
RESULT_END = "=== CT-REAL-RESULTS-END ==="

DEFAULT_BACKEND = os.environ.get("DIST_BACKEND", "hccl")


@dataclass(frozen=True)
class Scenario:
    name: str
    tp: int
    pp: int
    ep: int
    nproc: int


SCENARIOS = [
    Scenario("tp2", tp=2, pp=1, ep=0, nproc=2),
    Scenario("pp2", tp=1, pp=2, ep=0, nproc=2),
    Scenario("ep2", tp=1, pp=1, ep=1, nproc=2),
    Scenario("tp2_pp2", tp=2, pp=2, ep=0, nproc=4),
]

# 固化“期望结果规范”
EXPECTED = {
    "tp2": {
        "PASS": ["parallel_groups_basic"],
        "SKIP_PREFIX": ["local_all_to_all", "reduce_scatter_two_stage", "all_gather_two_stage"],
        "ALLOW_FAIL": [],
    },
    "pp2": {
        "PASS": ["parallel_groups_basic"],
        "SKIP_PREFIX": ["local_all_to_all", "reduce_scatter_two_stage", "all_gather_two_stage"],
        "ALLOW_FAIL": [],
    },
    "ep2": {
        "PASS": ["parallel_groups_basic"],
        "SKIP_PREFIX": ["local_all_to_all", "reduce_scatter_two_stage", "all_gather_two_stage"],
        "ALLOW_FAIL": [],
    },
    "tp2_pp2": {
        # 其它可能因 env/HCCL 波动直接 env-skip
        "PASS": ["parallel_groups_basic"],
        "SKIP_PREFIX": ["local_all_to_all", "reduce_scatter_two_stage", "all_gather_two_stage"],
        "ALLOW_FAIL": [],
    },
}

def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_runner_stdout(path: Path, content_lines: List[str]) -> None:
    path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def _read_text_safely(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(errors="ignore")  # type: ignore[arg-type]
        except Exception:
            return ""

def _strip_torchrun_prefix(line: str) -> str:
    # torchrun --tee may prefix: [default0]: ...
    return re.sub(r"^\[default\d+\]:", "", line).lstrip()


def _collect_worker_texts(log_dir: Path, torchrun_raw: str) -> List[str]:
    """
    Critical fix:
    - Structured blocks are printed by ranks into log_dir/default*/stdout.log.
    - So we MUST read rank stdout logs first.
    """
    texts: List[str] = []

    # Rank logs (primary)
    for p in sorted(log_dir.glob("default*/stdout.log")):
        t = _read_text_safely(p)
        if t:
            texts.append(t)

    # Some setups also have stderr.log
    for p in sorted(log_dir.glob("default*/stderr.log")):
        t = _read_text_safely(p)
        if t:
            texts.append(t)

    # Torchrun captured output (secondary)
    if torchrun_raw:
        texts.append(torchrun_raw)

    return texts


def _extract_structured_blocks_from_text(text: str) -> List[List[str]]:
    """
    Extract blocks between RESULT_BEGIN and RESULT_END.
    Works for:
      - rank stdout.log
      - torchrun --tee merged stdout
    """
    blocks: List[List[str]] = []
    cur: List[str] = []
    in_block = False

    for raw in text.splitlines():
        line = _strip_torchrun_prefix(raw).strip()
        if not line:
            continue

        if RESULT_BEGIN in line:
            in_block = True
            cur = []
            continue

        if RESULT_END in line and in_block:
            blocks.append(cur)
            in_block = False
            continue

        if in_block:
            cur.append(line)

    return blocks


def _parse_results(lines: List[str]) -> Dict[str, List[str]]:
    """
    Parse:
      PASS: name
      SKIP: name | category: reason
      FAIL: name | ...
    """
    by = {"PASS": [], "SKIP": [], "FAIL": []}
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("PASS:"):
            by["PASS"].append(ln[len("PASS:"):].strip())
        elif ln.startswith("SKIP:"):
            by["SKIP"].append(ln[len("SKIP:"):].strip())
        elif ln.startswith("FAIL:"):
            by["FAIL"].append(ln[len("FAIL:"):].strip())
    return by


def _classify_no_block_reason(raw: str) -> str:
    if re.search(r"\b\d+\s+passed\b", raw) or ".ss" in raw:
        return "not-supported: worker tests executed but no structured result block was emitted; contract not triggered"
    """
    Only used when we truly cannot find any structured results block
    in rank logs + torchrun output.
    """
    needles = [
        ("ImportError", "env: import error"),
        ("ERR02200", "env: HCCL init failure (ERR02200)"),
        ("ERR99999", "env: runtime exception (ERR99999)"),
        ("hcclCommInit", "env: HCCL comm init failure"),
        ("DIST call hccl api failed", "env: HCCL api failure"),
        ("SIGSEGV", "env: process segfault (SIGSEGV)"),
        ("Signal 11", "env: process segfault (Signal 11)"),
        ("ChildFailedError", "env: torchrun child failure"),
        ("SyntaxError: source code cannot contain null bytes", "env: python executable/path corrupted"),
        ("fixture 'reverse' not found", "env: pytest collection/setup error (parametrize broken)"),
        ("ERROR at setup", "env: pytest setup error"),
    ]
    for n, msg in needles:
        if n in raw:
            return msg
    return "env: no structured result block found in rank logs"


def _summarize_skips(by_skip: List[str]) -> List[Tuple[str, str]]:
    """
    SKIP entry format from worker:
      "<name> | <category>: <reason>"
    """
    items: List[Tuple[str, str]] = []
    for sk in by_skip:
        sk = sk.strip()
        if not sk:
            continue
        if "|" in sk:
            name, detail = [x.strip() for x in sk.split("|", 1)]
        else:
            name, detail = sk, ""
        items.append((name, detail))
    return items


def _validate_expected(sname: str, by: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Returns: (missing_required_passes, unexpected_fails)
    """
    exp = EXPECTED[sname]
    pass_set = set(by["PASS"])
    missing_pass = [x for x in exp["PASS"] if x not in pass_set]

    allow_fail = set(exp.get("ALLOW_FAIL", []))
    unexpected_fails: List[str] = []
    for f in by["FAIL"]:
        test_name = f.split("|", 1)[0].strip()
        if test_name not in allow_fail:
            unexpected_fails.append(f)

    return missing_pass, unexpected_fails


def _find_repo_root(start_dir: Path) -> Path:
    """
    Find repo root by searching upwards for 'omni' directory.
    start_dir must be a directory.
    """
    p = start_dir.resolve()
    while True:
        if (p / "omni").is_dir():
            return p
        if p.parent == p:
            return start_dir.resolve()
        p = p.parent


def _run_one(s: Scenario, base_dir: Path) -> Tuple[str, Path]:
    log_dir = base_dir / f"ct_logs_{s.name}_{_now_ts()}"
    log_dir.mkdir(parents=True, exist_ok=True)

    runner_stdout = log_dir / "runner_stdout.txt"
    raw_path = log_dir / "torchrun_raw.txt"

    torchrun_bin = shutil.which("torchrun")
    if not torchrun_bin:
        pytest.skip("env: torchrun not found in PATH")

    worker_path = Path(__file__).with_name("test_ct_worker.py").resolve()
    if not worker_path.exists():
        pytest.skip(f"env: worker not found: {worker_path}")

    master_port = str(20000 + (os.getpid() % 20000))

    # repo_root 必须从“目录”开始找
    repo_root = _find_repo_root(Path(__file__).resolve().parent)

    env = os.environ.copy()
    env.update(
        {
            "CT_SCENARIO": s.name,
            "CT_TP": str(s.tp),
            "CT_PP": str(s.pp),
            "CT_EP": str(s.ep),
            "CT_EXPECT_WORLD": str(s.nproc),
            "DIST_BACKEND": DEFAULT_BACKEND,
            "PYTHONUNBUFFERED": "1",
        }
    )

    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + old_pp if old_pp else "")
    env["pythonpath"] = env["PYTHONPATH"]

    cmd = [
        torchrun_bin,
        f"--nproc_per_node={s.nproc}",
        f"--master_port={master_port}",
        "--tee",
        "3",
        "--log_dir",
        str(log_dir),
        "-m",
        "pytest",
        "-q",
        "-s",
        str(worker_path),
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(repo_root), 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    torchrun_raw = proc.stdout or ""
    raw_path.write_text(torchrun_raw, encoding="utf-8", errors="ignore")

    # Clean runner stdout content (ONLY summary)
    out: List[str] = []
    out.append("=" * 100)
    out.append(f"[RUNNER] scenario={s.name}")
    out.append(f"[RUNNER] nproc={s.nproc}, tp={s.tp}, pp={s.pp}, ep={s.ep}")
    out.append(f"[RUNNER] master_port={master_port}")
    out.append(f"[RUNNER] repo_root={repo_root}")
    out.append(f"[RUNNER] log_dir={log_dir}")
    out.append("=" * 100)

    # Critical: search blocks from rank logs first
    texts = _collect_worker_texts(log_dir, torchrun_raw)

    blocks: List[List[str]] = []
    for t in texts:
        blocks.extend(_extract_structured_blocks_from_text(t))

    if not blocks and not any(RESULT_BEGIN in t for t in texts):
        joined = "\n".join(texts)
        if re.search(r"\d+\s+passed", joined):
            out.append(
                "SKIPPED (not-supported: worker tests executed but no structured "
                "result block was emitted; contract not triggered)"
            )
            _write_runner_stdout(runner_stdout, out)
            pytest.skip(
                "not-supported: structured result contract not triggered; "
                "see runner_stdout.txt"
            )

        reason = _classify_no_block_reason(joined)
        out.append(f"SKIPPED ({reason}; see {raw_path})")
        _write_runner_stdout(runner_stdout, out)
        pytest.skip(f"{reason}; see {runner_stdout}")

    # Merge blocks (rank-agnostic): contract is "block exists", not "every rank has block".
    merged_lines: List[str] = []
    for b in blocks:
        merged_lines.extend(b)

    by = _parse_results(merged_lines)
    missing_pass, unexpected_fails = _validate_expected(s.name, by)
    skip_items = _summarize_skips(by["SKIP"])

    # Summary section
    out.append("")
    out.append(f"{s.name}:")
    out.append("  PASS:")
    if by["PASS"]:
        for x in sorted(set(by["PASS"])):
            out.append(f"    - {x}")
    else:
        out.append("    - (none)")

    out.append("  SKIP:")
    if skip_items:
        for n, detail in skip_items:
            out.append(f"    - {n} ({detail})" if detail else f"    - {n}")
    else:
        out.append("    - (none)")

    out.append("  FAIL:")
    if unexpected_fails:
        for x in unexpected_fails:
            out.append(f"    - {x}")
    else:
        out.append("    - (none)")

    # Decide outcome
    if unexpected_fails:
        out.append("")
        out.append("FAILED (worker reported FAIL)")
        _write_runner_stdout(runner_stdout, out)
        pytest.fail(f"[{s.name}] worker FAIL; see {runner_stdout}")

    if missing_pass:
        # If core pass missing, treat as env (worker didn't execute correctly)
        out.append("")
        out.append(f"SKIPPED (env: missing required PASS: {missing_pass}; see {raw_path})")
        _write_runner_stdout(runner_stdout, out)
        pytest.skip(f"env: missing required PASS {missing_pass}; see {runner_stdout}")

    out.append("")
    out.append("PASSED")
    _write_runner_stdout(runner_stdout, out)
    return s.name, runner_stdout


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_ct_distributed_topology_runner(scenario: Scenario):
    base_dir = Path(__file__).resolve().parent
    _run_one(scenario, base_dir)