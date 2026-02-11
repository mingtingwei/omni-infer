#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# usage
# python ut_CI_fail_case_check.py --log a.log --log b.log --known /path/to/failure_case.txt
# failure_case.txt demo
# unit_tests/adaptors/vllm_adaptors/worker/st/test_npu_hybrid_chunk_prefill.py::test_hybrid_chunk_prefill_graph_mode_npu
# unit_tests/adaptors/vllm_adaptors/worker/test_npu_model_runner.py::test_init_spec_decode_and_default_gears
# unit_tests/models/pangu/st/test_pangu_moe_npu.py::test_pangu_moe_paths_on_npu
# ...

from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable


# -----------------------------
# Data structure
# -----------------------------
@dataclass
class PytestParseResult:
    """
    单个日志文件的解析结果
    """
    sessions: int
    counts: Dict[str, int]          # passed/failed/errors/skipped/warnings/xfailed/xpassed/deselected/rerun...
    failed_tests: List[str]         # nodeid list
    error_tests: List[str]          # nodeid list
    summary_lines: List[str]        # 命中的 summary 行，便于排查


# -----------------------------
# Regex & helpers
# -----------------------------
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HTML_RE = re.compile(r"<[^>]*>")

# pytest nodeid 形式：path/to/test_x.py::TestCls::test_x[param]
NODEID_RE = re.compile(r"(?P<nodeid>\S+::\S+)")
# summary 行示例：====== 32 failed, 575 passed, 7 skipped, 8 warnings in 1421.40s (0:23:41) ======
SUMMARY_LINE_RE = re.compile(r"^=+.*?\bin\b.*?=+$")

COUNT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("failed", re.compile(r"(\d+)\s+failed\b")),
    ("passed", re.compile(r"(\d+)\s+passed\b")),
    ("errors", re.compile(r"(\d+)\s+errors?\b")),
    ("skipped", re.compile(r"(\d+)\s+skipped\b")),
    ("warnings", re.compile(r"(\d+)\s+warnings?\b")),
    ("xfailed", re.compile(r"(\d+)\s+xfailed\b")),
    ("xpassed", re.compile(r"(\d+)\s+xpassed\b")),
    ("deselected", re.compile(r"(\d+)\s+deselected\b")),
    ("rerun", re.compile(r"(\d+)\s+rerun\b")),
]


def _clean_line(s: str) -> str:
    """去掉 ANSI 色彩码与 HTML 标签，做基础清洗。"""
    s = ANSI_RE.sub("", s)
    s = HTML_RE.sub("", s)
    return s.rstrip("\n")


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_counts_from_summary_line(line: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for k, pat in COUNT_PATTERNS:
        m = pat.search(line)
        if m:
            counts[k] = counts.get(k, 0) + int(m.group(1))
    return counts


def _extract_nodeid_from_failed_or_error_line(line: str) -> Optional[str]:
    """
    从 'FAILED xxx - msg' / 'ERROR xxx - msg' / 'FAILED xxx' / 'ERROR xxx' 提取 nodeid
    """
    line = line.strip()

    if line.startswith("FAILED "):
        rest = line[len("FAILED "):].strip()
    elif line.startswith("ERROR "):
        rest = line[len("ERROR "):].strip()
    else:
        return None

    # 去掉 " - " 后面的报错信息（如果有）
    rest = rest.split(" - ", 1)[0].strip()

    # nodeid 通常以 path::... 形式出现
    m = NODEID_RE.search(rest)
    if m:
        return m.group("nodeid")

    # 兜底：如果 rest 本身就像 nodeid
    if "::" in rest:
        return rest.split()[0].strip()

    return None


# -----------------------------
# Public function (you want)
# -----------------------------
def parse_pytest_log_text(text: str) -> PytestParseResult:
    """
    ✅ 核心函数：从“单个 pytest 日志文本”提取：
      ◦ sessions 数

      ◦ passed/failed/errors/... 计数（汇总该日志里的所有 summary 段）

      ◦ failed_tests 列表（nodeid）

      ◦ error_tests 列表（nodeid）


    说明：
      ◦ 优先直接抓全局的 'FAILED ...' / 'ERROR ...' 行

      ◦ 如果抓不到（某些格式），再从 'short test summary info' 区域兜底提取

      ◦ 自动去重，保留出现顺序

    """
    raw_lines = text.splitlines()
    lines = [_clean_line(x) for x in raw_lines]

    # 1) summary 计数与 session
    sessions = 0
    counts_total: Dict[str, int] = {k: 0 for k, _ in COUNT_PATTERNS}
    summary_lines: List[str] = []

    for ln in lines:
        if SUMMARY_LINE_RE.match(ln) and any(p.search(ln) for _, p in COUNT_PATTERNS):
            sessions += 1
            summary_lines.append(ln)
            c = _parse_counts_from_summary_line(ln)
            for k, v in c.items():
                counts_total[k] = counts_total.get(k, 0) + v

    # 2) 直接抓 FAILED/ERROR 行
    failed_tests: List[str] = []
    error_tests: List[str] = []

    for ln in lines:
        if ln.startswith("FAILED "):
            nid = _extract_nodeid_from_failed_or_error_line(ln)
            if nid:
                failed_tests.append(nid)
        elif ln.startswith("ERROR "):
            nid = _extract_nodeid_from_failed_or_error_line(ln)
            if nid:
                error_tests.append(nid)

    # 3) 兜底：从 short test summary info 块提取
    if not failed_tests and not error_tests:
        in_summary = False
        for ln in lines:
            if re.search(r"short test summary info", ln):
                in_summary = True
                continue
            if in_summary and SUMMARY_LINE_RE.match(ln):
                in_summary = False
                continue
            if not in_summary:
                continue

            if ln.startswith("FAILED "):
                nid = _extract_nodeid_from_failed_or_error_line(ln)
                if nid:
                    failed_tests.append(nid)
            elif ln.startswith("ERROR "):
                nid = _extract_nodeid_from_failed_or_error_line(ln)
                if nid:
                    error_tests.append(nid)

    failed_tests = _uniq_keep_order(failed_tests)
    error_tests = _uniq_keep_order(error_tests)

    return PytestParseResult(
        sessions=sessions,
        counts=counts_total,
        failed_tests=failed_tests,
        error_tests=error_tests,
        summary_lines=summary_lines,
    )


def parse_pytest_log_file(path: str) -> PytestParseResult:
    """文件版封装：读文件 -> parse_pytest_log_text"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_pytest_log_text(f.read())


# -----------------------------
# Optional: multi-log aggregation helper
# -----------------------------
def aggregate_results(results: List[PytestParseResult]) -> PytestParseResult:
    """
    将多个日志（比如多切片）聚合成一个总结果：
      ◦ sessions 求和

      ◦ counts 各字段求和

      ◦ failed_tests / error_tests 合并去重

      ◦ summary_lines 拼接

    """
    total_sessions = sum(r.sessions for r in results)

    # counts 取 union，然后求和
    all_keys = set()
    for r in results:
        all_keys.update(r.counts.keys())
    total_counts = {k: 0 for k in all_keys}
    for r in results:
        for k, v in r.counts.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)

    all_failed = _uniq_keep_order([t for r in results for t in r.failed_tests])
    all_error = _uniq_keep_order([t for r in results for t in r.error_tests])
    all_summary_lines = [ln for r in results for ln in r.summary_lines]

    return PytestParseResult(
        sessions=total_sessions,
        counts=total_counts,
        failed_tests=all_failed,
        error_tests=all_error,
        summary_lines=all_summary_lines,
    )

@dataclass
class KnownFailureCheck:
    """
    用于“本次失败(FAILED+ERROR) 是否都在已知失败集合中”的结果
    """
    ok: bool
    actual_total: int                 # 本次失败总数（fail+error 去重）
    known_total: int                  # 已知失败总数（过滤后）
    new_failures: List[str]           # 本次新增失败（不在 known）
    fixed_failures: List[str]         # 已修复（在 known 但本次没出现）
    remaining_known: List[str]        # 仍然失败的已知用例（在 known 且本次出现）


def check_known_failures(
    result: PytestParseResult,
    known_failures: Iterable[str],
    *,
    include_errors: bool = True,
    normalize: bool = True,
    require_nodeid: bool = True,
    fixed_sort: bool = True,
) -> KnownFailureCheck:
    """
    ✅ 单独的函数：对比“本次失败集合”与“已知失败集合”，判断是否出现新失败。

    参数：
      ◦ result: parse_pytest_log_* / aggregate_results 的返回

      ◦ known_failures: 已知失败列表/集合（可来自文件、配置、硬编码）

      ◦ include_errors: True 表示把 error_tests 也算进失败集合（你要的就是 True）

      ◦ normalize: True 会对 known 条目做 strip，并可选做进一步规范化（目前是基础 strip）

      ◦ require_nodeid: True 会过滤 known_failures 中不包含 '::' 的条目

        （避免你同事 bash 里那种 "[ 2%]" 这种噪音导致 fixed_failures 很多）
      ◦ fixed_sort: True 则 fixed_failures 稳定按字典序输出（便于 CI diff）


    返回：
      KnownFailureCheck(ok, new/fixed/remaining 等)

    判定逻辑：
      actual = unique(failed_tests + error_tests(if include_errors))
      new = actual - known
      fixed = known - actual
      remaining = actual ∩ known
      ok = (len(new) == 0)
    """
    # 1) 组装本次实际失败集合（fail + error）
    actual_list = list(result.failed_tests)
    if include_errors:
        actual_list += list(result.error_tests)
    actual_list = _uniq_keep_order(actual_list)
    actual_set = set(actual_list)

    # 2) 处理 known_failures
    known_list = [str(x) for x in known_failures if str(x)]
    if normalize:
        known_list = [x.strip() for x in known_list if x.strip()]
    if require_nodeid:
        known_list = [x for x in known_list if "::" in x]

    known_set = set(known_list)

    # 3) 计算 diff
    new_failures = [x for x in actual_list if x not in known_set]
    remaining_known = [x for x in actual_list if x in known_set]
    fixed_failures = [x for x in known_set if x not in actual_set]
    if fixed_sort:
        fixed_failures = sorted(fixed_failures)

    return KnownFailureCheck(
        ok=(len(new_failures) == 0),
        actual_total=len(actual_list),
        known_total=len(known_set),
        new_failures=new_failures,
        fixed_failures=fixed_failures,
        remaining_known=remaining_known,
    )


def load_known_failures_txt(path: str, *, require_nodeid: bool = True) -> List[str]:
    """
    从 txt 文件读取已知失败用例列表：
      ◦ 支持空行、支持 # 注释

      ◦ 默认过滤掉不包含 '::' 的行（避免噪音）

    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # 支持行尾注释：xxx::yyy  # comment
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if require_nodeid and "::" not in line:
                continue
            out.append(line)
    return _uniq_keep_order(out)


# -----------------------------
# Example usage (safe to delete)
# -----------------------------
if __name__ == "__main__":
    # 例子：你多切片日志路径列表（自己替换）
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="pytest log file path (repeatable), e.g. --log a.log --log b.log",
    )
    parser.add_argument(
        "--known",
        required=True,
        help="known failures txt path, one nodeid per line",
    )
    args = parser.parse_args()

    log_paths = args.log

    results = [parse_pytest_log_file(p) for p in log_paths]
    merged = aggregate_results(results)

    print("sessions:", merged.sessions)
    for k in ["passed", "failed", "errors", "skipped", "warnings", "xfailed", "xpassed", "rerun"]:
        if k in merged.counts:
            print(f"{k}: {merged.counts.get(k, 0)}")
    print("failed_tests:", len(merged.failed_tests))
    print("error_tests :", len(merged.error_tests))
    print("failed_tests:", merged.failed_tests)
    print("error_tests :", merged.error_tests)

    known = load_known_failures_txt(args.known, require_nodeid=True)

    
    check = check_known_failures(merged, known, include_errors=True)
    print("ok:", check.ok)
    print("new_failures:", check.new_failures)
    print("fixed_failures:", check.fixed_failures)
    print("remaining_known:", check.remaining_known)