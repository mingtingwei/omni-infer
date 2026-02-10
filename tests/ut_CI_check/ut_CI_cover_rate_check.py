#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# usage
# python3 ut_CI_cover_rate_check.py \
#   --file vllm=/path/to/coverage/vllm_report/coverage_vllm.xml \
#   --file omni=/path/to/coverage/omni_report/coverage_omni.xml \
#   --file proxy=/path/to/coverage/proxy_report/coverage_report.xml \
#   --file patch=/path/to/coverage/patch_report/patch_coverage.html \
#   --print \ # 可选 是否打印结果
#   --json /path/to/coverage_rates.json # 可选 是否保存json文件


from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List


# -----------------------------
# Regex patterns
# -----------------------------
# 兼容 Cobertura / coverage.py xml 里常见的 line-rate="0.123"
LINE_RATE_RE = re.compile(r'\bline-rate\s*=\s*"([^"]+)"', re.IGNORECASE)

# patch html 中常见 Coverage</b>: 84% 或 Coverage: 84%
COVERAGE_PERCENT_RE = re.compile(
    r"Coverage(?:</b>)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)

# 兜底：任意百分号（谨慎）
ANY_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _to_rate_0_1(x: float) -> float:
    """把 0~100 的百分数自动归一化到 0~1；0~1 的保持不变。"""
    if x > 1.0:
        return x / 100.0
    return x


def parse_coverage_rate_from_xml_text(text: str) -> Optional[float]:
    """
    从 xml 文本中解析 line-rate="..."
    返回 0~1 float；解析不到返回 None
    """
    m = LINE_RATE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_coverage_rate_from_html_text(text: str) -> Optional[float]:
    """
    从 html 文本中解析 Coverage: xx%
    返回 0~1 float；解析不到返回 None
    """
    m = COVERAGE_PERCENT_RE.search(text)
    if m:
        try:
            return _to_rate_0_1(float(m.group(1)))
        except ValueError:
            return None

    # 兜底：如果页面结构不一致，尝试抓第一个百分号
    m2 = ANY_PERCENT_RE.search(text)
    if m2:
        try:
            return _to_rate_0_1(float(m2.group(1)))
        except ValueError:
            return None

    return None


def parse_coverage_rate(path: str) -> float:
    """
    ✅ 单入口：基于文件路径自动解析 coverage 覆盖率
    ◦ xml 优先解析 line-rate

    ◦ html/htm 解析 Coverage: xx%

    ◦ 其他后缀：先按 xml 试，再按 html 试


    返回值：0~1 float
    解析失败：抛出 ValueError
    """
    if not os.path.isfile(path):
        raise ValueError(f"coverage file not found: {path}")

    text = read_text(path)
    ext = os.path.splitext(path)[1].lower()

    # 1) 按后缀判断
    if ext in (".xml",):
        rate = parse_coverage_rate_from_xml_text(text)
        if rate is None:
            raise ValueError(f"cannot parse line-rate from xml: {path}")
        return rate

    if ext in (".html", ".htm"):
        rate = parse_coverage_rate_from_html_text(text)
        if rate is None:
            raise ValueError(f"cannot parse coverage percent from html: {path}")
        return rate

    # 2) 兜底：先 xml 再 html
    rate = parse_coverage_rate_from_xml_text(text)
    if rate is not None:
        return rate
    rate = parse_coverage_rate_from_html_text(text)
    if rate is not None:
        return rate

    raise ValueError(f"cannot parse coverage rate from file: {path}")


# -----------------------------
# Optional: batch parsing
# -----------------------------
@dataclass
class CoverageParseItem:
    name: str
    path: str
    rate: float  # 0~1


def parse_many(paths: Dict[str, str]) -> Dict[str, CoverageParseItem]:
    """
    输入：{"omni": "/.../coverage_omni.xml", "patch": "/.../patch_coverage.html"}
    输出：{"omni": CoverageParseItem(...), ...}
    """
    out: Dict[str, CoverageParseItem] = {}
    for name, p in paths.items():
        rate = parse_coverage_rate(p)
        out[name] = CoverageParseItem(name=name, path=p, rate=rate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse coverage rate from coverage xml/html files. Output rate is 0~1 float."
    )
    ap.add_argument("--file", action="append", default=[], help='single item: name=PATH (repeatable)')
    ap.add_argument("--json", default="", help="optional: write json result to path")
    ap.add_argument("--print", action="store_true", help="print parsed results")
    args = ap.parse_args()

    if not args.file:
        print("Usage examples:\n"
              "  python3 parse_coverage_rates.py --file omni=/path/coverage_omni.xml --file patch=/path/patch_coverage.html\n"
              "  python3 parse_coverage_rates.py --file vllm=... --json /tmp/coverage.json --print",
              file=sys.stderr)
        raise SystemExit(2)

    mapping: Dict[str, str] = {}
    for item in args.file:
        if "=" not in item:
            raise SystemExit(f"bad --file format: {item} (expect name=PATH)")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise SystemExit(f"empty name in --file {item}")
        mapping[name] = path

    parsed = parse_many(mapping)

    if args.print:
        for name in sorted(parsed.keys()):
            it = parsed[name]
            print(f"{it.name}: {it.rate:.6f}  ({it.path})")

    if args.json:
        payload = {k: asdict(v) for k, v in parsed.items()}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if args.print:
            print(f"json written: {args.json}")


if __name__ == "__main__":
    main()