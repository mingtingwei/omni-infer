#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Optional


def load_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict")
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception as exc:
            raise ValueError(f"{path} has non-numeric duration for {k}: {v}") from exc
    return out


def infer_name(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"test_durations_(.+)\.json$", base)
    if m:
        return m.group(1)
    return base


def collect_durations(
    container_json_paths: List[str],
) -> Tuple[Dict[str, float], List[Tuple[str, str, float, float]], List[Tuple[str, float]]]:
    """Load per-container duration json files.

    Returns:
      merged: nodeid -> duration seconds
      duplicates: (container_name, nodeid, old_duration, new_duration)
      totals: (container_name, total_seconds)
    """

    merged: Dict[str, float] = {}
    duplicates: List[Tuple[str, str, float, float]] = []
    totals: List[Tuple[str, float]] = []

    for path in container_json_paths:
        if not os.path.isfile(path):
            print(f"[WARN] missing file: {path}")
            continue
        name = infer_name(path)
        data = load_json(path)
        total = sum(data.values())
        totals.append((name, total))

        for k, v in data.items():
            if k in merged and abs(merged[k] - v) > 1e-9:
                duplicates.append((name, k, merged[k], v))
                # keep the larger value to be conservative
                merged[k] = max(merged[k], v)
            else:
                merged[k] = v

    return merged, duplicates, totals


def balance_ratio(totals: List[Tuple[str, float]]) -> float:
    if not totals:
        return 1.0
    min_t = min(t for _, t in totals)
    max_t = max(t for _, t in totals)
    if min_t <= 0.0:
        return float("inf") if max_t > 0 else 1.0
    return max_t / min_t


def print_totals_and_balance(totals: List[Tuple[str, float]],) -> Tuple[float, bool]:
    if not totals:
        print("[WARN] no per-container totals to check")
        return 1.0, True

    print("[INFO] per-container total durations (seconds):")
    for name, total in sorted(totals, key=lambda x: x[0]):
        print(f"  {name}: {total:.2f}")

    # 时间均衡规则 260224：DT_5 DT_6 DT_7中 最大值与最小值的比值不超过1.5
    min_between_DT_567 = min(t for n, t in totals[4:7])
    max_between_DT_567 = max(t for n, t in totals[4:7])
    ratio = max_between_DT_567 / min_between_DT_567 if min_between_DT_567 > 0 else 1.0
    threshold = 1.5
    ok = ratio <= threshold
    print(f"[INFO] balance check: max(DT_6,DT_7)/min(DT_6,DT_7) = {ratio:.2f} (threshold {threshold:.2f})")
    if not ok:
        print("[WARN] containers are imbalanced. Please update test_durations.json.")
    return ratio, ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge per-container test duration json files, and/or check load balance among containers."
        )
    )
    ap.add_argument(
        "--action",
        choices=["both", "merge", "check"],
        default="both",
        help="what to do: both (default), merge only, or check only",
    )
    ap.add_argument("--out", default=None, help="output merged json path (required for merge)")
    ap.add_argument("--container-json", action="append", default=[], help="path to a container json (repeatable)")
    ap.add_argument(
        "--fail-on-imbalance",
        action="store_true",
        help="exit non-zero if balance ratio exceeds threshold (only meaningful when doing check)",
    )
    args = ap.parse_args()

    if not args.container_json:
        print("[ERROR] no --container-json provided", file=sys.stderr)
        return 2

    merged, duplicates, totals = collect_durations(args.container_json)

    if not merged:
        print("[ERROR] no data loaded from container json files", file=sys.stderr)
        return 3

    did_merge = args.action in ("both", "merge")
    did_check = args.action in ("both", "check")

    if did_merge:
        if not args.out:
            print("[ERROR] --out is required when action includes merge", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, sort_keys=True, ensure_ascii=False)

        print(f"[INFO] merged durations json written: {args.out}")
        print(f"[INFO] total tests: {len(merged)}")

    if duplicates:
        print(f"[WARN] duplicate nodeids detected: {len(duplicates)} (kept max duration)")

    ok = True
    if did_check:
        _, ok = print_totals_and_balance(totals,)
        if args.fail_on_imbalance and not ok:
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())