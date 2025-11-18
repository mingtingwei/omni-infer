#!/usr/bin/env python3
"""
analyze_jaeger_all_spans_to_csv.py

Fetch traces from Jaeger Query API for a given service and time window, aggregate per-span (operationName)
durations and write a CSV summary with these columns:

operation,count,avg_ms,min_ms,max_ms,p50_ms,p90_ms,p95_ms,p99_ms


Usage:
  python analyze_jaeger_all_spans_to_csv.py \
    --host http://collector-host:16686 \
    --service <server_name> \
    --lookback-minutes <time_window_in_minutes> \
    --limit <max_traces_to_fetch> \
    --output <yourfile.csv> \
    --operation <operation_name>

Example:
  python analyze_jaeger_all_spans_to_csv.py \
    --host http://7.150.8.141:16686  \
    --service transformers \
    --lookback-minutes 120 \
    --limit 2000 \
    --output spans_summary.csv \
    --operation "Prefill done execute_model"

Notes:
 - Jaeger API span.duration is in microseconds (us). CSV outputs durations in milliseconds (ms).
 - The script fetches up to --limit <max_traces_to_fetch> traces (default 1000). Increase limit or lookback window to collect more samples.
 - optional parameter: --operation <operation_name> prints the the details of spans with the minimum and maximum duration for that operation, helping you identify which requests correspond to these extremes.
"""
from __future__ import annotations
import requests
import time
import argparse
from datetime import datetime, timedelta
import math
import csv
import sys
from typing import List, Dict, Tuple, Optional

DEFAULT_LIMIT = 1000
MAX_MINMAX_PRINT = 10

def percentile_from_sorted(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (n - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_vals[lower])
    frac = pos - lower
    return float(sorted_vals[lower]) * (1 - frac) + float(sorted_vals[upper]) * frac

def fetch_traces(host: str, service: str, start_micros: int, end_micros: int, limit: int) -> List[Dict]:
    url = f"{host.rstrip('/')}/api/traces"
    params = {
        "service": service,
        "start": start_micros,
        "end": end_micros,
        "limit": limit
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def try_parse_start_us(val) -> Optional[int]:
    """
    Try to parse various possible start time representations into microseconds (int).
    Accepts numeric microsecond epoch, numeric milliseconds, or ISO8601 string.
    Returns None if cannot parse.
    """
    if val is None:
        return None
    # numeric-like
    try:
        if isinstance(val, (int, float)):
            # Heuristic: if value > 1e15, treat as nanoseconds -> convert to us
            v = int(val)
            if v > 10**15:
                return v // 1000
            # if value looks like microseconds (>=1e12) return as-is
            if v >= 10**12:
                return v
            # if value looks like milliseconds (>=1e9) convert to microseconds
            if v >= 10**9:
                return v * 1000
            return None
    except Exception:
        return None
    return None

def get_span_start_us(span: Dict) -> Optional[int]:
    """
    Try several common keys to obtain span start time in microseconds.
    """
    candidate_keys = ["startTime", "start", "startTimeUnixNano", "startTimeMillis", "start_time", "start_time_us", "start_time_ns"]
    for k in candidate_keys:
        v = span.get(k)
        parsed = try_parse_start_us(v)
        if parsed:
            return parsed
    return None

def extract_tag_value(span: Dict, candidate_keys: List[str]) -> Optional[str]:
    for t in span.get("tags", []) or []:
        k = t.get("key")
        if k in candidate_keys:
            return t.get("value")
    return None

def aggregate_spans_with_start(traces: List[Dict]) -> Tuple[Dict[str, List[int]], Dict[str, Optional[int]], List[Dict]]:
    """
    Return:
      - mapping operationName -> list of durations (microseconds)
      - mapping operationName -> earliest start time (microseconds) or None
      - list of all matched span dicts for possible per-operation inspection
    """
    groups: Dict[str, List[int]] = {}
    earliest_start: Dict[str, Optional[int]] = {}
    all_spans: List[Dict] = []  # each entry contains keys: operation, duration_us, start_us, trace_id, span_id, request_id
    for trace in traces:
        trace_id = trace.get("traceID") or trace.get("traceId") or trace.get("trace_id")
        for span in trace.get("spans", []):
            opname = span.get("operationName") or span.get("operation") or "<no-op>"
            duration = span.get("duration", 0)
            if duration is None:
                continue
            try:
                duration_us = int(duration)
            except Exception:
                continue
            # get start time if present
            start_us = get_span_start_us(span)
            # record groups
            groups.setdefault(opname, []).append(duration_us)
            # update earliest start
            prev = earliest_start.get(opname)
            if start_us is not None:
                if prev is None or start_us < prev:
                    earliest_start[opname] = start_us
            else:
                # keep None if no start times seen yet
                earliest_start.setdefault(opname, prev)
            # collect for later per-operation inspection
            span_id = span.get("spanID") or span.get("spanId") or span.get("span_id")
            req_id = extract_tag_value(span, ["request_id", "request id", "requestId", "request-id", "requestid"])
            all_spans.append({
                "operation": opname,
                "duration_us": duration_us,
                "start_us": start_us,
                "trace_id": trace_id,
                "span_id": span_id,
                "request_id": req_id,
                "raw_span": span
            })
    return groups, earliest_start, all_spans

def compute_summary_for_group(durations_us: List[int]) -> Dict[str, float]:
    if not durations_us:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    sorted_vals = sorted(durations_us)
    count = len(sorted_vals)
    total_us = sum(sorted_vals)
    avg_ms = (total_us / count) / 1000.0
    min_ms = sorted_vals[0] / 1000.0
    max_ms = sorted_vals[-1] / 1000.0
    p50_ms = percentile_from_sorted(sorted_vals, 50) / 1000.0
    p90_ms = percentile_from_sorted(sorted_vals, 90) / 1000.0
    p95_ms = percentile_from_sorted(sorted_vals, 95) / 1000.0
    p99_ms = percentile_from_sorted(sorted_vals, 99) / 1000.0
    return {
        "count": count,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
    }

def write_csv(output_path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["operation", "count", "avg_ms", "min_ms", "max_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def pretty_ms(x: float) -> str:
    return f"{x:.3f} ms"

def print_operation_detail(operation: str, groups: Dict[str, List[int]], all_spans: List[Dict]):
    durations = groups.get(operation)
    if not durations:
        print(f"No data for operation '{operation}'")
        return
    summary = compute_summary_for_group(durations)
    print("\n=== Summary ===")
    print(f"Operation: {operation}")
    print(f"Samples: {summary['count']}")
    print(f"Avg duration: {summary['avg_ms']:.3f} ms")
    print(f"Min duration: {summary['min_ms']:.3f} ms")
    print(f"P50 (median): {summary['p50_ms']:.3f} ms")
    print(f"P90: {summary['p90_ms']:.3f} ms")
    print(f"P95: {summary['p95_ms']:.3f} ms")
    print(f"P99: {summary['p99_ms']:.3f} ms")
    print(f"Max duration: {summary['max_ms']:.3f} ms\n")

    # find min/max entries among all_spans for this operation
    min_us = int(summary['min_ms'] * 1000)
    max_us = int(summary['max_ms'] * 1000)
    min_entries = [e for e in all_spans if e["operation"] == operation and e["duration_us"] == min_us]
    max_entries = [e for e in all_spans if e["operation"] == operation and e["duration_us"] == max_us]

    print(f"Spans with MIN duration ({pretty_ms(summary['min_ms'])}) -- printing up to {MAX_MINMAX_PRINT}:")
    for e in min_entries[:MAX_MINMAX_PRINT]:
        print(f" - {operation} | {pretty_ms(e['duration_us']/1000.0)} | trace_id={e.get('trace_id') or '<no-trace-id>'} | span_id={e.get('span_id') or '<no-span-id>'} | request_id={e.get('request_id') or '<no-request-id>'}")

    print(f"\nSpans with MAX duration ({pretty_ms(summary['max_ms'])}) -- printing up to {MAX_MINMAX_PRINT}:")
    for e in max_entries[:MAX_MINMAX_PRINT]:
        print(f" - {operation} | {pretty_ms(e['duration_us']/1000.0)} | trace_id={e.get('trace_id') or '<no-trace-id>'} | span_id={e.get('span_id') or '<no-span-id>'} | request_id={e.get('request_id') or '<no-request-id>'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:16686", help="Jaeger Query API host (including http:// and port)")
    parser.add_argument("--service", required=True, help="service name as shown in Jaeger UI")
    parser.add_argument("--lookback-minutes", type=int, default=60, help="lookback window in minutes")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max traces to fetch from Jaeger")
    parser.add_argument("--output", default="spans_summary.csv", help="CSV output path")
    parser.add_argument("--operation", default=None, help="If provided, print detailed summary and min/max spans for this operation")
    args = parser.parse_args()

    host = args.host.rstrip("/")
    service = args.service
    lookback = args.lookback_minutes
    limit = args.limit
    outpath = args.output
    op_to_inspect = args.operation

    now_us = int(time.time() * 1_000_000)
    start_us = int((datetime.now() - timedelta(minutes=lookback)).timestamp() * 1_000_000)
    print(f"Fetching traces for service='{service}' from {start_us} to {now_us} (microseconds), limit={limit} ...")

    try:
        traces = fetch_traces(host, service, start_us, now_us, limit)
    except Exception as e:
        print("Error fetching traces from Jaeger:", e, file=sys.stderr)
        sys.exit(2)

    print(f"Fetched {len(traces)} traces. Aggregating spans by operationName...")
    groups, earliest_start, all_spans = aggregate_spans_with_start(traces)
    print(f"Found {len(groups)} distinct operationName(s). Computing summaries...")

    # Build rows and sort by earliest_start (ascending). If earliest_start missing, place at end.
    rows = []
    sort_list = []
    for opname, durations in groups.items():
        summary = compute_summary_for_group(durations)
        min_start = earliest_start.get(opname)
        sort_key = min_start if min_start is not None else 10**30
        sort_list.append((sort_key, opname, summary))

    # sort by sort_key (earliest start time)
    sort_list.sort(key=lambda t: (t[0], t[1]))

    for sort_key, opname, summary in sort_list:
        row = {
            "operation": opname,
            "count": summary["count"],
            "avg_ms": f"{summary['avg_ms']:.6f}",
            "min_ms": f"{summary['min_ms']:.6f}",
            "max_ms": f"{summary['max_ms']:.6f}",
            "p50_ms": f"{summary['p50_ms']:.6f}",
            "p90_ms": f"{summary['p90_ms']:.6f}",
            "p95_ms": f"{summary['p95_ms']:.6f}",
            "p99_ms": f"{summary['p99_ms']:.6f}",
        }
        rows.append(row)

    print(f"Writing CSV to '{outpath}' ({len(rows)} rows)...")
    try:
        write_csv(outpath, rows)
    except Exception as e:
        print("Error writing CSV:", e, file=sys.stderr)
        sys.exit(3)

    print("Done. Summary CSV written.")

    # If user requested an operation detail, print it now
    if op_to_inspect:
        print_operation_detail(op_to_inspect, groups, all_spans)

if __name__ == "__main__":
    main()