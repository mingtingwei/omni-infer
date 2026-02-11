#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# usage
# python parse_ut_logs.py -o merged_summary.log /path/to/install_logs

SUMMARY_RE = re.compile(
    r"=+.*?(?P<summary>(\d+\s+passed|"
    r"\d+\s+failed|"
    r"\d+\s+error|"
    r"\d+\s+errors|"
    r"\d+\s+skipped|"
    r"\d+\s+xfailed|"
    r"\d+\s+xpassed|"
    r"\d+\s+warnings?).*?)=+",
    re.IGNORECASE,
)

# Common pytest section headers
SECTION_RE = re.compile(r"=+\s+(FAILURES|ERRORS)\s+=+")

# Typical pytest node id lines inside failure/error sections
NODEID_RE = re.compile(r"^_+ (.+?) _+$")

# Alternative: "FAILED path::test_name - message"
FAILED_LINE_RE = re.compile(r"^(FAILED|ERROR)\s+(.+?)(\s+-\s+.+)?$")
SHORT_SUMMARY_RE = re.compile(r"=+\s+short test summary info\s+=+", re.IGNORECASE)
COVERAGE_SECTION_RE = re.compile(r"=+\s+tests coverage\s+=+", re.IGNORECASE)

# Coverage table lines (pytest-cov)
COVERAGE_SEP_RE = re.compile(r"^-{10,}$")
COVERAGE_LINE_RE = re.compile(r"^/.*\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+%$")
TOTAL_LINE_RE = re.compile(r"^TOTAL\s+\d+\s+\d+.*\d+%$")


@dataclass
class LogResult:
    name: str
    summary_line: Optional[str] = None
    short_summary_lines: List[str] = field(default_factory=list)
    fail_error_blocks: List[str] = field(default_factory=list)


def _extract_summaries(lines: List[str]) -> dict:
    # Keep last summary line as-is
    summary_line = None
    for line in lines:
        m = SUMMARY_RE.search(line)
        if not m:
            continue
        summary_line = line
    return summary_line


def _extract_short_summary(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    in_short = False
    while i < len(lines):
        line = lines[i]
        if SHORT_SUMMARY_RE.match(line):
            in_short = True
            i += 1
            continue
        if in_short:
            if line.startswith("=") or line.strip() == "":
                in_short = False
                i += 1
                continue
            if FAILED_LINE_RE.match(line):
                out.append(line)
        i += 1
    return out


def _extract_failed_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in lines:
        if FAILED_LINE_RE.match(line):
            if line not in seen:
                out.append(line)
                seen.add(line)
    return out


def _extract_fail_error_blocks(lines: List[str]) -> List[str]:
    blocks: List[str] = []
    i = 0
    in_section = False
    current: List[str] = []
    skip_coverage = False
    while i < len(lines):
        line = lines[i]
        if COVERAGE_SECTION_RE.match(line):
            skip_coverage = True
            i += 1
            continue
        if skip_coverage:
            if line.startswith("=") and "summary" in line.lower():
                skip_coverage = False
            i += 1
            continue

        if SECTION_RE.match(line):
            if current:
                blocks.append("\n".join(current).rstrip())
                current = []
            in_section = True
            current.append(line)
            i += 1
            continue

        if in_section:
            if SUMMARY_RE.search(line) or SHORT_SUMMARY_RE.match(line):
                if current:
                    blocks.append("\n".join(current).rstrip())
                    current = []
                in_section = False
                i += 1
                continue
            if not (
                COVERAGE_SEP_RE.match(line)
                or COVERAGE_LINE_RE.match(line)
                or TOTAL_LINE_RE.match(line)
            ):
                current.append(line)
        i += 1

    if current:
        blocks.append("\n".join(current).rstrip())
    return blocks


def parse_log(path: str) -> LogResult:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    name = os.path.splitext(os.path.basename(path))[0]
    summary_line = _extract_summaries(lines)
    short_summary_lines = _extract_short_summary(lines)
    if not short_summary_lines:
        short_summary_lines = _extract_failed_lines(lines)
    fail_error_blocks = _extract_fail_error_blocks(lines)

    return LogResult(
        name=name,
        summary_line=summary_line,
        short_summary_lines=short_summary_lines,
        fail_error_blocks=fail_error_blocks,
    )


def format_result(res: LogResult) -> str:
    parts = []
    parts.append(f"===={res.name} total summary====")
    if res.short_summary_lines:
        parts.append("=====cases=====")
        parts.extend(res.short_summary_lines)
    if res.summary_line:
        parts.append(res.summary_line)
    return "\n".join(parts).rstrip() + "\n"


def format_detail(res: LogResult) -> str:
    parts = []
    if res.fail_error_blocks:
        parts.append(f"===={res.name} errors====")
        parts.append("=====info=====")
        for block in res.fail_error_blocks:
            if block:
                parts.append(block)
                parts.append("")
    return "\n".join(parts).rstrip() + ("\n" if parts else "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse pytest logs and produce a merged summary with fail/error details."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Log files or directories. If omitted, uses *.log in current directory.",
    )
    ap.add_argument(
        "-o",
        "--output",
        help="Output file path. Defaults to stdout.",
    )
    args = ap.parse_args()

    paths: List[str] = []
    output_basename = os.path.basename(args.output) if args.output else None
    def _is_excluded(name: str) -> bool:
        base = os.path.basename(name)
        if output_basename and base == output_basename:
            return True
        if base.lower().startswith("merged_summary") and base.lower().endswith(".log"):
            return True
        return False

    if not args.paths:
        paths = [p for p in os.listdir(".") if p.lower().endswith(".log")]
        paths = [p for p in paths if not _is_excluded(p)]
        paths.sort()
    else:
        for p in args.paths:
            if os.path.isdir(p):
                for name in os.listdir(p):
                    if name.lower().endswith(".log"):
                        if not _is_excluded(name):
                            paths.append(os.path.join(p, name))
            else:
                if not _is_excluded(p):
                    paths.append(p)
        paths.sort()

    results = [parse_log(p) for p in paths]
    summary = "".join(format_result(r) for r in results)
    details = "".join(format_detail(r) for r in results)
    output = summary + details

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
