import unittest
import sys
from pathlib import Path

from omni.tools.profiler.trace.analyze_jaeger_all_spans_to_csv import (
    percentile_from_sorted,
    try_parse_start_us,
    get_span_start_us,
    extract_tag_value,
    aggregate_spans_with_start,
    compute_summary_for_group
)

class TestJaegerSpanAnalyzer(unittest.TestCase):
    """ Test for analyzing jaeger all spans to csv"""

    def test_percentile_from_sorted(self):
        """ Test sorted correct """

        self.assertEqual(percentile_from_sorted([], 50), 0.0)

        self.assertEqual(percentile_from_sorted([2.0], 99), 2.0)

        self.assertEqual(percentile_from_sorted([1.0, 2.0, 3.0, 4.0], 50), 2.5)

        sorted_odd = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(percentile_from_sorted(sorted_odd, 50), 3.0)

        self.assertAlmostEqual(percentile_from_sorted(sorted_odd, 90), 4.6, places=3)

    def test_try_parse_start_us(self):
        """ Test float in ms, us, ns """

        self.assertEqual(try_parse_start_us(1234567890123), 1234567890123)

        self.assertEqual(try_parse_start_us(1234567890), 1234567890 * 1000)

        self.assertEqual(try_parse_start_us(1234567890123456), 1234567890123456 // 1000)

        self.assertEqual(try_parse_start_us(123), None)

        self.assertEqual(try_parse_start_us(None), None)
        self.assertEqual(try_parse_start_us("invalid"), None)

        self.assertEqual(try_parse_start_us(1234567890.123), int(1234567890 * 1000))

    def test_get_span_start_us(self):
        """ Test span start """

        self.assertEqual(get_span_start_us({"startTime": 1234567890123}), 1234567890123)
        self.assertEqual(get_span_start_us({"startTimeMillis": 1234567890}), 1234567890 * 1000)
        self.assertEqual(get_span_start_us({"startTimeUnixNano": 1234567890123456}), 1234567890123456 // 1000)
        # invalid key
        self.assertEqual(get_span_start_us({"no_start_key": 123}), None)

    def test_extract_tag_value(self):
        """ Test tag extract """

        span_with_tag = {"tags": [{"key": "request_id", "value": "r1"}]}
        self.assertEqual(extract_tag_value(span_with_tag, ["request_id"]), "r1")

        self.assertEqual(extract_tag_value(span_with_tag, ["trace_id"]), None)

        span_no_tags = {"operationName": "test-op"}
        self.assertEqual(extract_tag_value(span_no_tags, ["request_id"]), None)

    def test_aggregate_spans_with_start(self):
        """ Test span aggregate """
        test_traces = [
            {
                "traceID": "t1",
                "spans": [
                    {"operationName": "op1", "duration": 1000, "startTime": 1234567890123},
                    {"operation": "op2", "duration": 3500, "startTimeMillis": 1234567891}
                ]
            },
            {
                "trace_id": "t2",
                "spans": [{"operationName": "op1", "duration": 5000, "startTime": 1234567890124}]
            }
        ]
        groups, earliest_start, all_spans = aggregate_spans_with_start(test_traces)

        self.assertEqual(groups["op1"], [1000, 5000])
        self.assertEqual(groups["op2"], [3500])
        self.assertEqual(earliest_start["op1"], 1234567890123)
        self.assertEqual(len(all_spans), 3)

    def test_compute_summary_for_group(self):
        """ Test None group and valid group"""

        empty_summary = compute_summary_for_group([])
        self.assertEqual(empty_summary["count"], 0)

        durations = [1000, 2000, 3000, 4000, 5000]
        summary = compute_summary_for_group(durations)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["max_ms"], 5.0)
        self.assertEqual(summary["avg_ms"], 3.0)
        self.assertAlmostEqual(int(summary["p99_ms"]), 4, places=3)

if __name__ == "__main__":
    unittest.main()