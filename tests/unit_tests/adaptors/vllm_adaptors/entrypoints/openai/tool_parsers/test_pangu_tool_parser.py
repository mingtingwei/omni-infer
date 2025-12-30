import sys
import types
import contextlib
import unittest
import importlib
from unittest.mock import Mock, patch, MagicMock  # noqa: F401

# --- Simple isolated import skeleton (copied from pangu_dense style) ---
# SUT under test:
SUT_MODULE = "omni.adaptors.vllm.entrypoints.openai.tool_parsers.pangu_tool_parser"

pangu_tool_parser = None  # will be imported in setup_module()
_ISOLATION_STATE = {}


def setup_module():
    global pangu_tool_parser, _ISOLATION_STATE

    # ---- snapshot global state we may touch ----
    _ISOLATION_STATE = {
        # optional deps (keep it very light; delete if your env always has them)
        "sys_has_vllm": "vllm" in sys.modules,
        "sys_vllm_obj": sys.modules.get("vllm"),
        "sys_has_openai": "openai" in sys.modules,
        "sys_openai_obj": sys.modules.get("openai"),

        # SUT itself
        "sut_in_sysmodules": SUT_MODULE in sys.modules,
        "sut_obj": sys.modules.get(SUT_MODULE),
    }

    # ---- provide minimal stubs only when the module truly does not exist ----
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError:
        vllm_stub = types.ModuleType("vllm")
        vllm_stub.__path__ = []  # make it package-like
        sys.modules["vllm"] = vllm_stub
        _ISOLATION_STATE["created_vllm_stub"] = True
    else:
        _ISOLATION_STATE["created_vllm_stub"] = False

    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        openai_stub = types.ModuleType("openai")
        openai_stub.__path__ = []
        sys.modules["openai"] = openai_stub
        _ISOLATION_STATE["created_openai_stub"] = True
    else:
        _ISOLATION_STATE["created_openai_stub"] = False

    # ---- import SUT (or reuse existing) ----
    if _ISOLATION_STATE["sut_in_sysmodules"]:
        pangu_tool_parser = _ISOLATION_STATE["sut_obj"]
    else:
        pangu_tool_parser = importlib.import_module(SUT_MODULE)


def teardown_module():
    global pangu_tool_parser, _ISOLATION_STATE

    # ---- unload SUT if we imported it (avoid leaking import-side-effects to others) ----
    if _ISOLATION_STATE and not _ISOLATION_STATE.get("sut_in_sysmodules", False):
        with contextlib.suppress(Exception):
            sys.modules.pop(SUT_MODULE, None)
    pangu_tool_parser = None

    # ---- restore openai ----
    if _ISOLATION_STATE.get("sys_has_openai", False):
        with contextlib.suppress(Exception):
            sys.modules["openai"] = _ISOLATION_STATE.get("sys_openai_obj")
    else:
        with contextlib.suppress(Exception):
            sys.modules.pop("openai", None)

    # ---- restore vllm ----
    if _ISOLATION_STATE.get("sys_has_vllm", False):
        with contextlib.suppress(Exception):
            sys.modules["vllm"] = _ISOLATION_STATE.get("sys_vllm_obj")
    else:
        with contextlib.suppress(Exception):
            sys.modules.pop("vllm", None)

    _ISOLATION_STATE = {}


class TestPanguToolParser(unittest.TestCase):
    # -------------------------
    # Minimal DTOs for isolation
    # -------------------------
    class _EI:
        def __init__(self, tools_called, tool_calls, content):
            self.tools_called = tools_called
            self.tool_calls = tool_calls
            self.content = content

    class _FunctionCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _ToolCall:
        def __init__(self, type, function):
            self.type = type
            self.function = function

    class _DeltaFunctionCall:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments

        def model_dump(self, exclude_none=True):
            d = {}
            if self.name is not None:
                d["name"] = self.name
            if self.arguments is not None:
                d["arguments"] = self.arguments
            return d

    class _DeltaToolCall:
        def __init__(self, index, function, type=None, id=None):
            self.index = index
            self.type = type
            self.id = id
            self.function = function

    class _DeltaMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Allow:
        # int-like bitmask
        ALL = 0b11
        STR = 0b01

    def setUp(self):
        super().setUp()
        # When running under unittest directly, pytest xunit hooks may not run.
        # Ensure the SUT is imported for this module.
        global pangu_tool_parser
        if pangu_tool_parser is None:
            setup_module()
        self.assertIsNotNone(pangu_tool_parser)

        self._patchers = []

        # Patch protocol/DTO classes used by the SUT to minimal stable objects.
        self._start_patch(patch.object(pangu_tool_parser, "ExtractedToolCallInformation", self._EI))
        self._start_patch(patch.object(pangu_tool_parser, "FunctionCall", self._FunctionCall))
        self._start_patch(patch.object(pangu_tool_parser, "ToolCall", self._ToolCall))
        self._start_patch(patch.object(pangu_tool_parser, "DeltaFunctionCall", self._DeltaFunctionCall))
        self._start_patch(patch.object(pangu_tool_parser, "DeltaToolCall", self._DeltaToolCall))
        self._start_patch(patch.object(pangu_tool_parser, "DeltaMessage", self._DeltaMessage))

        # Make tool-call id deterministic.
        self._start_patch(patch.object(pangu_tool_parser, "random_tool_call_id", lambda: "call_123"))

        # Avoid relying on vllm utils implementation details.
        def _find_common_prefix(a: str, b: str) -> str:
            n = 0
            for ca, cb in zip(a, b):
                if ca != cb:
                    break
                n += 1
            return a[:n]

        self._start_patch(patch.object(pangu_tool_parser, "find_common_prefix", _find_common_prefix))

        # Use simple bitmask options to avoid version differences.
        self._start_patch(patch.object(pangu_tool_parser, "Allow", self._Allow))

    def tearDown(self):
        # Ensure all patches are stopped (no leakage to other tests).
        for p in reversed(getattr(self, "_patchers", [])):
            with contextlib.suppress(Exception):
                p.stop()
        self._patchers = []
        super().tearDown()

    def _start_patch(self, patcher):
        started = patcher.start()
        self._patchers.append(patcher)
        return started

    # -------------------------
    # Helpers
    # -------------------------
    class _DummyTokenizer:
        def __init__(self, vocab):
            self._vocab = dict(vocab)

        def get_vocab(self):
            return dict(self._vocab)

    def _make_parser(self, vocab=None):
        """Create a fresh parser instance per test to keep state isolated."""
        if vocab is None:
            vocab = {"[unused11]": 101, "[unused12]": 102}
        tok = self._DummyTokenizer(vocab)

        # Patch ToolParser.__init__ ONLY during instantiation.
        def _toolparser_init(self_obj, tokenizer_obj):
            self_obj.tokenizer = tokenizer_obj
            self_obj.vocab = tokenizer_obj.get_vocab()

        with patch.object(pangu_tool_parser.ToolParser, "__init__", _toolparser_init):
            return pangu_tool_parser.PanguToolParser(tok)

    # ============================================================
    # A) __init__ validation
    # ============================================================
    def test_init_raises_when_start_or_end_token_missing_in_tokenizer_vocab(self):
        # missing both ids -> must raise
        with self.assertRaises(RuntimeError):
            self._make_parser(vocab={"[unused11]": 1})  # missing end token

        with self.assertRaises(RuntimeError):
            self._make_parser(vocab={"[unused12]": 2})  # missing start token

    # ============================================================
    # B) extract_tool_calls (complete output)
    # ============================================================
    def test_extract_tool_calls_returns_plain_text_when_no_tool_tags_present(self):
        parser = self._make_parser()
        req = Mock()

        out = parser.extract_tool_calls("hello world", req)
        self.assertFalse(out.tools_called)
        self.assertEqual(out.tool_calls, [])
        self.assertEqual(out.content, "hello world")

        # start present but end missing -> still plain text
        out2 = parser.extract_tool_calls("hi [unused11]oops", req)
        self.assertFalse(out2.tools_called)
        self.assertEqual(out2.tool_calls, [])
        self.assertEqual(out2.content, "hi [unused11]oops")

    def test_extract_tool_calls_parses_single_tool_call_with_arguments_field(self):
        parser = self._make_parser()
        req = Mock()

        model_output = (
            "prefix "
            "[unused11]"
            '[{"name":"foo","arguments":{"a":1,"b":"x"}}]'
            "[unused12]"
            " suffix"
        )
        out = parser.extract_tool_calls(model_output, req)

        self.assertTrue(out.tools_called)
        self.assertEqual(out.content, "prefix ")
        self.assertEqual(len(out.tool_calls), 1)
        tc = out.tool_calls[0]
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function.name, "foo")

        expected_args = pangu_tool_parser.json.dumps({"a": 1, "b": "x"}, ensure_ascii=False)
        self.assertEqual(tc.function.arguments, expected_args)

    def test_extract_tool_calls_parses_tool_call_fallback_to_parameters_field(self):
        parser = self._make_parser()
        req = Mock()

        model_output = (
            "x"
            "[unused11]"
            '[{"name":"bar","parameters":{"k":"v"}}]'
            "[unused12]"
        )
        out = parser.extract_tool_calls(model_output, req)

        self.assertTrue(out.tools_called)
        self.assertEqual(out.content, "x")
        self.assertEqual(len(out.tool_calls), 1)
        tc = out.tool_calls[0]
        self.assertEqual(tc.function.name, "bar")
        expected_args = pangu_tool_parser.json.dumps({"k": "v"}, ensure_ascii=False)
        self.assertEqual(tc.function.arguments, expected_args)

    def test_extract_tool_calls_sets_content_to_prefix_before_start_token_or_none_when_empty(self):
        parser = self._make_parser()
        req = Mock()

        model_output = (
            "[unused11]"
            '[{"name":"foo","arguments":{"a":1}}]'
            "[unused12]"
        )
        out = parser.extract_tool_calls(model_output, req)

        self.assertTrue(out.tools_called)
        self.assertIsNone(out.content)  # empty prefix => None
        self.assertEqual(len(out.tool_calls), 1)

    def test_extract_tool_calls_falls_back_to_plain_text_on_json_or_schema_error(self):
        parser = self._make_parser()
        req = Mock()

        # invalid JSON inside tag -> must not raise, must downgrade to plain text
        model_output = "prefix [unused11]{bad json}[unused12]"
        out = parser.extract_tool_calls(model_output, req)

        self.assertFalse(out.tools_called)
        self.assertEqual(out.tool_calls, [])
        self.assertEqual(out.content, model_output)

    def test_extract_tool_calls_flattens_multiple_tag_segments_into_single_tool_calls_list(self):
        parser = self._make_parser()
        req = Mock()

        model_output = (
            "A "
            "[unused11]"
            '[{"name":"f1","arguments":{"x":1}}]'
            "[unused12]"
            " B "
            "[unused11]"
            '[{"name":"f2","parameters":{"y":2}}]'
            "[unused12]"
        )
        out = parser.extract_tool_calls(model_output, req)

        self.assertTrue(out.tools_called)
        self.assertEqual(out.content, "A ")
        self.assertEqual(len(out.tool_calls), 2)
        self.assertEqual(out.tool_calls[0].function.name, "f1")
        self.assertEqual(out.tool_calls[1].function.name, "f2")

    # ============================================================
    # C) extract_tool_calls_streaming (stateful streaming)
    # ============================================================
    def test_streaming_returns_none_when_delta_is_only_end_control_token(self):
        parser = self._make_parser()
        req = Mock()

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="...",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[parser.tool_call_end_token_id],
            request=req,
        )
        self.assertIsNone(delta)

    def test_streaming_returns_none_when_delta_is_only_start_control_token(self):
        parser = self._make_parser()
        req = Mock()

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=parser.tool_call_start_token,
            delta_text=parser.tool_call_start_token,
            previous_token_ids=[],
            current_token_ids=[parser.tool_call_start_token_id],
            delta_token_ids=[parser.tool_call_start_token_id],
            request=req,
        )
        self.assertIsNone(delta)

    def test_streaming_emits_plain_text_before_start_token_and_strips_start_token_from_output(self):
        parser = self._make_parser()
        req = Mock()

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="hello" + parser.tool_call_start_token,
            delta_text="hello" + parser.tool_call_start_token,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, "hello")

    def test_streaming_emits_plain_text_when_no_start_token_seen_yet(self):
        parser = self._make_parser()
        req = Mock()

        delta = parser.extract_tool_calls_streaming(
            previous_text="he",
            current_text="hello",
            delta_text="llo",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, "llo")

    def test_streaming_emits_plain_text_when_end_token_in_current_text_but_not_in_delta_text(self):
        parser = self._make_parser()
        req = Mock()

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="..."+parser.tool_call_end_token,
            delta_text="TAIL",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, "TAIL")

    def test_streaming_emits_function_name_once_then_marks_name_sent(self):
        parser = self._make_parser()
        req = Mock()

        tool_arr = [{"name": "foo"}]

        # partial_json_parser.loads is called; stub it.
        self._start_patch(patch.object(pangu_tool_parser.partial_json_parser, "loads", return_value=tool_arr))
        self._start_patch(patch.object(pangu_tool_parser, "is_complete_json", return_value=False))

        current_text = parser.tool_call_start_token + "[{"
        # 1) first call advances to tool 0 but returns None
        d1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNone(d1)
        self.assertEqual(parser.current_tool_id, 0)
        self.assertFalse(parser.current_tool_name_sent)

        # 2) second call emits name
        d2 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="x",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(d2)
        self.assertIsNotNone(d2.tool_calls)
        self.assertEqual(len(d2.tool_calls), 1)
        tc = d2.tool_calls[0]
        self.assertEqual(tc.index, 0)
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.id, "call_123")
        self.assertEqual(tc.function, {"name": "foo"})
        self.assertTrue(parser.current_tool_name_sent)

        # 3) third call should NOT emit name again (no args, incomplete -> None)
        d3 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="y",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNone(d3)

    def test_streaming_emits_arguments_diff_incrementally_and_updates_streamed_args_for_tool(self):
        parser = self._make_parser()
        req = Mock()

        # We will stream args for a single tool:
        # prev args: {"x": 1}
        # cur  args: {"x": 12}
        prev_args = {"x": 1}
        cur_args = {"x": 12}
        prev_arr = [{"name": "foo", "arguments": prev_args}]
        cur_arr = [{"name": "foo", "arguments": cur_args}]

        # Sequence of loads calls:
        # 1) first call -> prev_arr (new tool init, returns None)
        # 2) second call -> prev_arr (emit name)
        # 3) third call -> cur_arr (incomplete JSON => send common prefix diff)
        # 4) fourth call -> cur_arr (complete JSON => send remaining diff)
        self._start_patch(
            patch.object(
                pangu_tool_parser.partial_json_parser,
                "loads",
                side_effect=[prev_arr, prev_arr, cur_arr, cur_arr],
            )
        )
        self._start_patch(
            patch.object(
                pangu_tool_parser,
                "is_complete_json",
                side_effect=[False, False, False, True],
            )
        )

        current_text = parser.tool_call_start_token + "[{"
        # 1) new tool init
        self.assertIsNone(
            parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text="",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[999],
                request=req,
            )
        )
        # 2) emit name
        d2 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="x",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(d2)
        self.assertTrue(parser.current_tool_name_sent)
        self.assertEqual(parser.current_tool_id, 0)
        self.assertEqual(parser.streamed_args_for_tool, [""])

        # 3) incremental diff (incomplete): should emit common prefix from prev->cur
        d3 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="y",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(d3)
        self.assertIsNotNone(d3.tool_calls)
        diff1 = d3.tool_calls[0].function.get("arguments")
        self.assertIsInstance(diff1, str)
        self.assertGreaterEqual(len(diff1), 1)
        self.assertEqual(parser.streamed_args_for_tool[0], diff1)

        # 4) complete: should emit remaining suffix, and streamed_args becomes full cur_args_json
        d4 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="z",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(d4)
        diff2 = d4.tool_calls[0].function.get("arguments")
        self.assertIsInstance(diff2, str)

        full = pangu_tool_parser.json.dumps(cur_args, ensure_ascii=False)
        self.assertEqual(parser.streamed_args_for_tool[0], full)

    def test_streaming_emits_empty_object_arguments_when_json_complete_but_arguments_missing(self):
        parser = self._make_parser()
        req = Mock()

        tool_arr = [{"name": "foo"}]

        self._start_patch(patch.object(pangu_tool_parser.partial_json_parser, "loads", side_effect=[tool_arr, tool_arr, tool_arr]))
        self._start_patch(patch.object(pangu_tool_parser, "is_complete_json", side_effect=[False, False, True]))

        current_text = parser.tool_call_start_token + "[{"

        # 1) init new tool
        self.assertIsNone(
            parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text="",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[999],
                request=req,
            )
        )
        # 2) emit name
        self.assertIsNotNone(
            parser.extract_tool_calls_streaming(
                previous_text=current_text,
                current_text=current_text,
                delta_text="x",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[999],
                request=req,
            )
        )
        self.assertTrue(parser.current_tool_name_sent)
        self.assertEqual(parser.streamed_args_for_tool[0], "")

        # 3) json complete, but no arguments => must emit "{}" once
        d3 = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="y",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        self.assertIsNotNone(d3)
        self.assertEqual(d3.tool_calls[0].function, {"arguments": "{}"})
        self.assertEqual(parser.streamed_args_for_tool[0], "{}")

    def test_streaming_advances_to_next_tool_in_array_and_resets_state_for_new_tool(self):
        parser = self._make_parser()
        req = Mock()

        # Pretend we've already started tool0 and streamed a prefix.
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True

        cur_args = {"x": 12}
        cur_args_json = pangu_tool_parser.json.dumps(cur_args, ensure_ascii=False)

        # Streamed only a prefix so far.
        prefix = cur_args_json[: max(1, len(cur_args_json) // 2)]
        parser.streamed_args_for_tool = [prefix]

        # Now the tool_call_arr grows to 2, which should trigger "start new tool" branch.
        tool_arr = [
            {"name": "foo", "arguments": cur_args},
            {"name": "bar", "arguments": {"y": 1}},
        ]
        self._start_patch(patch.object(pangu_tool_parser.partial_json_parser, "loads", return_value=tool_arr))
        self._start_patch(patch.object(pangu_tool_parser, "is_complete_json", return_value=True))

        current_text = parser.tool_call_start_token + "[{"

        d = parser.extract_tool_calls_streaming(
            previous_text=current_text,
            current_text=current_text,
            delta_text="chunk",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )

        # It should flush remaining args for tool0 (if any) and advance cursor to tool1.
        self.assertEqual(parser.current_tool_id, 1)
        self.assertFalse(parser.current_tool_name_sent)
        self.assertEqual(len(parser.streamed_args_for_tool), 2)
        self.assertEqual(parser.streamed_args_for_tool[1], "")
        self.assertEqual(parser.is_complete, [])

        # Flush delta may be None if no remainder, but usually should contain the remaining suffix.
        remainder = cur_args_json[len(prefix):]
        if remainder:
            self.assertIsNotNone(d)
            self.assertEqual(d.tool_calls[0].index, 0)
            self.assertEqual(d.tool_calls[0].function, {"arguments": remainder})
        else:
            self.assertIsNone(d)

    def test_streaming_returns_none_on_partial_json_parser_malformed_json_without_throwing(self):
        parser = self._make_parser()
        req = Mock()

        # Use the real MalformedJSON class if present; otherwise create a local one
        exc_type = None
        with contextlib.suppress(Exception):
            exc_type = pangu_tool_parser.partial_json_parser.core.exceptions.MalformedJSON  # type: ignore[attr-defined]

        if exc_type is None:
            class _MalformedJSON(Exception):
                pass
            exc_type = _MalformedJSON
            # Best-effort patch the nested attr so the SUT catches it specifically.
            with contextlib.suppress(Exception):
                self._start_patch(
                    patch.object(pangu_tool_parser.partial_json_parser.core.exceptions, "MalformedJSON", exc_type)  # type: ignore[attr-defined]
                )

        self._start_patch(patch.object(pangu_tool_parser.partial_json_parser, "loads", side_effect=exc_type()))
        self._start_patch(patch.object(pangu_tool_parser, "is_complete_json", return_value=False))

        current_text = parser.tool_call_start_token + "[{"

        before_state = (parser.current_tool_id, parser.current_tool_name_sent, list(parser.streamed_args_for_tool), list(parser.is_complete))
        d = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text="x",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[999],
            request=req,
        )
        after_state = (parser.current_tool_id, parser.current_tool_name_sent, list(parser.streamed_args_for_tool), list(parser.is_complete))

        self.assertIsNone(d)
        self.assertEqual(before_state, after_state)


if __name__ == "__main__":
    unittest.main()
