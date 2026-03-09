import sys
import types
import contextlib
import unittest
import importlib
import os
import json
from unittest.mock import Mock, patch, MagicMock  # noqa: F401

# --- Simple isolated import skeleton (copied from pangu_dense style) ---
# SUT under test:
SUT_MODULE = "omni.adaptors.vllm.entrypoints.openai.tool_parsers.openai_tool_parser"

OPENAI_TOOL_PARSER = None  # will be imported in setup_module()
_ISOLATION_STATE = {}


def setup_module():
    global OPENAI_TOOL_PARSER, _ISOLATION_STATE

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
        OPENAI_TOOL_PARSER = _ISOLATION_STATE["sut_obj"]
    else:
        OPENAI_TOOL_PARSER = importlib.import_module(SUT_MODULE)


def teardown_module():
    global OPENAI_TOOL_PARSER, _ISOLATION_STATE

    # ---- unload SUT if we imported it (avoid leaking import-side-effects to others) ----
    if _ISOLATION_STATE and not _ISOLATION_STATE.get("sut_in_sysmodules", False):
        with contextlib.suppress(Exception):
            sys.modules.pop(SUT_MODULE, None)
    OPENAI_TOOL_PARSER = None

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


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    """Minimal message-like object to drive extract_tool_calls() logic.

    It only implements attributes used by the SUT:
      - content: list with .text attribute
      - recipient: str | None
      - content_type: str | None
      - channel: str | None
    """

    def __init__(
        self,
        *,
        text: str | None = None,
        content=None,
        recipient: str | None = None,
        content_type: str | None = None,
        channel: str | None = None,
    ):
        if content is not None:
            self.content = content
        elif text is None:
            self.content = []
        else:
            self.content = [_FakeContent(text)]
        self.recipient = recipient
        self.content_type = content_type
        self.channel = channel


class _FakeParser:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.process_calls = []
        self.current_content = None

    def process(self, token_id: int):
        self.process_calls.append(token_id)


class TestOpenAIToolParser(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # When running under unittest directly, pytest xunit hooks may not run.
        # Ensure the SUT is imported for this module.
        global OPENAI_TOOL_PARSER
        if OPENAI_TOOL_PARSER is None:
            setup_module()
        self.assertIsNotNone(OPENAI_TOOL_PARSER)

    def tearDown(self):
        super().tearDown()

    def _new_parser_instance(self):
        """Create OpenAIToolParser instance with base-class init patched to avoid external deps."""
        m = OPENAI_TOOL_PARSER
        with patch.object(m.ToolParser, "__init__", lambda self, tokenizer: None):
            return m.OpenAIToolParser(tokenizer=Mock())

    def test_get_encoding_cached_singleton(self):
        m = OPENAI_TOOL_PARSER
        old = getattr(m, "_harmony_encoding", None)
        try:
            m._harmony_encoding = None
            sentinel = object()
            with patch.object(m, "load_harmony_encoding", return_value=sentinel) as load_mock:
                enc1 = m.get_encoding()
                enc2 = m.get_encoding()

                self.assertIs(enc1, sentinel)
                self.assertIs(enc2, sentinel)
                self.assertEqual(load_mock.call_count, 1)
                load_mock.assert_called_with(m.HarmonyEncodingName.HARMONY_GPT_OSS)
        finally:
            m._harmony_encoding = old

    def test_parse_output_into_messages_process_order_without_decode_role(self):
        m = OPENAI_TOOL_PARSER
        fake_parser = _FakeParser(messages=[])

        with patch.object(m, "get_streamable_parser_for_assistant", return_value=fake_parser):
            with patch.dict(os.environ, {"ROLE": "prefill"}, clear=False):
                out = m.parse_output_into_messages([10, 11, 12])

        self.assertIs(out, fake_parser)
        self.assertEqual(fake_parser.process_calls, [10, 11, 12])

    def test_parse_output_into_messages_prepends_default_channel_token_when_role_decode(self):
        m = OPENAI_TOOL_PARSER
        fake_parser = _FakeParser(messages=[])

        with patch.object(m, "get_streamable_parser_for_assistant", return_value=fake_parser):
            with patch.dict(os.environ, {"ROLE": "decode"}, clear=False):
                out = m.parse_output_into_messages([1, 2, 3])

        self.assertIs(out, fake_parser)
        self.assertEqual(fake_parser.process_calls, [200005, 1, 2, 3])

    def test_extract_tool_calls_requires_token_ids_and_rejects_text_only(self):
        m = OPENAI_TOOL_PARSER
        parser = self._new_parser_instance()

        with self.assertRaises(NotImplementedError):
            parser.extract_tool_calls(
                model_output="ignored",
                request=Mock(),
                token_ids=None,
            )

    def test_extract_tool_calls_extracts_function_tool_call_with_valid_json_content_type_default_or_json(self):
        m = OPENAI_TOOL_PARSER
        parser = self._new_parser_instance()

        for content_type in (None, "application/json; charset=utf-8"):
            with self.subTest(content_type=content_type):
                msg_text = '{"a": 1}'
                fake_parser = _FakeParser(messages=[
                    _FakeMessage(
                        text=msg_text,
                        recipient="functions.my_func",
                        content_type=content_type,
                        channel="assistant",
                    )
                ])

                with patch.object(m, "parse_output_into_messages", return_value=fake_parser):
                    info = parser.extract_tool_calls(
                        model_output="ignored",
                        request=Mock(),
                        token_ids=[123],
                    )

                self.assertTrue(getattr(info, "tools_called"))
                tool_calls = getattr(info, "tool_calls")
                self.assertEqual(len(tool_calls), 1)

                tc0 = tool_calls[0]
                fn = getattr(tc0, "function")
                self.assertEqual(getattr(fn, "name"), "my_func")
                expected_args = json.dumps(json.loads(msg_text))
                self.assertEqual(getattr(fn, "arguments"), expected_args)
                self.assertIsNone(getattr(info, "content"))

    def test_extract_tool_calls_falls_back_to_raw_args_and_logs_when_json_invalid(self):
        m = OPENAI_TOOL_PARSER
        parser = self._new_parser_instance()

        bad_json = "{bad json"
        fake_parser = _FakeParser(messages=[
            _FakeMessage(
                text=bad_json,
                recipient="functions.bad",
                content_type=None,  # treated as json branch
                channel="assistant",
            )
        ])

        with patch.object(m, "parse_output_into_messages", return_value=fake_parser):
            with patch.object(m.logger, "exception", autospec=True) as exc_mock:
                info = parser.extract_tool_calls(
                    model_output="ignored",
                    request=Mock(),
                    token_ids=[1],
                )

        self.assertTrue(getattr(info, "tools_called"))
        self.assertEqual(len(getattr(info, "tool_calls")), 1)
        fn = getattr(getattr(info, "tool_calls")[0], "function")
        self.assertEqual(getattr(fn, "name"), "bad")
        self.assertEqual(getattr(fn, "arguments"), bad_json)
        self.assertGreaterEqual(exc_mock.call_count, 1)

    def test_extract_tool_calls_passthrough_args_when_content_type_non_json(self):
        m = OPENAI_TOOL_PARSER
        parser = self._new_parser_instance()

        raw_args = "hello world"
        fake_parser = _FakeParser(messages=[
            _FakeMessage(
                text=raw_args,
                recipient="functions.echo",
                content_type="text/plain",
                channel="assistant",
            )
        ])

        with patch.object(m, "parse_output_into_messages", return_value=fake_parser):
            with patch.object(m.logger, "exception", autospec=True) as exc_mock:
                info = parser.extract_tool_calls(
                    model_output="ignored",
                    request=Mock(),
                    token_ids=[9],
                )

        self.assertTrue(getattr(info, "tools_called"))
        self.assertEqual(len(getattr(info, "tool_calls")), 1)
        fn = getattr(getattr(info, "tool_calls")[0], "function")
        self.assertEqual(getattr(fn, "name"), "echo")
        self.assertEqual(getattr(fn, "arguments"), raw_args)
        self.assertEqual(exc_mock.call_count, 0)

    def test_extract_tool_calls_returns_final_content_and_ignores_messages_with_empty_content(self):
        m = OPENAI_TOOL_PARSER
        parser = self._new_parser_instance()

        tool_text = '{"x": 2}'
        final_text = "DONE"
        fake_parser = _FakeParser(messages=[
            # should be ignored because content is empty
            _FakeMessage(
                content=[],
                recipient="functions.empty",
                content_type=None,
                channel="assistant",
            ),
            # valid tool call
            _FakeMessage(
                text=tool_text,
                recipient="functions.ok",
                content_type=None,
                channel="assistant",
            ),
            # unrelated non-final, non-tool message: ignored
            _FakeMessage(
                text="some assistant text",
                recipient=None,
                content_type="text/plain",
                channel="assistant",
            ),
            # final content
            _FakeMessage(
                text=final_text,
                recipient=None,
                content_type="text/plain",
                channel="final",
            ),
        ])

        with patch.object(m, "parse_output_into_messages", return_value=fake_parser):
            info = parser.extract_tool_calls(
                model_output="ignored",
                request=Mock(),
                token_ids=[42],
            )

        tool_calls = getattr(info, "tool_calls")
        self.assertEqual(len(tool_calls), 1)  # empty-content tool msg must not count
        fn = getattr(tool_calls[0], "function")
        self.assertEqual(getattr(fn, "name"), "ok")
        self.assertEqual(getattr(fn, "arguments"), json.dumps(json.loads(tool_text)))
        self.assertEqual(getattr(info, "content"), final_text)


if __name__ == "__main__":
    unittest.main()
