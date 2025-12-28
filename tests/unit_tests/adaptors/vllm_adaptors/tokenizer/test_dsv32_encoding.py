import pytest

from omni.adaptors.vllm.tokenizer.deepseek_v32_encoding import (
    to_json,
    tools_from_openai_format,
    tool_calls_from_openai_format,
    tool_calls_to_openai_format,
)
def test_to_json_basic():
    assert to_json({"a": 1}) == '{"a": 1}'
    assert to_json("你好") == '"你好"'
def test_tool_format_conversion():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "add numbers",
                "parameters": {"x": 1},
            },
        }
    ]

    simplified = tools_from_openai_format(tools)
    assert simplified[0]["name"] == "add"

    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "arguments": '{"x": 1}',
            },
        }
    ]

    parsed = tool_calls_from_openai_format(tool_calls)
    assert parsed[0]["name"] == "add"

    roundtrip = tool_calls_to_openai_format(parsed)
    assert roundtrip[0]["function"]["name"] == "add"


from omni.adaptors.vllm.tokenizer.deepseek_v32_encoding import encode_arguments_to_dsml, decode_dsml_to_arguments

def test_encode_arguments_to_dsml():
    tool_call = {
        "name": "add",
        "arguments": {
            "x": 1,
            "y": "hello",
        },
    }

    dsml = encode_arguments_to_dsml(tool_call)

    assert 'name="x"' in dsml
    assert 'string="false"' in dsml
    assert 'name="y"' in dsml
    assert 'string="true"' in dsml


def test_decode_dsml_to_arguments():
    tool_args = {
        "x": ("1", "false"),
        "y": ("hello", "true"),
    }

    decoded = decode_dsml_to_arguments("add", tool_args)

    assert decoded["name"] == "add"
    assert '"x": 1' in decoded["arguments"]
    assert '"y": "hello"' in decoded["arguments"]


from omni.adaptors.vllm.tokenizer.deepseek_v32_encoding import encode_messages

def test_encode_simple_chat():
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]

    prompt = encode_messages(messages, thinking_mode="chat")

    assert "你好" in prompt
    assert "你好！" in prompt

def test_encode_thinking_mode():
    messages = [
        {"role": "user", "content": "1+1等于几？"},
        {
            "role": "assistant",
            "content": "2",
            "reasoning_content": "这是一个加法问题",
        },
    ]

    prompt = encode_messages(messages, thinking_mode="thinking")

    assert "<think>" in prompt
    assert "这是一个加法问题" in prompt
    assert "</think>" in prompt


def test_encode_with_tool_call():
    messages = [
        {"role": "user", "content": "调用工具"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": {"x": 1, "y": 2},
                    },
                }
            ],
        },
    ]

    prompt = encode_messages(messages, thinking_mode="chat")

    assert "function_calls" in prompt
    assert "add" in prompt
    assert "parameter" in prompt


from omni.adaptors.vllm.tokenizer.deepseek_v32_encoding import parse_message_from_completion_text, eos_token

def test_parse_simple_completion():
    text = "答案是 2" + eos_token

    msg = parse_message_from_completion_text(text, thinking_mode="chat")

    assert msg["role"] == "assistant"
    assert msg["content"] == "答案是 2"
    assert msg["tool_calls"] == []

def test_parse_tool_call_completion():
    text = (
        "调用工具</think>\n\n"
        "<｜DSML｜function_calls>\n"
        "<｜DSML｜invoke name=\"add\">\n"
        "<｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"y\" string=\"false\">2</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
        + eos_token
    )

    msg = parse_message_from_completion_text(text, thinking_mode="thinking")

    assert msg["reasoning_content"] == "调用工具"
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "add"


def test_thinking_summary_then_tool_not_supported():
    text = (
        "<think>调用工具</think>"
        "我需要计算\n\n"
        "<｜DSML｜function_calls>\n"
        "</｜DSML｜function_calls>"
        + eos_token
    )

    with pytest.raises(AssertionError):
        parse_message_from_completion_text(text, thinking_mode="thinking")