from vllm.entrypoints.openai import tool_parsers
from .pangu_tool_parser import PanguToolParser
from .openai_tool_parser import OpenAIToolParser


def register_tool():
    tool_parsers.__all__.append("PanguToolParser")
    tool_parsers.__all__.append("OpenAIToolParser")
