from vllm import reasoning
from .pangu_reasoning_parser import PanguReasoningParser
from .kimi2_thinking_reasoning_parser import Kimi2ThinkingReasoningParser


def register_reasoning():
    reasoning.__all__.append("PanguReasoningParser")
    reasoning.__all__.append("Kimi2ThinkingReasoningParser")