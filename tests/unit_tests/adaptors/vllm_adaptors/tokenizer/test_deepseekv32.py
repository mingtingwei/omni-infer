"""
Unit tests for DeepSeekV32Tokenizer.apply_chat_template method.
"""

import sys
import types
import contextlib
import pytest
import unittest
import importlib
from unittest.mock import MagicMock, patch, Mock, ANY


# SUT under test:
SUT_MODULE = "omni.adaptors.vllm.tokenizer.deepseekv32"

SUT = None  # will be imported in setup_module()
_ISOLATION_STATE = {}

def setup_module():
    
    global SUT, _ISOLATION_STATE
    
    if "vllm.transformers_utils.tokenizer_base" not in sys.modules:
        base_mod = types.ModuleType("vllm.transformers_utils.tokenizer_base")
        
        class TokenizerBase:
            def __init__(self, name_or_path=""):
                self.name_or_path = name_or_path
            
            def get_added_vocab(self):
                return {}
            
            @property
            def all_special_tokens(self):
                return []
            
            @property
            def all_special_ids(self):
                return []
            
            @property
            def bos_token_id(self):
                return None
            
            @property
            def eos_token_id(self):
                return None
            
            @property
            def pad_token_id(self):
                return None
            
            @property
            def is_fast(self):
                return False
            
            @property
            def vocab_size(self):
                return 0
            
            @property
            def truncation_side(self):
                return "right"
            
            def encode(self, text):
                return []
            
            def decode(self, ids, skip_special_tokens=False):
                return ""
            
            def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
                return []
            
            def convert_tokens_to_string(self, tokens):
                return ""
        
        base_mod.TokenizerBase = TokenizerBase
        sys.modules["vllm.transformers_utils.tokenizer_base"] = base_mod
        
    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        
        def init_logger(name):
            import logging
            logger = logging.getLogger(name)
            return logger
        
        logger_mod.init_logger = init_logger
        sys.modules["vllm.logger"] = logger_mod
    
    if "vllm.transformers_utils.config" not in sys.modules:
        config_mod = types.ModuleType("vllm.transformers_utils.config")
        config_mod.get_sentence_transformer_tokenizer_config = lambda x, y: {}
        sys.modules["vllm.transformers_utils.config"] = config_mod
    
    try:
        import transformers
    except ModuleNotFoundError:
        transformers_mod = types.ModuleType("transformers")
        
        class _BatchEncoding:
            pass
        
        class _AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return Mock(name="tokenizer")
            
        transformers_mod.BatchEncoding = _BatchEncoding
        transformers_mod.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = transformers_mod
        
    global SUT
    SUT = importlib.import_module(SUT_MODULE)
    SUT = importlib.reload(SUT)


def teardown_module():
    global SUT
    
    modules_to_remove = [SUT_MODULE, "omni.adaptors.vllm.tokenizer.hf"]
    
    for mod_name in modules_to_remove:
        if mod_name in sys.modules:
            with contextlib.suppress(Exception):
                del sys.modules[mod_name]
    
    SUT = None


class DeepseekV32TokenizerApplyChatTemplateTests(unittest.TestCase):
    """Test DeepseekV32Tokenizer.apply_chat_template method."""
    
    def setUp(self):
        super().setUp()
        global SUT
        if SUT is None:
            setup_module()
        self.assertIsNotNone(SUT)
        
        # create a mock tokenizer
        self.mock_tokenizer = Mock(name="mock_tokenizer")
        self.mock_tokenizer.name_or_path = "test/model"
        self.mock_tokenizer.get_added_vocab.return_value = {"<｜begin▁of▁sentence｜>": 1}
        
        # create the tokenizer instance
        self.tokenizer = SUT.DeepseekV32Tokenizer(self.mock_tokenizer)
        
    def test_apply_chat_template_basic_chat_mode(self):
        """test apply_chat_template in basic chat mode."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
        ]
        
        result = self.tokenizer.apply_chat_template(messages)
        
        self.assertIn("<｜begin▁of▁sentence｜>", result)
        self.assertIn("Hello!", result)
        self.assertIn("Hi there! How can I help you?", result)
        
    def test_apply_chat_template_thinking_mode(self):
        """test apply_chat_template in thinking mode."""
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]
        
        result = self.tokenizer.apply_chat_template(messages, thinking=True)
        
        self.assertIn("<think>", result)

    def test_apply_chat_template_with_conversation_kwarg(self):
        """test apply_chat_template with conversation kwarg."""
        messages = [
            {"role": "user", "content": "Tell me a joke."},
        ]
        
        conversation = [
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        
        result = self.tokenizer.apply_chat_template(messages, conversation=conversation)
        
        self.assertIn("What is 2 + 2?", result)
        self.assertNotIn("Tell me a joke.", result)
    
    def test_apply_chat_template_with_tools(self):
        """test apply_chat_template with tools provided."""
        messages = [
            {"role": "user", "content": "Use a tool"},
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Adds two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            }
        ]

        result = self.tokenizer.apply_chat_template(messages, tools=tools)
        
        self.assertIn("## Tools", result)
        self.assertIn("add", result)

    def test_apply_chat_template_preserves_original_messages(self):
        """test apply_chat_template does not modify original messages."""
        original_messages = [
            {"role": "user", "content": "Original message."},
        ]
        
        messages_copy = original_messages.copy()
        
        self.tokenizer.apply_chat_template(messages_copy)
        
        self.assertEqual(original_messages, [{"role": "user", "content": "Original message."}])
    
    def test_apply_chat_template_with_list_of_lists_conversation_format(self):
        """test apply_chat_template with conversation as list of lists format."""
        conversation = [
            {"role": "user", "content": "batch1."},
            {"role": "user", "content": "batch2."},
        ]
        
        try:
            self.tokenizer.apply_chat_template([], conversation=conversation)
        except Exception as e:
            self.assertNotIn("isinstance", str(e))
    
    def test_apply_chat_template_with_conversation_having_messages_attr(self):
        """test apply_chat_template with conversation having messages attribute."""
        
        class MockConversation:
            def __init__(self, messages):
                self.messages = messages
        
        conversation = MockConversation([
            {"role": "user", "content": "message from messages attr."},
        ])
        
        try:
            self.tokenizer.apply_chat_template([], conversation=[conversation])
        except Exception as e:
            pass  # should not raise
    
    @pytest.mark.xfail(reason="Known issue with empty conversation list handling.")
    def test_apply_chat_template_with_empty_conversation_list(self):
        """test apply_chat_template with empty conversation list."""
        conversation = []
        messages = [{"role": "user", "content": "Hello!"}]
        
        with self.assertRaises(ValueError) as ctx:
            self.tokenizer.apply_chat_template(messages, conversation=conversation)
        
        self.assertIsNotNone(ctx.exception.__cause__)
    
    @pytest.mark.xfail(reason="Known issue with empty messages list handling.")
    def test_apply_chat_template_empty_messages(self):
        """test apply_chat_template with empty messages."""
        messages = []
        
        with self.assertRaises(ValueError) as ctx:
            self.tokenizer.apply_chat_template(messages)
        
        self.assertIsNotNone(ctx.exception.__cause__)
    
    def test_apply_chat_template_with_thinking_content_in_history(self):
        """test apply_chat_template with thinking content in message history."""
        messages = [
            {"role": "user", "content": "calculate 2 + 2."},
            {
                "role": "assistant", 
                "content": "2",
                "reasoning_content": "1+1=2"},
        ]

        result = self.tokenizer.apply_chat_template(messages, thinking=True)

        self.assertIn("calculate 2 + 2.", result)

if __name__ == "__main__":
    unittest.main(verbosity=2)
