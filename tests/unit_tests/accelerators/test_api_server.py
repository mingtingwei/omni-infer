import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import os
from pathlib import Path
import requests
from typing import Optional
from dataclasses import dataclass

from test_proxy import setup_teardown

from vllm.entrypoints.openai import serving_engine 
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput

CUR_DIR = Path(__file__).parent
MODEL_NAME=f"{CUR_DIR}/mock_model/"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


def test_chat_completions_gptoss():
    @dataclass
    class MockHFConfig:
        model_type: str = "gpt_oss"


    @dataclass
    class MockModelConfig:
        task = "generate"
        tokenizer = MODEL_NAME
        trust_remote_code = False
        tokenizer_mode = "auto"
        max_model_len = 100
        tokenizer_revision = None
        truncation_side = 'left'
        hf_config = MockHFConfig()
        logits_processor_pattern = None
        diff_sampling_param: Optional[dict] = None
        allowed_local_media_path: str = ""
        encoder_config = None
        generation_config: str = "auto"

        def get_diff_sampling_param(self):
            return self.diff_sampling_param or {}

    with patch("vllm.entrypoints.harmony_utils.get_encoding"),\
        patch("vllm.entrypoints.openai.serving_chat.parse_chat_output"),\
        patch("vllm.entrypoints.openai.serving_chat.get_streamable_parser_for_assistant"):

        mock_engine = MagicMock(spec=MQLLMEngineClient)
        mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
        mock_output = MagicMock()
        mock_output_outputs_with_index = MagicMock()
        mock_output_outputs_with_index.index = 0
        mock_output_outputs_with_index.finish_reason = "stop"
        mock_output.outputs = [mock_output_outputs_with_index]
        mock_generator = AsyncMock()
        mock_generator.__aiter__.return_value = iter([mock_output, mock_output])
        mock_engine.generate.return_value = mock_generator
        mock_engine.errored = False

        models = OpenAIServingModels(engine_client=mock_engine,
                                    base_model_paths=BASE_MODEL_PATHS,
                                    model_config=MockModelConfig())
        serving_chat = OpenAIServingChat(mock_engine,
                                        MockModelConfig(),
                                        models,
                                        response_role="assistant",
                                        chat_template=CHAT_TEMPLATE,
                                        chat_template_content_format="auto",
                                        request_logger=None)
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": "what is 1+1?"
            }],
        )

        try:
            asyncio.run(serving_chat.create_chat_completion(req))

        except Exception as e:
            pytest.fail(f"OpenAIServing.__init__ raised unexpected exception: {e}")

        # Reset generator
        mock_generator.__aiter__.return_value = iter([mock_output, mock_output])
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": "what is 1+1?"
            }],
            stream=True,
        )

        try:
            async def test_async_output():
                response = await serving_chat.create_chat_completion(req)
                async for r in response:
                    pass

            asyncio.run(test_async_output())

        except Exception as e:
            pytest.fail(f"OpenAIServing.__init__ raised unexpected exception: {e}")


def test_tokenizer_worker_num_cannot_exceed_available_cores():
    env_vars = {
        "TOKENIZER_PROC_POOL": "1",
        "TOKENIZER_WORKER_NUM": "6",
        "TOKENIZER_PROC_POOL_THRES": "128",
        "TOKENIZER_AFFINITY_CORES": "0,1,2,3"
    }

    # Use patch.dict to temporarily overwrite os.environ
    with patch.dict(os.environ, env_vars):
        mock_engine_client = MagicMock()
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_models = MagicMock()
        mock_request_logger = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            openai_serving = serving_engine.OpenAIServing(
                engine_client=mock_engine_client,
                model_config=mock_model_config,
                models=mock_models,
                request_logger=mock_request_logger,
                return_tokens_as_token_ids=False,
            )

    assert str(exc_info.value) == "tokenizer_worker_num (6) cannot exceed available_cores (4)"

def test_api_server_health(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/health"
    response = requests.get(url, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

    try:
        file_path = "/opt/cloud/node/npu_status.yaml"
        path_exists = os.path.exists(file_path)
        if not path_exists:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                mock_yaml_content = """
        resources:
        - status: 
            - npu: 0
              errLevel: L1
        - errLevelName: NotHandle
        """
                f.write(mock_yaml_content)

        url = f"http://127.0.0.1:{prefill_port_list[0]}/health"
        response = requests.get(url, timeout=10)
        print("Status Code:", response.status_code)
        print("Response Headers:", response.headers)
        print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        assert response.status_code == 200
    finally:
        if not path_exists:
            os.remove(file_path)

def test_completions_using_process_pool(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }

    # Contruct a long enough prompt that will enable process pool
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "prompt": list("""Act as an expert creative writing instructor and narrative analyst. 
                    Generate a detailed, original fantasy world setting in 3-4 paragraphs. 
                    Include unique geographical features, a brief historical conflict, 
                    and a description of at least two distinct cultures or societies within it."""),
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

def test_chat_completions_using_proc_pool(setup_teardown):
    from test_proxy import prefill_port_list

    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    
    # Contruct a long enough prompt that will enable process pool
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", 
                        "prefix": True,
                        "content": """Generate a detailed, original fantasy world setting in 3-4 paragraphs. 
                                    Include unique geographical features, a brief historical conflict, and 
                                    a description of at least two distinct cultures or societies within it."""}],
        "stream": True,
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200
