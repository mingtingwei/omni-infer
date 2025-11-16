import os
from PIL import Image
from vllm import LLM, SamplingParams

from omni.models.deepseek.deepseek_ocr_utils.deepseek_ocr_processor import DeepseekOCRProcessor


os.environ.update({
    "GLOO_SOCKET_IFNAME": "",
    "TP_SOCKET_IFNAME": "",
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_INTRA_ROCE_ENABLE": "1",
    "HCCL_INTRA_PCIE_ENABLE": "0",
})

# User Configuration
MODEL_PATH = "/data/models/DeepSeek-OCR"

# Test requests list - each request contains (imgpath, prompt)
TEST_REQUESTS = [
    ("/your_path/image1.jpg", '<image>\n<|grounding|>Convert the document to markdown.'),
    ("/your_path/image2.png", '<image>\nFree OCR.'),
]

def create_input():
    model_input = []
    for img_path, prompt in TEST_REQUESTS:
        img = Image.open(img_path).convert("RGB")
        model_input.append({"prompt": prompt, "multi_modal_data": {"image": img}})
    
    return model_input

model_input = create_input()

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
    trust_remote_code=True,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=True,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    max_tokens=8192,
    temperature=0.0,
    skip_special_tokens=False,
)

# Generate outputs
model_outputs = llm.generate(model_input, sampling_params=sampling_params)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)
