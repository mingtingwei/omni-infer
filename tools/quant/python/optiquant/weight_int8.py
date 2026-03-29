import os
import json
import logging
from glob import glob
from tqdm import tqdm
import torch

try:
    import torch_npu
except ImportError:
    pass

from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
from optiquant.optiquant_utils import (
    build_disable_names,
    build_disable_names_pangu,
    weight_quant,
    NUM_LAYERS,
    BF16_ELEMENT_SIZE,
)

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Constants
QUANT_PREFIX = "quant_model_weight_w8a8_dynamic"


def main(args, bf16_path, output_path, pangu_mode, model_name="deepseek-ai/DeepSeek-R1"):
    """Main function to perform INT8 quantization.

    Args:
        args: Command line arguments.
        bf16_path: Path to BF16 weight directory.
        output_path: Path for quantized output.
        pangu_mode: Whether to apply Pangu-specific exclusions.
        model_name: HuggingFace model repository name.
    """
    disable_names = build_disable_names_pangu() if pangu_mode else build_disable_names()

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    model_index_file = os.path.join(output_path, "model.safetensors.index.json")

    if not os.path.exists(model_index_file):
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.safetensors"],
            local_dir=output_path,
            local_dir_use_symlinks=False
        )
        logger.info("Model index and config downloaded to %s", output_path)

    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    quant_count = 0
    new_weight_map = {}

    for safetensor_file in tqdm(safetensor_files, desc="INT8 quantization"):
        file_name = os.path.basename(safetensor_file)
        file_name = file_name.replace("model", QUANT_PREFIX)

        state_dict = load_file(safetensor_file, device=args.device)
        new_state_dict = {}

        for weight_name, weight in state_dict.items():
            if weight_name in disable_names:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
                continue

            scale_inv_name = f"{weight_name}_scale_inv"

            if scale_inv_name in weight_map or pangu_mode:
                if weight.element_size() != BF16_ELEMENT_SIZE:
                    raise ValueError(
                        "Expected BF16 weight with element_size %s, got %s"
                        % (BF16_ELEMENT_SIZE, weight.element_size())
                    )

                quant_count += 1
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight

                new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

    logger.info("%s weights are quantized", quant_count)

    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    model_index["weight_map"] = new_weight_map

    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    logger.info("Model index saved to %s", model_index_file)
