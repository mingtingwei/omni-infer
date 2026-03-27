import os
import json
import shutil
from glob import glob
from tqdm import tqdm
import torch
import logging

try:
    import torch_npu
except ImportError:
    pass

from safetensors.torch import load_file, save_file
from optiquant.optiquant_utils import (
    QType,
    quant_ssz,
    weight_quant,
    pack_4bit,
    unpack_from_int32,
    INT8_QMAX,
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
NUM_LAYERS = 62
BF16_ELEMENT_SIZE = 2
GROUP_SIZE = 32
NUM_EXPERTS = 384

# Weight names to quantize to INT8
INT8_WEIGHT_NAMES = []
# Weight names to quantize to INT4
INT4_WEIGHT_NAMES = []


def _build_weight_names():
    """Build lists of INT8 and INT4 weight names for all layers.

    Populates INT8_WEIGHT_NAMES and INT4_WEIGHT_NAMES module-level lists.
    """
    global INT8_WEIGHT_NAMES, INT4_WEIGHT_NAMES

    INT8_WEIGHT_NAMES = []
    INT4_WEIGHT_NAMES = []

    for i in range(NUM_LAYERS):
        # MLP layers - INT8
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.gate_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.up_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.down_proj.weight" % i)

        # Attention linear layers - INT8
        INT8_WEIGHT_NAMES.append("model.layers.%s.self_attn.q_a_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.self_attn.kv_a_proj_with_mqa.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.self_attn.q_b_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.self_attn.o_proj.weight" % i)

        # MoE shared experts - INT8
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.shared_experts.gate_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.shared_experts.up_proj.weight" % i)
        INT8_WEIGHT_NAMES.append("model.layers.%s.mlp.shared_experts.down_proj.weight" % i)

        # MoE routed experts - INT4
        for j in range(NUM_EXPERTS):
            INT4_WEIGHT_NAMES.append(
                "model.layers.%s.mlp.experts.%s.gate_proj.weight_packed" % (i, j)
            )
            INT4_WEIGHT_NAMES.append(
                "model.layers.%s.mlp.experts.%s.up_proj.weight_packed" % (i, j)
            )
            INT4_WEIGHT_NAMES.append(
                "model.layers.%s.mlp.experts.%s.down_proj.weight_packed" % (i, j)
            )


def copy_config_files(bf16_path, int8_path):
    """Copy non-safetensor files from source to destination.

    Args:
        bf16_path: Source directory.
        int8_path: Destination directory.
    """
    files_to_copy = [
        f for f in os.listdir(bf16_path)
        if not f.endswith(".safetensors") and not f.startswith(".")
    ]

    for file_name in files_to_copy:
        src_path = os.path.join(bf16_path, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(int8_path, file_name)
            shutil.copy2(src_path, dst_path)


def main(args, bf16_path, output_path, model_name="deepseek-ai/DeepSeek-R1"):
    """Main function to convert per-group INT4 to per-channel INT4.

    Args:
        args: Command line arguments.
        bf16_path: Path to BF16 weight directory.
        output_path: Path for quantized output.
        model_name: HuggingFace model repository name.
    """
    # Build weight name lists if not already done
    if not INT8_WEIGHT_NAMES:
        _build_weight_names()

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    copy_config_files(bf16_path, output_path)

    w4_type = QType(args.qtype)

    model_index_file = os.path.join(output_path, "model.safetensors.index.json")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]
    scale_count = len([key for key in weight_map.keys()])

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    quant_count = 0
    new_weight_map = {}

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        file_name = file_name.replace("model", QUANT_PREFIX)

        state_dict = load_file(safetensor_file, device=args.device)
        new_state_dict = {}

        for weight_name, weight in state_dict.items():
            if "weight_scale" in weight_name or "weight_shape" in weight_name:
                continue
            elif weight_name in INT8_WEIGHT_NAMES:
                assert weight.element_size() == BF16_ELEMENT_SIZE, \
                    "Expected BF16 weight with element_size %s" % BF16_ELEMENT_SIZE

                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_scale_name = weight_name + "_scale"
                new_state_dict[new_scale_name] = scale_inv

                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name

            elif weight_name in INT4_WEIGHT_NAMES:
                weight_name_base = weight_name[:-7]

                assert weight.element_size() == 4, \
                    "Expected weight element_size 4, got %s" % weight.element_size()

                quant_count += 1

                unpacked_int4 = unpack_from_int32(
                    value=weight,
                    num_bits=4,
                    shape=torch.Size([weight.shape[0], weight.shape[1] * 8]),
                    packed_dim=1
                )

                scale_name = weight_name_base + "_scale"
                weight_scale = state_dict[scale_name]
                scale_expanded = weight_scale.repeat_interleave(GROUP_SIZE, dim=1)
                weight_dequant = unpacked_int4.to(torch.bfloat16) * scale_expanded

                int4_weight, int4_scale, bias = quant_ssz(
                    weight_dequant, w4_type, -1, w4=True
                )

                new_state_dict[weight_name_base] = pack_4bit(int4_weight)
                new_scale_int4 = weight_name_base + "_int4_scale"
                new_bias = weight_name_base + "_bias"

                new_state_dict[new_scale_int4] = int4_scale
                new_state_dict[new_bias] = bias

                new_weight_map[weight_name_base] = file_name
                new_weight_map[new_scale_int4] = file_name
                new_weight_map[new_bias] = file_name

                logger.info("%s int4 %s", weight_name_base, int4_weight.shape)

            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        del state_dict
        del new_state_dict

    logger.info("%s %s", quant_count, scale_count)
    logger.info("%s weights are quantized", quant_count)

    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    model_index["weight_map"] = new_weight_map

    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    logger.info("Model index saved to %s", model_index_file)
