import os
import json
import logging
from argparse import ArgumentParser

import optiquant.weight_int8 as weight_int8
import optiquant.weight_int4 as weight_int4
import optiquant.int4_group_to_channel as int4_g2c
import optiquant.kv_cache as kv_cache
from optiquant.optiquant_utils import NUM_LAYERS

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
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1"
"""
sszs50g0a0b4sym1
     │ ││ ││ │
     │ ││ ││ └── 1 = symmetric
     │ ││ │└───── b4 = 4 bits
     │ ││ └────── a0 = activation integer (0 means not integer)
     │ │└──────── g0 = groupsize=0 (Per-Channel!)
     │ └───────── 0 = offset
     └─────────── 50 = num_step (迭代次数)
"""
DEFAULT_QTYPE = "sszs50g0a0b4sym1"
GLOBAL_COMPRESSION_RATIO = 1.5943962512751309

# Quantization bits for different layer types
NUM_BITS_CONFIG = {
    "self_attn.kv_a_proj_with_mqa": 8,
    "self_attn.q_a_proj": 8,
    "self_attn.q_b_proj": 8,
    "self_attn.o_proj": 8,
    "mlp.down_proj": 8,
    "mlp.gate_up_proj": 8,
    "mlp.shared_experts": 8,
    "mlp.experts": 4,
}

# Layer patterns to ignore in quantization
IGNORE_PATTERNS = [
    "self_attn.kv_b_proj",
]


def build_ignore_list():
    """Build list of layers to ignore during quantization.

    Returns:
        List of layer names to ignore.
    """
    ignores = []
    for i in range(NUM_LAYERS):
        for pattern in IGNORE_PATTERNS:
            ignores.append("model.layers.%s.%s" % (i, pattern))
    return ignores


def build_quant_config(num_bits):
    """Build quantization configuration dictionary.

    Args:
        num_bits: Either an int for all weights, or dict mapping layer types to bit widths.

    Returns:
        Quantization configuration dict for config.json.
    """
    ignores = build_ignore_list()

    quant_config = {
        "config_groups": {"group_0": {}},
        "format": "int-quantized",
        "global_compression_ratio": GLOBAL_COMPRESSION_RATIO,
        "ignore": ignores,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    quant_config["config_groups"]["group_0"]["input_activations"] = {
        "actorder": None,
        "block_structure": None,
        "dynamic": True,
        "group_size": None,
        "num_bits": 8,
        "observer": "memoryless",
        "observer_kwargs": {},
        "strategy": "token",
        "symmetric": True,
        "type": "int",
    }

    quant_config["config_groups"]["group_0"]["output_activations"] = None
    quant_config["config_groups"]["group_0"]["targets"] = ["Linear"]
    quant_config["config_groups"]["group_0"]["weights"] = {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": None,
        "num_bits": num_bits,
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": "channel",
        "symmetric": True,
        "type": "int",
    }

    return quant_config


def main():
    """Main entry point for DeepSeek/Kimi2 model quantization."""
    parser = ArgumentParser()

    parser.add_argument(
        "--input-bf16-hf-path",
        type=str,
        required=True,
        help="bf16 weight path"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="quantized weight path"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="support cpu and npu"
    )
    parser.add_argument(
        "--file_count",
        type=int,
        default=0,
        help="File count when loading model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Huggingface repo name"
    )

    parser.add_argument(
        "--pangu-mode",
        default=False,
        action="store_true",
        help="pangu mode"
    )
    parser.add_argument(
        "--w4",
        default=False,
        action="store_true",
        help="int4 quantization flag"
    )
    parser.add_argument(
        "--pergroup-to-perchannel",
        default=False,
        action="store_true",
        help="pergroup-to-perchannel"
    )
    parser.add_argument(
        "--qtype",
        type=str,
        default=DEFAULT_QTYPE,
        help="quantization config. only support sszs50g0a0b4sym1 now"
    )
    parser.add_argument(
        "--c8-calib-path",
        type=str,
        default=None,
        help="mla c8 calibration data path"
    )
    parser.add_argument(
        "--kvs-safetensor-name",
        type=str,
        default=None,
        help="mla c8 (faquant) safetensor name"
    )

    args = parser.parse_args()

    # KV cache quantization
    if args.c8_calib_path is not None:
        logger.info("Starting KV cache quantization")
        kv_cache.main(
            args.output_path,
            args.c8_calib_path,
            args.kvs_safetensor_name
        )

    # Weight quantization
    if args.w4:
        logger.info("Using INT4 weight quantization")
        if args.pergroup_to_perchannel:
            logger.info("Converting per-group to per-channel")
            int4_g2c.main(
                args,
                args.input_bf16_hf_path,
                args.output_path,
                args.model_name
            )
        else:
            weight_int4.main(
                args,
                args.input_bf16_hf_path,
                args.output_path,
                args.pangu_mode,
                args.model_name
            )
        num_bits = NUM_BITS_CONFIG
    else:
        logger.info("Using INT8 weight quantization")
        weight_int8.main(
            args,
            args.input_bf16_hf_path,
            args.output_path,
            args.pangu_mode,
            args.model_name
        )
        num_bits = 8

    # Generate and save quantization config
    logger.info("Building quantization configuration")
    quant_config = build_quant_config(num_bits)

    config_path = os.path.join(args.output_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info("Quantization config saved to %s", config_path)


if __name__ == "__main__":
    main()
