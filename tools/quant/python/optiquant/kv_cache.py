import os
import json
import torch
import logging
from safetensors.torch import load_file, save_file

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Constants
SCALE_MAX_VALUE = 127.0
SCALE_CLAMP_MIN = 1e-5


def cal_scale(faquant_path, layer_idx, method="max"):
    """Calculate scale for KV cache quantization.

    Args:
        faquant_path: Path to calibration data directory.
        layer_idx: Layer index to process.
        method: Scaling method ('max' supported).

    Returns:
        Calculated scale tensor.

    Raises:
        ValueError: If no matching calibration files found.
    """
    tensors = []

    # Iterate through all .pth files matching the layer index
    for fname in os.listdir(faquant_path):
        if fname.endswith("_%s.pth" % layer_idx):
            fpath = os.path.join(faquant_path, fname)
            t = torch.load(fpath, map_location="cpu")

            if isinstance(t, torch.Tensor):
                tensors.append(t)
            elif isinstance(t, dict):
                # If file contains dict, extract first tensor
                for v in t.values():
                    if isinstance(v, torch.Tensor):
                        tensors.append(v)
                        break

    if not tensors:
        raise ValueError(
            "No tensors found matching _%s.pth pattern in %s"
            % (layer_idx, faquant_path)
        )

    merged = torch.cat(tensors, dim=0)

    if method == "max":
        scale = (merged.max() / SCALE_MAX_VALUE).clamp(min=SCALE_CLAMP_MIN).cpu()
    else:
        raise ValueError("Unsupported scaling method: %s" % method)

    return scale


def main(args, model_path, faquant_path, kvs_safetensor_name, layer_num=62):
    """Main function to apply KV cache quantization.

    Args:
        args: Command line arguments.
        model_path: Path to model directory.
        faquant_path: Path to calibration data.
        kvs_safetensor_name: Name for KV scale safetensor file.
        layer_num: Number of layers in the model.
    """
    model_config = os.path.join(model_path, "model.safetensors.index.json")

    with open(model_config, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]
    faquant_scale = {}

    for layer_idx in range(layer_num):
        kvs = cal_scale(faquant_path, layer_idx)
        logger.info("The KV scale of layer_idx=%s is %s", layer_idx, kvs)

        scale_key = "model.layers.%s.self_attn.kv_scale" % layer_idx
        faquant_scale[scale_key] = kvs
        weight_map[scale_key] = kvs_safetensor_name

    file = os.path.join(model_path, kvs_safetensor_name)

    if not os.path.exists(file):
        init_dict = {}
        save_file(init_dict, file)

    state_dict = load_file(file)
    model_index["weight_map"] = weight_map
    save_file(state_dict, file + ".bak")
    state_dict.update(faquant_scale)
    save_file(state_dict, file)

    with open(model_config, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    logger.info("KV scales saved to %s", file)
