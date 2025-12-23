from omni.accelerators.reasoning_compression.config import ThinkCompressDict

def init_reasoner_compression_configs(vllm_config):
    if vllm_config is None or vllm_config.additional_config is None:
        ThinkCompressDict.reasoner_early_think_stopping_enabled = 0
        return
    additional_config = vllm_config.additional_config
    enable_think_stop = additional_config.get("reasoner_early_think_stopping_enabled", False)
    if enable_think_stop is False:
        ThinkCompressDict.reasoner_early_think_stopping_enabled = 0
        return

    think_stop_config = vllm_config.additional_config.get("reasoner_early_think_stopping_config", {})
    if think_stop_config is None or len(think_stop_config) == 0:
        raise RuntimeError(
            "reasoner_early_think_stopping_config is required when enable early think stop.")
    if "reasoner_early_think_stopping_string" not in think_stop_config or "reasoner_early_think_stopping_tags" not in think_stop_config or "reasoner_early_think_stopping_step" not in think_stop_config:
        raise RuntimeError(
            "reasoner_early_think_stopping_string, reasoner_early_think_stopping_tags and reasoner_early_think_stopping_step is required when enable early think stop.")
    else:
        ThinkCompressDict.reasoner_early_think_stopping_enabled = 1
        ThinkCompressDict.reasoner_early_think_stopping_think_start_string = think_stop_config.get("reasoner_early_think_stopping_think_start_string", "")
        ThinkCompressDict.reasoner_early_think_stopping_step = think_stop_config.get("reasoner_early_think_stopping_step")
        ThinkCompressDict.reasoner_early_think_stopping_string = think_stop_config.get("reasoner_early_think_stopping_string")
        ThinkCompressDict.reasoner_early_think_stopping_tags = think_stop_config.get("reasoner_early_think_stopping_tags")
