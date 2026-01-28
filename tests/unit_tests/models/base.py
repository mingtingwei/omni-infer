import torch
import logging
import weakref
from typing import Dict
import numpy as np
from unittest.mock import MagicMock, Mock

from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder)
from vllm.config import set_current_vllm_config, get_layers_from_vllm_config, CompilationLevel, KVTransferConfig
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.attention import AttentionType, get_attn_backend
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
from vllm.utils import is_pin_memory_available, supports_dynamo
from vllm.platforms import current_platform
from vllm.v1.utils import bind_kv_cache
from vllm.model_executor.model_loader.utils import set_default_torch_dtype, process_weights_after_loading, configure_quant_config
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.block_table import BlockTable
from omni.adaptors.vllm.forward_context import set_forward_context
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder
from omni.accelerators.reasoning_compression.config import ThinkCompressDict
from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration, mark_static_for_graph_default

__origin_get_device_properties__ = torch.npu.get_device_properties
class NPUDeviceProperties:
    def __init__(self, device):
        self.properties = __origin_get_device_properties__(device)
        self.multi_processor_count = self.properties.multi_processor_count \
            if hasattr(self.properties, 'multi_processor_count') else 0

class FakeKVCacheGroup:
    def __init__(self, kv_cache_spec, layer_names):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
class FakeKVCacheConfig:
    def __init__(self, kv_cache_spec):
        self.kv_cache_groups = [
            FakeKVCacheGroup(
                kv_cache_spec=kv_cache_spec,
                layer_names=[]
            )
        ]

def get_device_properties(device):
    return NPUDeviceProperties(device)

class MockRunner:
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.kv_caches: list[torch.Tensor] = []
        self.attn_metadata_builders: list[AttentionMetadataBuilder] = []
        self.attn_backends: list[type[AttentionBackend]] = []
        self.device = device
        torch.npu.set_device(device)
        self.dtype = vllm_config.model_config.dtype
        self.is_hybrid_chunked_prefill_graph_mode = False
        self.block_size = vllm_config.cache_config.block_size
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs
        self.speculative_config = vllm_config.speculative_config
        self.uses_mrope = False
        self.use_spec_decode = False
        if self.speculative_config:
            self.use_spec_decode = True
        self.omni_cache = None
        self.attn_mask = None
        self.attn_state = None
        self.model_mark_static = False
        self.dummy_model_mark_static = False
        self.drafter_mark_static = False
        self.dummy_drafter_mark_static = False
        self.pin_memory = is_pin_memory_available()
        self.decode_gear_list = self.vllm_config.npu_compilation_config.decode_gear_list
        self.max_batch_size = self.vllm_config.scheduler_config.max_batch_size
        if self.use_spec_decode:
            if self.decode_gear_list is None or len(self.decode_gear_list) == 0:
                raise RuntimeError("When enable adaptive speculative decoding, decode_gear_list must be set.")


        self.max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.req_cnt = 0
        self.req_prefix = 'chatcmpl-req-'
        self.requests: dict[str, CachedRequestState] = {}
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=is_pin_memory_available())
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=is_pin_memory_available())
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int64,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)
        self.graph_block_tables = np.zeros(
            (self.max_batch_size,
            (self.model_config.max_model_len + self.block_size - 1) // self.block_size),
            dtype=np.int32)
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int64,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        ThinkCompressDict.reasoner_early_think_stopping_enabled = 0  
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.hf_text_config.vocab_size,
            block_size=self.block_size,
        )
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )

        self.enable_torchair_graph_mode = (
                    self.vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())

        self.dcp_kv_cache_interleave_size = self.block_size
        logging.info(f"MockRunner: enable_torchair_graph_mode: {self.enable_torchair_graph_mode}")

        torch.cuda.get_device_properties = get_device_properties

    def get_new_req_id(self):
        req_id = self.req_prefix + str(self.req_cnt)
        self.req_cnt += 1
        return req_id

    def init_model(self, model_cls):
        device_config = self.vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(self.vllm_config.model_config.dtype):
            with target_device:
                with set_current_vllm_config(self.vllm_config, check_compile=False):
                    self.model = model_cls(vllm_config=self.vllm_config)
            # quantization
            if self.model_config.hf_text_config.quantization_config is not None:
                configure_quant_config(self.model_config.hf_text_config.quantization_config, model_cls)
                
            process_weights_after_loading(self.model, self.vllm_config.model_config, target_device)
        self.model.eval()
        return self.model

    def init_attn_backends(self):
        for i, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
            attn_backend_i = get_attn_backend(
                kv_cache_spec.head_size,
                self.vllm_config.model_config.dtype,
                kv_cache_spec.dtype,
                kv_cache_spec.block_size,
                self.vllm_config.model_config.is_attention_free,
                use_mla=kv_cache_spec.use_mla,
            )
            if attn_backend_i is None:
                error_msg = (
                    f"Error with get_attn_backend: {kv_cache_spec.head_size=}, "
                    f"{self.vllm_config.model_config.dtype=}, {kv_cache_spec.dtype=}, "
                    f"{kv_cache_spec.block_size=}, "
                    f"{self.vllm_config.model_config.is_attention_free=}, "
                    f"{kv_cache_spec.use_mla=}")
                logging.error(error_msg)
            assert attn_backend_i is not None, "Non-Attention backend is not supported by V1."

            logging.info(f"[init_attn_backends] {self.vllm_config.compilation_config.full_cuda_graph=}")
            if self.vllm_config.compilation_config.full_cuda_graph:
                attn_backend_name = attn_backend_i.__name__
                flash_attn_version = get_flash_attn_version()
                assert not (attn_backend_name != "FlashAttentionBackend" or flash_attn_version != 3), (
                            f"full_cuda_graph is only supported with "
                            f"FA3. Current attention backend is "
                            f"{attn_backend_name}, FlashAttention version is "
                            f"{flash_attn_version}.")

            block_table_i = self.input_batch.block_table[i]
            attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
                weakref.proxy(self), kv_cache_spec, block_table_i)
            self.attn_backends.append(attn_backend_i)
            self.attn_metadata_builders.append(attn_metadata_builder_i)
        logging.info(f"[init_attn_backends] {self.attn_backends=}")
        logging.info(f"[init_attn_backends] {self.attn_metadata_builders=}")

    def get_kv_cache_groups(self):
        grouped_layer_names = [list(self.kv_cache_spec.keys())]
        self.kv_cache_group_specs = create_kv_cache_group_specs(self.kv_cache_spec, grouped_layer_names)
        logging.info(f"return {self.kv_cache_group_specs=}")

    def initialize_kv_cache(self):
        # TODO:
        num_blocks = 128
        kv_caches: Dict[str, torch.Tensor] = {}

        for i, kv_cache_group in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
                # adapted for Pangu 72Bv2
                hf_config = self.vllm_config.model_config.hf_config
                v_channels = getattr(hf_config, "v_channels", None)
                if v_channels is None:
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, 
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, 
                        kv_cache_spec.head_size)
                else:
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, 
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, 
                        kv_cache_spec.head_size,
                        v_channels)

                kv_caches[layer_name] = self.attn_backends[i].init_kv_cache_each_layer(
                    kv_cache_shape, 
                    self.vllm_config.model_config.dtype,
                    self.device,
                    self.vllm_config.model_config,
                    False)

        # bind kv cache to Attention and runner.kv_caches
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def init_kv_cache(self):
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        self.kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            logging.info(f"{layer_name=}, {attn_module=}, {attn_module.attn_type=}, {attn_module.sliding_window=}")
            assert attn_module.attn_type == AttentionType.DECODER
            if attn_module.sliding_window is not None:
                self.kv_cache_spec[layer_name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.vllm_config.model_config.dtype,
                    sliding_window=attn_module.sliding_window,
                    use_mla=use_mla)
            else:
                self.kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.vllm_config.model_config.dtype,
                    use_mla=use_mla)
            logging.info(f"{layer_name=}, {self.kv_cache_spec[layer_name].type_id=}, {self.kv_cache_spec[layer_name].page_size_bytes=}")
        self.kv_cache_config = FakeKVCacheConfig(self.kv_cache_spec)
        # get Dict[str, KVCacheSpec]
        logging.info(f"return {self.kv_cache_spec=}")
        self.get_kv_cache_groups()
        self.init_attn_backends()
        # alloc kv cache tensor
        self.initialize_kv_cache()

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int, total_steps: int = 1):
        input_ids, inputs_embeds = self.input_ids[:num_tokens], None
        intermediate_tensors = None
        positions = self.positions[:num_tokens]

        # No kv_caches: profile run
        if not self.kv_caches:
            logging.debug("Start running profile dummy.")
            with set_forward_context(None, self.vllm_config):
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            return forward_results
        fake_input = torch.zeros(self.max_batch_size, dtype=input_ids.dtype, device=input_ids.device)
        fake_positions = torch.zeros(self.max_batch_size, dtype=input_ids.dtype, device=input_ids.device)
        input_ids, positions = fake_input, fake_positions
        self.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            builder = self.attn_metadata_builders[kv_cache_group_id]
            assert isinstance(builder, DummyAttentionMetadataBuilder), f"{builder} does not implement DummyAttentionMetadataBuilder"
            attn_metadata_i = builder.build_dummy(num_tokens, self.max_batch_size)
            if self.enable_torchair_graph_mode:
                builder.mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i
        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata,
            "selected_indices": None
        }
        with set_forward_context(attn_metadata, self.vllm_config):
            use_compile = self.enable_torchair_graph_mode
            for _ in range(total_steps):
                if use_compile:
                    logging.debug("Start running dummy compiled model.")
                    if not self.dummy_model_mark_static:
                        if isinstance(self.model, GraphCompileConfiguration):
                            self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                        else:
                            mark_static_for_graph_default(input_ids, inputs_embeds, positions, self.kv_caches)
                        self.dummy_model_mark_static = True
                else:
                    logging.debug("Start running dummy eager model.")
                logging.debug(f"Start running dummy {input_ids.shape=}, {positions.shape=}")
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs
                )

        return forward_results

    @torch.inference_mode()
    def forward_prefill(self, prompt_token_ids, max_num_tokens):
        num_token = len(prompt_token_ids)
        print(f"forward_prefill {num_token=}")
        req_id = self.get_new_req_id()
        self.req_id = req_id
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
            max_tokens=1,
            detokenize=False,
        )
        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_inputs=[],
            mm_positions=[],
            sampling_params=sampling_params,
            generator=None,
            block_ids=[[1]],
            num_computed_tokens=0,
            output_token_ids=[],
            lora_request=None,
        )
        req_state = self.requests[req_id]
        self.input_batch.add_request(req_state, None)

        if self.vllm_config.kv_transfer_config is None:
            self.vllm_config.kv_transfer_config = KVTransferConfig()
        self.vllm_config.kv_transfer_config.kv_role = "kv_producer"
        scheduler_output = Mock()
        scheduler_output.num_scheduled_tokens = {
            req_id: num_token for req_id in self.input_batch.req_ids
        }
        scheduler_output.scheduled_spec_decode_tokens = set()
        scheduler_output.total_num_scheduled_tokens = num_token
        batch_reordered = self.attn_metadata_builders[0].reorder_batch(
            self.input_batch, scheduler_output)
        for i in range(1, len(self.kv_cache_group_specs)):
            assert not self.attn_metadata_builders[i].reorder_batch(
                self.input_batch, scheduler_output)

        num_reqs = self.input_batch.num_reqs
        self.input_batch.block_table.commit(num_reqs)
        num_scheduled_tokens = np.array([
            num_token
        ], dtype=np.int32)
        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
        arange = self.arange_np[:num_token] - cumsums_offsets
        positions_np = self.positions_np[:num_token]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        self.positions[:num_token].copy_(
            self.positions_cpu[:num_token], non_blocking=True)
        positions = self.positions[:num_token]

        self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[kv_cache_group_id]
            # NOTE(runze): since each request has at most M blocks, the offset is at most M-1
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                np.minimum(positions_np // block_size, block_table.max_num_blocks_per_req - 1))
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:num_token])

        self.attn_state = AscendAttentionState.PrefillNoCache

        graph_pad_size = 0
        extra_builder_kwargs = {'graph_pad_size': 0}

        # build attention metadata
        attn_metadata = {}
        self.full_attn_metadata = None
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            # Prepare for cascade attention if enabled & beneficial.
            attn_metadata_i = self.attn_metadata_builders[kv_cache_group_id].build(
                num_reqs=num_reqs,
                num_actual_tokens=num_token,
                max_query_len=num_token,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
            if kv_cache_group_id == 0:
                self.full_attn_metadata = attn_metadata_i

            if not isinstance(self.attn_metadata_builders[kv_cache_group_id], DummyAttentionMetadataBuilder):
                raise ValueError(f"{self.attn_metadata_builders[kv_cache_group_id]} does not implement DummyAttentionMetadataBuilder")
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        # Prepare input_ids
        token_indices = (positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:num_token])

        # Copy the tensors to the NPU.
        self.input_ids[:num_token].copy_(
            self.input_ids_cpu[:num_token], non_blocking=True)

        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device, non_blocking=True)
        spec_decode_metadata = None

        input_ids = self.input_ids[:num_token]
        model_kwargs = {}
        model_kwargs["selected_indices"] = sample_indices

        with set_forward_context(attn_metadata, self.vllm_config):
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            forward_results = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                **model_kwargs,
            )

        return forward_results

    @torch.inference_mode()
    def forward_decode(self, num_token, max_num_tokens):
        # reuse the req created in forward_prefill
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        req_index = self.input_batch.req_id_to_index.get(self.req_id)
        self.input_batch.num_computed_tokens_cpu[req_index] = (self.input_batch.num_prompt_tokens[req_index])
        if self.vllm_config.kv_transfer_config is None:
            self.vllm_config.kv_transfer_config = KVTransferConfig()
        self.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        scheduler_output = Mock()
        scheduler_output.total_num_scheduled_tokens = num_token
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.num_scheduled_tokens[self.req_id] = num_token
        scheduler_output.scheduled_spec_decode_tokens = {}
        batch_reordered = self.attn_metadata_builders[0].reorder_batch(
            self.input_batch, scheduler_output)
        for i in range(1, len(self.kv_cache_group_specs)):
            assert not self.attn_metadata_builders[i].reorder_batch(
                self.input_batch, scheduler_output)

        self.input_batch.block_table.commit(num_reqs)
        num_scheduled_tokens = np.array([
            num_token
        ], dtype=np.int32)
        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
        arange = self.arange_np[:num_token] - cumsums_offsets
        positions_np = self.positions_np[:num_token]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        self.positions[:num_token].copy_(
            self.positions_cpu[:num_token], non_blocking=True)
        positions = self.positions[:num_token]

        self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[kv_cache_group_id]
            # NOTE(runze): since each request has at most M blocks, the offset is at most M-1
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                np.minimum(positions_np // block_size, block_table.max_num_blocks_per_req - 1))
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:num_token])

        self.attn_state = AscendAttentionState.DecodeOnly

        graph_pad_size = 0
        graph_pad_size = self.max_batch_size - num_token
        if graph_pad_size >= 0:
            if self.uses_mrope:
                padding_positions = torch.zeros(positions.size(0), graph_pad_size, dtype=positions.dtype, device=positions.device)
                positions = torch.cat([positions, padding_positions], dim=1)
            else:
                padding_positions = torch.zeros(graph_pad_size, dtype=positions.dtype, device=positions.device)
                positions = torch.cat([positions, padding_positions])
        extra_builder_kwargs = {'graph_pad_size': graph_pad_size}

        # build attention metadata
        attn_metadata = {}
        self.full_attn_metadata = None
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            # Prepare for cascade attention if enabled & beneficial.
            attn_metadata_i = self.attn_metadata_builders[kv_cache_group_id].build(
                num_reqs=num_reqs,
                num_actual_tokens=num_token,
                max_query_len=num_token,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
            if kv_cache_group_id == 0:
                self.full_attn_metadata = attn_metadata_i

            if not isinstance(self.attn_metadata_builders[kv_cache_group_id], DummyAttentionMetadataBuilder):
                raise ValueError(f"{self.attn_metadata_builders[kv_cache_group_id]} does not implement DummyAttentionMetadataBuilder")
            if self.enable_torchair_graph_mode and self.attn_state == AscendAttentionState.DecodeOnly:
                self.attn_metadata_builders[kv_cache_group_id].mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        # Prepare input_ids
        token_indices = (positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:num_token])

        # Copy the tensors to the NPU.
        self.input_ids[:num_token].copy_(
            self.input_ids_cpu[:num_token], non_blocking=True)

        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device, non_blocking=True)
        spec_decode_metadata = None

        input_ids = self.input_ids[:num_token]
        if graph_pad_size >= 0:
            if self.attn_state == AscendAttentionState.DecodeOnly:
                padding = torch.zeros(graph_pad_size, dtype=input_ids.dtype, device=input_ids.device)
            else:
                vocab_size = self.model_config.get_vocab_size()
                padding = torch.randint(1, vocab_size, (graph_pad_size,), dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding])
        inputs_embeds = None
        model_kwargs = {}
        model_kwargs["selected_indices"] = None

        with set_forward_context(attn_metadata, self.vllm_config):
            if not self.model_mark_static:
                if isinstance(self.model, GraphCompileConfiguration):
                    self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                else:
                    mark_static_for_graph_default(input_ids, inputs_embeds, positions, self.kv_caches)
                self.model_mark_static = True
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            forward_results = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                **model_kwargs,
            )

        return forward_results
