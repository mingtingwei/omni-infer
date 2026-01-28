import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

torch_npu = pytest.importorskip("torch_npu")

from omni.layers.attention import deepseek_mla as mla_mod


# -----------------------------
# Lightweight stubs (avoid heavy NPU ops in init)
# -----------------------------
class _RopeStub:
    def __init__(self, dim: int):
        self.dim = dim

    def get_cos_sin(self, positions: torch.Tensor):
        n = positions.numel()
        cos = torch.zeros((n, 1, 1, self.dim), device=positions.device, dtype=torch.bfloat16)
        sin = torch.zeros((n, 1, 1, self.dim), device=positions.device, dtype=torch.bfloat16)
        return cos, sin


class _IndexerStub(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.head_dim = 128  # keep attribute referenced in DeepseekMLA init

    def forward(self, x, q_norm, attn_metadata, kv_cache, is_prefill):
        # Return (tokens, 1, topk) int32
        topk = torch.zeros((q_norm.shape[0], 1, 2048), dtype=torch.int32, device=q_norm.device)
        return topk, None, None


def _make_model_extra_config(enable_dsa: bool, use_mlaprolog: bool, use_dcp: bool, is_prefill_node: bool):
    return SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            enable_dsa=enable_dsa,
            merge_qkv=False,
            use_mlaprolog=use_mlaprolog,
            moe_multi_stream_tune=False,
            use_omni_cache=False,
            mtp_remove_redundant_kv=False,
            prefill_enable_mla_alltoall=True,
            prefill_enable_mla_alltoall_local=False,
            enable_mla_prefill_multistream=False,
            prefill_moe_all_to_all=False,
            enable_indexer_quant=False,
            fa_quant=False,
            mla_multistream_limit_core=0,
            enable_prefill_micro_batch=False,
            c8_calib_path=None,
            use_dcp=use_dcp
        ),
        parall_config=SimpleNamespace(
            o_proj_tp_size=1,
            attn_sp_size=1,
        ),
        task_config=SimpleNamespace(
            decode_gear_list=[8],
            hardware_platform="A3",
            is_prefill_node=is_prefill_node
        ),
    )


@pytest.fixture
def mla():
    cfg = _make_model_extra_config(enable_dsa=True, use_mlaprolog=False, use_dcp=False, is_prefill_node=False)

    cur_vllm_cfg = SimpleNamespace(
        npu_compilation_config=SimpleNamespace(level=mla_mod.CompilationLevel.NO_COMPILATION),
        speculative_config=None,
    )

    pretrain_cfg = MagicMock()
    pretrain_cfg.rms_norm_eps = 1e-6
    pretrain_cfg.hidden_size = 7168
    pretrain_cfg.qk_rope_head_dim = 64
    pretrain_cfg.q_lora_rank = 1536

    cache_cfg = MagicMock()

    with patch.object(mla_mod, "model_extra_config", cfg), \
         patch.object(mla_mod, "get_current_vllm_config", return_value=cur_vllm_cfg), \
         patch.object(mla_mod, "supports_dynamo", return_value=False), \
         patch.object(mla_mod, "Indexer", _IndexerStub), \
         patch.object(mla_mod, "get_rope", return_value=_RopeStub(64)), \
         patch.object(mla_mod, "get_dp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
         patch.object(mla_mod, "get_world_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
         patch.object(mla_mod, "get_tensor_model_parallel_world_size", return_value=1), \
         patch.object(mla_mod, "get_tensor_model_parallel_rank", return_value=0):

        m = mla_mod.DeepseekMLA(
            config=pretrain_cfg,
            hidden_size=7168,
            num_heads=128,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=64,
            q_lora_rank=1536,
            kv_lora_rank=512,
            rope_theta=10000,
            rope_scaling=None,
            max_position_embeddings=8192,
            cache_config=cache_cfg,
            quant_config=None,
            prefix="model.layers.5.self_attn",
        )
        yield m


def test_init_basic(mla):
    assert mla.hidden_size == 7168
    assert mla.num_heads == 128
    assert mla.qk_nope_head_dim == 128
    assert mla.qk_rope_head_dim == 64
    assert mla.v_head_dim == 64
    assert mla.q_lora_rank == 1536
    assert mla.kv_lora_rank == 512
    assert mla.o_proj is not None
    assert mla.indexer is not None

@pytest.fixture
def mla_prefill():
    cfg = _make_model_extra_config(enable_dsa=False, use_mlaprolog=False, use_dcp=False, is_prefill_node=True)
    cfg.operator_opt_config.prefill_enable_mla_alltoall = True
    cfg.operator_opt_config.use_omni_cache = False
    cfg.parall_config.attn_sp_size = 1
    cfg.parall_config.o_proj_tp_size = 1
    
    cur_vllm_cfg = SimpleNamespace(
        npu_compilation_config=SimpleNamespace(level=mla_mod.CompilationLevel.NO_COMPILATION),
        speculative_config=None,
    )

    pretrain_cfg = MagicMock()
    pretrain_cfg.rms_norm_eps = 1e-6
    pretrain_cfg.hidden_size = 7168
    pretrain_cfg.qk_rope_head_dim = 64
    pretrain_cfg.q_lora_rank = 1536

    cache_cfg = MagicMock()
    with patch('vllm.distributed.parallel_state._TP') as mock_tp, \
         patch('vllm.distributed.parallel_state.get_tp_group') as mock_get_tp_group, \
         patch('vllm.distributed.parallel_state.get_tensor_model_parallel_world_size') as mock_tp_world_size, \
         patch('vllm.distributed.parallel_state.get_tensor_model_parallel_rank') as mock_tp_rank:

        mock_tp_group = SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            device_group=None
        )
        
        mock_tp.is_initialized.return_value = True
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_world_size.return_value = 1
        mock_tp_rank.return_value = 0
        with patch.object(mla_mod, "model_extra_config", cfg), \
            patch.object(mla_mod, "get_current_vllm_config", return_value=cur_vllm_cfg), \
            patch.object(mla_mod, "supports_dynamo", return_value=False), \
            patch.object(mla_mod, "get_rope", return_value=_RopeStub(64)), \
            patch.object(mla_mod, "get_dp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
            patch.object(mla_mod, "get_world_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
            patch.object(mla_mod, "get_tensor_model_parallel_world_size", return_value=1), \
            patch.object(mla_mod, "get_tensor_model_parallel_rank", return_value=0), \
            patch.object(mla_mod, "get_o_proj_dp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)):

            m = mla_mod.DeepseekMLA(
                config=pretrain_cfg,
                hidden_size=7168,
                num_heads=128,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=64,
                q_lora_rank=1536,
                kv_lora_rank=512,
                rope_theta=10000,
                rope_scaling=None,
                max_position_embeddings=8192,
                cache_config=cache_cfg,
                quant_config=None,
                prefix="model.layers.5.self_attn",
            )
            
            m.stream1 = None
            m.kv_a_proj_event = torch.npu.Event(blocking=False, enable_timing=False)
            m.q_norm_event = torch.npu.Event(blocking=False, enable_timing=False)
            m.kv_all_gather_event = torch.npu.Event(blocking=False, enable_timing=False)
            
            yield m

def test_forward_prefill_basic(mla_prefill):
    mla = mla_prefill
    bsz, seq_len = 4, 2048
    dev = mla_mod.current_platform.device_type
    
    hidden_states = torch.randn(bsz * seq_len, 7168, dtype=torch.bfloat16, device=dev)
    positions = torch.randint(0, seq_len, (bsz * seq_len,), dtype=torch.int64, device=dev)
    
    comm_group = SimpleNamespace(world_size=1, rank_in_group=0)
    
    with patch.object(mla_mod, "tensor_model_parallel_all_gather") as mock_tp_all_gather, \
         patch.object(mla_mod, "mla_tensor_model_parallel_all_gather") as mock_mla_all_gather, \
         patch.object(torch_npu, "npu_interleave_rope") as mock_rope_op, \
         patch.object(torch_npu, "npu_rms_norm") as mock_rmsnorm_op, \
         patch.object(torch_npu, "npu_kv_rmsnorm_rope_cache") as mock_kv_cache_op, \
         patch.object(torch.ops.npu, "npu_fused_infer_attention_score") as mock_attn_op:
        
        mock_tp_all_gather.side_effect = lambda x, *args, **kwargs: x
        mock_mla_all_gather.side_effect = lambda x, *args, **kwargs: x
        mock_rope_op.side_effect = lambda x, cos, sin: x
        mock_rmsnorm_op.side_effect = lambda x, *args, **kwargs: x
        
        mock_kv_cache_op.return_value = (None, None, None, None)
        
        attn_output_shape = (bsz * seq_len, mla.num_local_heads, mla.v_head_dim)
        mock_attn_output = torch.randn(attn_output_shape, dtype=torch.bfloat16, device=dev)
        mock_attn_op.return_value = (mock_attn_output, None)
        
        mla.q_a_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.q_lora_rank, dtype=torch.bfloat16, device=dev), None))
        mla.kv_a_proj_with_mqa.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.kv_lora_rank + mla.qk_rope_head_dim, 
                       dtype=torch.bfloat16, device=dev), None))
        mla.q_b_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.num_heads * mla.qk_head_dim, 
                       dtype=torch.bfloat16, device=dev), None))
        mla.kv_b_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim), 
                       dtype=torch.bfloat16, device=dev), None))
        mla.o_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.hidden_size, dtype=torch.bfloat16, device=dev), None))
        
        output = mla._forward_prefill(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=None,
            attn_metadata=None,
            comm_group=comm_group
        )
        
        assert output.shape == (bsz * seq_len, mla.hidden_size)
        mock_mla_all_gather.assert_called()

@pytest.fixture
def mla_prefill_dcp():
    cfg = _make_model_extra_config(enable_dsa=False, use_mlaprolog=False, use_dcp=True, is_prefill_node=True)
    
    cur_vllm_cfg = SimpleNamespace(
        npu_compilation_config=SimpleNamespace(level=mla_mod.CompilationLevel.NO_COMPILATION),
        speculative_config=None,
    )

    pretrain_cfg = MagicMock()
    pretrain_cfg.rms_norm_eps = 1e-6
    pretrain_cfg.hidden_size = 7168
    pretrain_cfg.qk_rope_head_dim = 64
    pretrain_cfg.q_lora_rank = 1536

    cache_cfg = MagicMock()

    with patch('vllm.distributed.parallel_state._TP') as mock_tp, \
         patch('vllm.distributed.parallel_state.get_tp_group') as mock_get_tp_group, \
         patch('vllm.distributed.parallel_state.get_tensor_model_parallel_world_size') as mock_tp_world_size, \
         patch('vllm.distributed.parallel_state.get_tensor_model_parallel_rank') as mock_tp_rank:

        mock_tp_group = SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            device_group=None
        )
        
        mock_tp.is_initialized.return_value = True
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_world_size.return_value = 1
        mock_tp_rank.return_value = 0

        with patch.object(mla_mod, "model_extra_config", cfg), \
             patch.object(mla_mod, "get_current_vllm_config", return_value=cur_vllm_cfg), \
             patch.object(mla_mod, "supports_dynamo", return_value=False), \
             patch.object(mla_mod, "get_rope", return_value=_RopeStub(64)), \
             patch.object(mla_mod, "get_dp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
             patch.object(mla_mod, "get_tp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
             patch.object(mla_mod, "get_world_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
             patch.object(mla_mod, "get_mla_cp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)), \
             patch.object(mla_mod, "get_tensor_model_parallel_world_size", return_value=1), \
             patch.object(mla_mod, "get_tensor_model_parallel_rank", return_value=0):

            m = mla_mod.DeepseekMLA(
                config=pretrain_cfg,
                hidden_size=7168,
                num_heads=128,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=64,
                q_lora_rank=1536,
                kv_lora_rank=512,
                rope_theta=10000,
                rope_scaling=None,
                max_position_embeddings=8192,
                cache_config=cache_cfg,
                quant_config=None,
                prefix="model.layers.5.self_attn",
            )
            yield m

def test_forward_prefill_dcp_basic(mla_prefill_dcp):
    mla = mla_prefill_dcp
    bsz, seq_len = 4, 2048
    dev = mla_mod.current_platform.device_type
    
    hidden_states = torch.randn(bsz * seq_len, 7168, dtype=torch.bfloat16, device=dev)
    positions = torch.randint(0, seq_len, (bsz * seq_len,), dtype=torch.int64, device=dev)

    comm_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0
    )
    
    with patch.object(mla_mod, "mla_tensor_model_parallel_all_gather") as mock_all_gather, \
         patch.object(mla_mod, "tensor_model_parallel_all_gather") as mock_tp_all_gather, \
         patch.object(torch_npu, "npu_interleave_rope") as mock_rope_op, \
         patch.object(torch_npu, "npu_rms_norm") as mock_rmsnorm_op, \
         patch.object(torch_npu, "npu_kv_rmsnorm_rope_cache") as mock_kv_cache_op, \
         patch.object(torch.ops.npu, "npu_fused_infer_attention_score") as mock_attn_op:
        
        mock_all_gather.side_effect = lambda x, *args, **kwargs: x  
        mock_tp_all_gather.side_effect = lambda x, *args, **kwargs: x
        mock_rope_op.side_effect = lambda x, cos, sin: x 
        mock_rmsnorm_op.side_effect = lambda x, *args, **kwargs: x
        mock_kv_cache_op.return_value = (None, None, None, None) 
        
        attn_output_shape = (bsz * seq_len // 2, mla.num_local_heads * 2, mla.v_head_dim)
        mock_attn_output = torch.randn(attn_output_shape, dtype=torch.bfloat16, device=dev)
        mock_attn_op.return_value = (mock_attn_output, None)
        
        mla.q_a_proj.forward = MagicMock(return_value=(torch.randn(bsz * seq_len, mla.q_lora_rank, 
                                                                 dtype=torch.bfloat16, device=dev), None))
        mla.kv_a_proj_with_mqa.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.kv_lora_rank + mla.qk_rope_head_dim, 
                       dtype=torch.bfloat16, device=dev), None))
        mla.q_b_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.num_heads * mla.qk_head_dim, 
                       dtype=torch.bfloat16, device=dev), None))
        mla.kv_b_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim), 
                       dtype=torch.bfloat16, device=dev), None))
        mla.o_proj.forward = MagicMock(return_value=(
            torch.randn(bsz * seq_len, mla.hidden_size, dtype=torch.bfloat16, device=dev), None))
        
        output = mla._forward_prefill_dcp(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=None,
            attn_metadata=None,
            comm_group=comm_group
        )
        
        assert output.shape == (bsz * seq_len, mla.hidden_size)
        mock_all_gather.assert_called() 

def test_forward_decode_patches_real_torch_npu_ops(mla):
    bsz = 8
    dev = mla_mod.current_platform.device_type

    hidden_states = torch.randn(bsz, 7168, dtype=torch.bfloat16, device=dev)
    positions = torch.arange(bsz, dtype=torch.int64, device=dev)

    block_num, block_size = 4, 128
    nope_cache = torch.randn(block_num, block_size, 1, 512, dtype=torch.bfloat16, device=dev)
    rope_cache = torch.randn(block_num, block_size, 1, 64, dtype=torch.bfloat16, device=dev)
    kv_cache = (nope_cache, rope_cache)

    attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(
            cos=torch.randn(bsz, 1, 1, 64, dtype=torch.bfloat16, device=dev),
            sin=torch.randn(bsz, 1, 1, 64, dtype=torch.bfloat16, device=dev),
            block_table=torch.zeros((bsz, 1), dtype=torch.int32, device=dev),
            seq_lens=torch.ones((bsz,), dtype=torch.int64, device=dev),
        ),
        slot_mapping=torch.arange(bsz, dtype=torch.int32, device=dev),
    )

    # IMPORTANT: provide W_UK/W_UV on NPU; __init__ doesn't build them when dp_world_size==1
    # W_UK: (num_local_heads=128, qk_nope_head_dim=128, kv_lora_rank=512)
    mla.W_UK = torch.randn(mla.num_local_heads, mla.qk_nope_head_dim, mla.kv_lora_rank,
                           dtype=torch.bfloat16, device=dev)
    # W_UV: shape used by matmul(attn_output, W_UV) where attn_output becomes (num_heads, B, kv_lora_rank)
    # so W_UV should be (num_local_heads=128, kv_lora_rank=512, v_head_dim=64)
    mla.W_UV = torch.randn(mla.num_local_heads, mla.kv_lora_rank, mla.v_head_dim,
                           dtype=torch.bfloat16, device=dev)

    with patch.object(mla_mod, "tensor_model_parallel_all_gather", side_effect=lambda x, dim=0: x), \
         patch.object(torch_npu, "npu_kv_rmsnorm_rope_cache", create=True) as mock_kv_cache_op, \
         patch.object(torch_npu, "npu_interleave_rope", side_effect=lambda x, cos, sin: x, create=True) as mock_rope_op, \
         patch.object(torch_npu, "npu_sparse_flash_attention", create=True) as mock_sparse_attn:

        q_lowrank = torch.randn(bsz, 1536, dtype=torch.bfloat16, device=dev)
        kv_latent = torch.randn(bsz, 576, dtype=torch.bfloat16, device=dev)
        mla.q_a_proj.forward = MagicMock(return_value=(q_lowrank, None))
        mla.kv_a_proj_with_mqa.forward = MagicMock(return_value=(kv_latent, None))

        # RMSNorm in this repo sometimes returns (out, res)
        mla.q_a_layernorm.forward = MagicMock(side_effect=lambda x, *args, **kwargs: (x, None))

        # q_b_proj output (bsz, num_heads*qk_head_dim)
        q_full = torch.randn(bsz, 128 * (128 + 64), dtype=torch.bfloat16, device=dev)
        mla.q_b_proj.forward = MagicMock(return_value=(q_full, None))

        # kv cache op output (k_rope, k_nope, _, _)
        k_rope = torch.randn(block_num, block_size, 1, 64, dtype=torch.bfloat16, device=dev)
        k_nope = torch.randn(block_num, block_size, 1, 512, dtype=torch.bfloat16, device=dev)
        mock_kv_cache_op.return_value = (k_rope, k_nope, None, None)

        # In enable_dsa branch, code expects attn_output to be (T, 1, num_heads, kv_lora_rank) so it can squeeze(1).transpose(0,1)
        attn_out = torch.randn(bsz, 1, mla.num_local_heads, mla.kv_lora_rank,
                               dtype=torch.bfloat16, device=dev)
        mock_sparse_attn.return_value = (attn_out,)

        # o_proj returns (output, _)
        mla.o_proj.forward = MagicMock(return_value=(torch.randn(bsz, 7168, dtype=torch.bfloat16, device=dev), None))

        out = mla._forward_decode(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        assert out.shape == (bsz, 7168)
        mock_kv_cache_op.assert_called_once()
        mock_rope_op.assert_called_once()
        mock_sparse_attn.assert_called_once()


def test_forward_mlaprolog_decode_patch_v3_and_v2(mla):
    bsz = 8
    dev = mla_mod.current_platform.device_type
    hidden_states = torch.randn(bsz, 7168, dtype=torch.bfloat16, device=dev)

    block_num, block_size = 2, 128
    nope_cache = torch.randn(block_num, block_size, 1, 512, dtype=torch.bfloat16, device=dev)
    rope_cache = torch.randn(block_num, block_size, 1, 64, dtype=torch.bfloat16, device=dev)

    attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(
            cos=torch.randn(bsz, 1, 1, 64, dtype=torch.bfloat16, device=dev),
            sin=torch.randn(bsz, 1, 1, 64, dtype=torch.bfloat16, device=dev),
        ),
        slot_mapping=torch.arange(bsz, dtype=torch.int32, device=dev).view(bsz, 1),
    )

    # v3 branch
    mla_mod.model_extra_config.operator_opt_config.enable_dsa = True
    with patch.object(torch_npu, "npu_mla_prolog_v3", create=True) as mock_v3:
        q_nope = torch.randn(bsz * 128, 512, dtype=torch.bfloat16, device=dev)
        q_pe = torch.randn(bsz * 128, 64, dtype=torch.bfloat16, device=dev)
        dq_q_nope = torch.randn(1, dtype=torch.bfloat16, device=dev)
        q_norm = torch.randn(bsz * 128, 1536, dtype=torch.bfloat16, device=dev)
        dq_q_norm = torch.randn(1, dtype=torch.bfloat16, device=dev)
        mock_v3.return_value = (q_nope, q_pe, dq_q_nope, q_norm, dq_q_norm)

        out = mla._forward_mlaprolog_decode(
            hidden_states=hidden_states,
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            attn_metadata=attn_metadata,
            nz_block_size=16,
        )

        mock_v3.assert_called_once()
        assert mock_v3.call_args.kwargs["cache_mode"] == "PA_BSND"
        assert mock_v3.call_args.kwargs["query_norm_flag"] is True
        assert out[0].shape == (bsz, 128, 512)
        assert out[1].shape == (bsz, 128, 64)

    # v2 branch
    mla_mod.model_extra_config.operator_opt_config.enable_dsa = False
    with patch.object(torch.ops.npu, "npu_mla_prolog_v2") as mock_v2:
        q_nope = torch.randn(bsz * 128, 512, dtype=torch.bfloat16, device=dev)
        q_pe = torch.randn(bsz * 128, 64, dtype=torch.bfloat16, device=dev)
        k_nope = torch.randn(block_num * block_size, 512, dtype=torch.bfloat16, device=dev)
        k_rope = torch.randn(block_num * block_size, 64, dtype=torch.bfloat16, device=dev)
        dq_q_nope = torch.randn(1, dtype=torch.bfloat16, device=dev)
        mock_v2.return_value = (q_nope, q_pe, k_nope, k_rope, dq_q_nope)

        out = mla._forward_mlaprolog_decode(
            hidden_states=hidden_states,
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            attn_metadata=attn_metadata,
            nz_block_size=16,
        )

        mock_v2.assert_called_once()
        assert mock_v2.call_args.kwargs["cache_mode"] == "PA_NZ"
        assert out[3].shape == (block_num, 1, 512 // 16, block_size, 16)
        assert out[4].shape == (block_num, 1, 64 // 16, block_size, 16)
