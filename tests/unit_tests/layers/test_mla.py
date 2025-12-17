import random
import torch
from unittest import TestCase
from unittest.mock import MagicMock, patch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.attention import Attention
from vllm.attention.backends.abstract import (
    AttentionMetadata,
)
from omni.models.config_loader.loader import model_extra_config
from omni.layers.attention.deepseek_mla import DeepseekMLA


class Test_DeepseekV32_MLA(TestCase):
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
    return_value=8)
    @patch('vllm.distributed.parallel_state._DP',
    new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state._TP',
    new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('transformers.PretrainedConfig')
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))
    @patch('vllm.config.CacheConfig',
    new_callable=lambda: MagicMock(spec=CacheConfig))
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    def setUp(self, 
              mock_model_extra_config,
              mock_cache_config,
              mock_quant_config,
              mock_pretrain_config,
              mock_vllm_tp,
              mock_vllm_dp,
              mock_vllm_tp_world_size
              ):

        # mock parameters
        self.hidden_size = 7168
        self.num_heads = 128
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.qk_head_dim ** -0.5
        
        self.v_head_dim = 64
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.rope_theta = 10000
        self.rope_scaling = {'beta_fast': 32, 
                        'beta_slow': 1, 
                        'factor': 40, 
                        'mscale': 1.0, 
                        'mscale_all_dim': 1.0, 
                        'original_max_position_embeddings': 4096,  
                        'type': 'yarn', 
                        'rope_type': 
                        'deepseek_yarn'}
        self.max_position_embeddings = 8192
        self.prefix = "model.layers.5.self_attn"

        # mock model extra configs for initialization
        mock_model_extra_config.operator_opt_config.enable_dsa = True
        mock_model_extra_config.operator_opt_config.merge_qkv = False
        mock_model_extra_config.operator_opt_config.use_mlaprolog = True
        mock_model_extra_config.operator_opt_config.moe_multi_stream_tune = False
        mock_model_extra_config.operator_opt_config.use_omni_cache = False
        mock_model_extra_config.parall_config.o_proj_tp_size = 8

        # mock vllm configs for initialization
        mock_pretrain_config.rms_norm_eps.return_value = 1e-6
        mock_pretrain_config.hidden_size = self.hidden_size
        mock_pretrain_config.qk_rope_head_dim = self.qk_rope_head_dim
        mock_pretrain_config.q_lora_rank = self.q_lora_rank

        mock_quant_config = None

        # mock communication configs for initialization
        mock_vllm_tp.world_size = 8
        mock_vllm_tp.rank_in_group = MagicMock()
        mock_vllm_tp.device_group = MagicMock()  
        mock_vllm_dp.world_size = 1
        mock_vllm_dp.rank_in_group = MagicMock()
        mock_vllm_dp.device_group = MagicMock() 

        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._TP = mock_vllm_tp
        vllm_ps._DP = mock_vllm_dp

        self.mla = DeepseekMLA(
                config=mock_pretrain_config,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                rope_theta=self.rope_theta,
                rope_scaling=self.rope_scaling,
                max_position_embeddings=self.max_position_embeddings,
                cache_config=mock_cache_config,
                quant_config=mock_quant_config,
                prefix=f"{self.prefix}.self_attn",
            )

    def test_init(self):
        self.assertEqual(self.mla.hidden_size, self.hidden_size)
        self.assertEqual(self.mla.qk_nope_head_dim, self.qk_nope_head_dim)
        self.assertEqual(self.mla.qk_rope_head_dim, self.qk_rope_head_dim)
        self.assertEqual(self.mla.qk_head_dim, self.qk_head_dim)
        self.assertEqual(self.mla.v_head_dim, self.v_head_dim)
        self.assertEqual(self.mla.q_lora_rank, self.q_lora_rank)
        self.assertEqual(self.mla.kv_lora_rank, self.kv_lora_rank)
        self.assertEqual(self.mla.num_heads, self.num_heads)
        self.assertEqual(self.mla.rope_theta, self.rope_theta)

        self.assertIsNotNone(self.mla.o_proj)
        self.assertIsNotNone(self.mla.q_a_proj)
        self.assertIsNotNone(self.mla.q_b_proj)
        self.assertIsNotNone(self.mla.kv_b_proj)
        self.assertIsNotNone(self.mla.q_a_layernorm)
        self.assertIsNotNone(self.mla.kv_a_layernorm)
        self.assertIsNotNone(self.mla.kv_a_proj_with_mqa)
        self.assertIsNotNone(self.mla.q_norm_event)
        self.assertIsNotNone(self.mla.kv_a_proj_event)
        self.assertIsNotNone(self.mla.kv_all_gather_event)
        self.assertIsNotNone(self.mla.rotary_emb)
        self.assertIsNotNone(self.mla.attn_mask)
        self.assertIsNotNone(self.mla.vllm_attn)

        self.assertEqual(self.mla.is_init, True)
        self.assertEqual(self.mla.layer_idx, 5)

    def tearDown(self):   
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._TP = None
        vllm_ps._DP = None  

    @patch("omni.layers.attention.deepseek_mla.mla_tensor_model_parallel_all_gather")
    @patch("torch_npu.npu_interleave_rope")
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_prefill_absorb(self, 
                                    mock_attn_meta_data,
                                    mock_npu_interleave_rope,
                                    mock_tp_all_gather
                                    ):
        # mock layers
        mock_bsz = 256
        mock_kv_cache = None
        mock_positions = torch.randint(low=0, high=1024, size=(2048,), dtype=torch.int64)
        mock_hidden_states = torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16)
        self.mla.W_UK = torch.randn(16, 128, self.kv_lora_rank, dtype=torch.bfloat16)

        mock_attn_meta_data.prefill.cos.return_value = torch.randn(2048, 1, 1, 64, dtype=torch.bfloat16)
        mock_attn_meta_data.prefill.sin.return_value = torch.randn(2048, 1, 1, 64, dtype=torch.bfloat16)

        q_lora = torch.randn(mock_bsz, self.q_lora_rank, dtype=torch.float)   
        self.mla.q_a_proj.forward = MagicMock(return_value=[q_lora])

        latent_cache = torch.randn(mock_bsz, 576, dtype=torch.bfloat16)
        self.mla.kv_a_proj_with_mqa.forward =  MagicMock(return_value=[latent_cache, None])
        
        mock_tp_all_gather.side_effect = lambda data, dim, comm_group: data
        
        self.mla.q_a_layernorm.forward = MagicMock(side_effect=lambda data: data)

        self.mla.q_b_proj.forward = MagicMock(return_value=torch.randn(mock_bsz, 128, 192, dtype=torch.bfloat16))

        self.mla.kv_a_layernorm.forward = MagicMock(side_effect=lambda data: data)

        mock_npu_interleave_rope.side_effect = lambda data, cos, sin: data

        self.mla._apply_attention = MagicMock()
        self.mla._apply_attention.side_effect = lambda idc, q_n, q_r, k_r, attn, is_second_attn: q_n

        self.mla.mla_epilog = MagicMock()
        self.mla.mla_epilog.return_value = torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16)

        self.mla.indexer = MagicMock()
        self.mla.indexer.return_value = [torch.randint(low=0, high=8, size=(8, 1, 2048), dtype=torch.int32),
                                         torch.randint(low=0, high=8, size=(8, 1, 2048), dtype=torch.int32),
                                         torch.randn(8, 1, 128, dtype=torch.bfloat16)]
                                         
        result = self.mla._forward_prefill_absorb(positions=mock_positions,
                                                  hidden_states=mock_hidden_states,
                                                  kv_cache=mock_kv_cache,
                                                  attn_metadata=mock_attn_meta_data,
                                                  comm_group=None
                                                )

        self.assertEqual(result.shape[0], mock_bsz)
        self.assertEqual(result.shape[1], self.hidden_size)
        self.assertEqual(mock_npu_interleave_rope.call_count, 2)

    @patch("omni.layers.attention.deepseek_mla.tensor_model_parallel_all_gather")
    @patch("torch.ops.custom.npu_sparse_flash_attention")
    @patch("torch_npu.npu_interleave_rope")
    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    @patch('omni.layers.attention.deepseek_mla.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")    
    def test_forward_decode(self,
                            mock_attn_meta_data,
                            mock_model_extra_config,
                            mock_npu_kv_rmsnorm_rope_cache,
                            mock_npu_interleave_rope,
                            mock_npu_sparse_flash_attention,
                            mock_tp_all_gather
                            ):
        
        mock_bsz = 8

        mock_min_blocks = 1024
        mock_max_blocks = 8192

        mock_model_extra_config.operator_opt_config.enable_dsa = True
        mock_model_extra_config.operator_opt_config.use_mlaprolog = False
        mock_model_extra_config.operator_opt_config.moe_multi_stream_tune = False
        mock_model_extra_config.operator_opt_config.use_omni_cache = False
        mock_model_extra_config.parall_config.o_proj_tp_size = 8

        mock_attn_meta_data.decode.cos.return_value = torch.randn(mock_bsz, 1, 1, 64, dtype=torch.bfloat16)
        mock_attn_meta_data.decode.sin.return_value = torch.randn(mock_bsz, 1, 1, 64, dtype=torch.bfloat16)
        mock_attn_meta_data.tmp_slot_mapping.return_value = torch.randint(low=0, high=mock_bsz, size=(mock_bsz,), dtype=torch.int64)
        mock_attn_meta_data.decode.block_table.return_value = torch.randint(low=0, high=mock_bsz, size=(mock_bsz, mock_bsz), dtype=torch.int32)
        mock_attn_meta_data.decode.seq_lens.return_value = torch.randint(low=0, high=mock_bsz, size=(mock_bsz,), dtype=torch.int64)

        mock_positions = torch.randint(low=0, high=mock_bsz, size=(mock_bsz,), dtype=torch.int64)
        mock_hidden_states = torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16)

        mock_kvcache_blocks = random.randint(mock_min_blocks, mock_max_blocks)  
        mock_kvcache = [torch.randn(mock_kvcache_blocks, 128, 1, self.kv_lora_rank, dtype=torch.bfloat16),
                        torch.randn(mock_kvcache_blocks, 128, 1, 64, dtype=torch.bfloat16)]

        self.mla.norm_res = list(range(128))
        self.mla.actual_seq_lengths = torch.randint(low=0, high=128, size=(16, 128), dtype=torch.int64)
        self.mla.num_local_heads = 128
        self.mla.W_UK = torch.randn(128, 128, self.kv_lora_rank, dtype=torch.bfloat16)
        self.mla.W_UV = torch.randn(128, self.kv_lora_rank, 128, dtype=torch.bfloat16)
    
        mock_tp_all_gather.side_effect = lambda data, dim: data

        q_lowrank = torch.randn(mock_bsz, self.q_lora_rank, dtype=torch.bfloat16)   
        self.mla.q_a_proj.forward = MagicMock(return_value=[q_lowrank])

        latent_cache = torch.randn(mock_bsz, 576, dtype=torch.bfloat16)
        self.mla.kv_a_proj_with_mqa.forward =  MagicMock(return_value=[latent_cache, None])

        self.mla.q_a_layernorm.forward = MagicMock(return_value=[q_lowrank, None])

        self.mla.q_b_proj.forward = MagicMock(side_effect=lambda data: [data.repeat(1, 16)])

        mock_npu_kv_rmsnorm_rope_cache.return_value = [torch.randn(mock_kvcache_blocks, 128, 1, 64, dtype=torch.bfloat16),
                                                       torch.randn(mock_kvcache_blocks, 128, 1, self.kv_lora_rank, dtype=torch.bfloat16),
                                                       None,
                                                       None]

        mock_npu_interleave_rope.side_effect = lambda data, cos, sin: data

        self.mla.indexer = MagicMock()
        self.mla.indexer.return_value = [torch.randint(low=0, high=2048, size=(8, 1, 2048), dtype=torch.int32), None, None]

        mock_npu_sparse_flash_attention.return_value = torch.randn(8, 128, self.kv_lora_rank, dtype=torch.bfloat16)

        self.mla.o_proj.forward = MagicMock(return_value=[torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16), None])

        result = self.mla._forward_decode(positions=mock_positions,
                                            hidden_states=mock_hidden_states,
                                            kv_cache=mock_kvcache,
                                            attn_metadata=mock_attn_meta_data,
                                        )

        mock_npu_kv_rmsnorm_rope_cache.assert_called_once()
        mock_npu_interleave_rope.assert_called_once()               
        mock_npu_sparse_flash_attention.assert_called_once()

        self.assertEqual(result.shape[0], mock_bsz)
        self.assertEqual(result.shape[1], self.hidden_size)  

    @patch("omni.layers.attention.deepseek_mla.get_o_proj_dp_group")
    @patch("omni.layers.attention.deepseek_mla.tensor_model_parallel_all_gather")
    @patch("omni.layers.attention.deepseek_mla.mla_tensor_model_parallel_all_gather")
    @patch('vllm.distributed.parallel_state._TP',
    new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("torch_npu.npu_interleave_rope")
    def test_forward_prefill(self,
                              mock_interleave,
                              mock_vllm_tp,
                              mock_mla_all_gather,
                              mock_tensor_all_gather,
                              mock_o_proj_group):
        mock_bsz = 8
        positions = torch.randint(low=0, high=16, size=(mock_bsz,), dtype=torch.int64)
        hidden_states = torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16)

        mock_vllm_tp.world_size = 8

        self.mla.merge_qkv = False
        self.mla.quant_symbol = False
        self.mla.model_parallel = True
        self.mla.o_proj.forward = MagicMock(return_value=[torch.randn(mock_bsz, self.hidden_size, dtype=torch.bfloat16)])

        self.mla.q_a_proj.forward = MagicMock(return_value=[torch.randn(mock_bsz, self.q_lora_rank, dtype=torch.bfloat16)])
        latent_cache = torch.randn(mock_bsz, self.kv_lora_rank + self.qk_rope_head_dim, dtype=torch.bfloat16)
        self.mla.kv_a_proj_with_mqa.forward = MagicMock(return_value=[latent_cache])
        self.mla.q_a_layernorm.forward = MagicMock(side_effect=lambda data: data)
        self.mla.q_b_proj.forward = MagicMock(
            return_value=torch.randn(mock_bsz, self.num_heads, self.qk_head_dim, dtype=torch.bfloat16)
        )
        self.mla.kv_a_layernorm.forward = MagicMock(side_effect=lambda data: data)

        mock_mla_all_gather.side_effect = lambda data, dim, comm_group=None: data
        mock_tensor_all_gather.side_effect = lambda data, dim: data
        mock_interleave.side_effect = lambda data, cos, sin: data
        mock_o_proj_group.return_value = MagicMock(world_size=1)

        model_extra_config.operator_opt_config.prefill_enable_mla_alltoall = False
        model_extra_config.parall_config.o_proj_tp_size = 1

        output = self.mla._forward_prefill(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=None,
            attn_metadata=None,
            comm_group=None,
        )

        self.assertEqual(output.shape[0], mock_bsz)
        self.assertEqual(output.shape[1], self.hidden_size)
        self.assertTrue(mock_interleave.called)

    @patch("torch.ops.npu.npu_mla_prolog_v2")
    @patch("torch.ops.custom.npu_mla_prolog_v3")
    @patch('omni.layers.attention.deepseek_mla.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    def test_forward_mlaprolog_decode(self, 
                                      mock_model_extra_config,
                                      mock_mla_prolog_v3, 
                                      mock_mla_prolog_v2):
        bsz = 8
        block_num, block_size = 1, 128
        nz_block_size = 16

        nope_cache = torch.randn(block_num, block_size, 1, self.kv_lora_rank, dtype=torch.bfloat16)
        rope_cache = torch.randn(block_num, block_size, 1, self.qk_rope_head_dim, dtype=torch.bfloat16)

        attn_metadata = MagicMock()
        attn_metadata.decode.cos = torch.randn(bsz, 1, 1, self.qk_rope_head_dim, dtype=torch.bfloat16)
        attn_metadata.decode.sin = torch.randn(bsz, 1, 1, self.qk_rope_head_dim, dtype=torch.bfloat16)
        attn_metadata.slot_mapping = torch.arange(bsz, dtype=torch.int32).view(bsz, 1)

        q_nope = torch.randn(bsz * self.mla.num_local_heads, self.kv_lora_rank, dtype=torch.bfloat16)
        q_pe = torch.randn(bsz * self.mla.num_local_heads, self.qk_rope_head_dim, dtype=torch.bfloat16)
        dequant_scale_q_nope = torch.randn(1, dtype=torch.bfloat16)
        q_norm = torch.randn(bsz * self.mla.num_local_heads, self.q_lora_rank, dtype=torch.bfloat16)
        dequant_scale_q_norm = torch.randn(1, dtype=torch.bfloat16)

        mock_mla_prolog_v3.return_value = (
            q_nope,
            q_pe,
            dequant_scale_q_nope,
            q_norm,
            dequant_scale_q_norm,
        )
        mock_model_extra_config.operator_opt_config.enable_dsa = True
        mock_model_extra_config.operator_opt_config.use_omni_cache = True


        hidden_states = torch.randn(bsz, self.hidden_size, dtype=torch.bfloat16)
        output = self.mla._forward_mlaprolog_decode(
            hidden_states=hidden_states,
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            attn_metadata=attn_metadata,
            nz_block_size=nz_block_size,
        )

        mock_mla_prolog_v3.assert_called_once()
        self.assertEqual(mock_mla_prolog_v3.call_args.kwargs["cache_mode"], "PA_BSND")
        self.assertEqual(output[0].shape, (bsz, self.mla.num_local_heads, self.kv_lora_rank))
        self.assertEqual(output[1].shape, (bsz, self.mla.num_local_heads, self.qk_rope_head_dim))
        self.assertIs(output[3], nope_cache)
        self.assertIs(output[4], rope_cache)

        self.mla.model_parallel = False
        self.mla.quant_symbol = False
        mock_model_extra_config.operator_opt_config.enable_dsa = False
        mock_model_extra_config.operator_opt_config.use_omni_cache = False

        k_nope = torch.randn(
            block_num,
            self.kv_lora_rank // nz_block_size,
            block_size,
            nz_block_size,
            dtype=torch.bfloat16,
        )
        k_rope = torch.randn(
            block_num,
            self.qk_rope_head_dim // 16,
            block_size,
            16,
            dtype=torch.bfloat16,
        )

        mock_mla_prolog_v2.return_value = (
            q_nope,
            q_pe,
            k_nope,
            k_rope,
            dequant_scale_q_nope,
        )

        output = self.mla._forward_mlaprolog_decode(
            hidden_states=hidden_states,
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            attn_metadata=attn_metadata,
            nz_block_size=nz_block_size,
        )

        mock_mla_prolog_v2.assert_called_once()
        self.assertEqual(mock_mla_prolog_v2.call_args.kwargs["cache_mode"], "PA_NZ")
        self.assertEqual(
            output[3].shape,
            (block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size),
        )
        self.assertEqual(
            output[4].shape,
            (
                block_num,
                1,
                self.qk_rope_head_dim // 16,
                block_size,
                16,
            ),
        )

if __name__ == "__main__":
    unittest.main()