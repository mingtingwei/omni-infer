import os
import torch
from unittest import TestCase
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from vllm.config import QuantizationConfig
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import GroupCoordinator as GroupCoordinatorGPU
from omni.models.config_loader.loader import model_extra_config
from omni.layers.moe.deepseek_moe import ReplicatedDeepseekMLP, ParallelDeepseekMLP, DeepseekMoE
from omni.adaptors.vllm.distributed.parallel_state import GroupCoordinator
from omni.adaptors.vllm.utils import get_attr_by_names
from omni.layers.moe.fused_moe.layer import FusedMoE

class test_ReplicatedDeepseekMLP(TestCase):
    @patch('vllm.platforms.current_platform')
    @patch('vllm.distributed.parallel_state._WORLD',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._PP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._EP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))    
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    def setUp(self,
              mock_model_extra_config,
              mock_quant_config,
              mock_vllm_ep,
              mock_vllm_pp,
              mock_vllm_world,
              mock_current_platform):
        self.mock_hidden_size = 7168
        self.mock_intermediate_size = 2048
        self.mock_hidden_act = "silu"
        self.mock_reduce_results = False
        self.mock_prefix = "model.layers.3.mlp.shared_experts"
        self.mock_ep_size = 8
        self.mock_world_size = 8
        self.mock_rank_in_group = 0
        self.mock_bsz = 256

        mock_model_extra_config.operator_opt_config.decode_moe_dispatch_combine = True
        mock_quant_config = None
        mock_current_platform.device_type = "npu"

        mock_vllm_ep.world_size = self.mock_ep_size
        mock_vllm_ep.rank_in_group = MagicMock()
        mock_vllm_ep.device_group = MagicMock()  
        mock_vllm_pp.world_size = self.mock_world_size
        mock_vllm_pp.rank_in_group = self.mock_rank_in_group
        mock_vllm_pp.device_group = MagicMock()  
        mock_vllm_world.world_size = self.mock_world_size
        mock_vllm_world.rank_in_group = self.mock_rank_in_group
        mock_vllm_world.device_group = MagicMock()  

        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._EP = mock_vllm_ep
        vllm_ps._PP = mock_vllm_pp
        vllm_ps._WORLD = mock_vllm_world

        self.mlp = ReplicatedDeepseekMLP(hidden_size=self.mock_hidden_size,
                                         intermediate_size=self.mock_intermediate_size,
                                         hidden_act=self.mock_hidden_act,
                                         quant_config=mock_quant_config,
                                         reduce_results=self.mock_reduce_results,
                                         prefix=self.mock_prefix)   
    
    def tearDown(self):   
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._EP = None
        vllm_ps._PP = None
        vllm_ps._WORLD = None   

    def test_initialization(self):
        self.assertIsNotNone(self.mlp.gate_up_proj)      
        self.assertIsNotNone(self.mlp.down_proj)     
        self.assertIsNotNone(self.mlp.act_fn_obj)      

        self.assertTrue(hasattr(self.mlp, 'ep_size'))
        self.assertTrue(hasattr(self.mlp, 'global_rank'))
        self.assertTrue(hasattr(self.mlp, 'world_size'))
        self.assertTrue(hasattr(self.mlp, 'moe_all_to_all_group'))
        self.assertTrue(hasattr(self.mlp, 'moe_all_to_all_group_name'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group_rank'))
        self.assertTrue(hasattr(self.mlp, 'moe_rs_group_name'))

        self.assertEqual(self.mlp.tp_size, 1)
        self.assertEqual(self.mlp.ep_size, self.mock_ep_size)
        self.assertEqual(self.mlp.global_rank, self.mock_rank_in_group)
        self.assertEqual(self.mlp.world_size, self.mock_world_size)

    def test_act_fn(self):
        mock_input = [torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                      torch.randn(self.mock_bsz, dtype=torch.float32)]
        mock_quant_symbol = True

        self.mlp.gate_up_proj.weight_scale = MagicMock(return_value=torch.randn(4096, dtype=torch.float32))
        self.mlp.act_fn_obj.forward = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                                "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})

        result = self.mlp.act_fn(x=mock_input, quant_symbol=mock_quant_symbol)

        self.assertIsInstance(result, dict)
        self.assertIn('x_int8', result)
        self.assertIn('pertoken_scale', result)
        self.assertIsInstance(result['x_int8'], torch.Tensor)
        self.assertIsInstance(result['pertoken_scale'], torch.Tensor)

    def test_forward(self):
        mock_input = torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16)
        self.mlp.quant_symbol = True
        
        self.mlp.gate_up_proj.forward = MagicMock(return_value=[torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                                                    torch.randn(self.mock_bsz, dtype=torch.float32)])
        self.mlp.act_fn = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                          "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})
        self.mlp.down_proj.forward = MagicMock(return_value=[torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16), None])

        result = self.mlp.forward(x=mock_input)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], self.mock_bsz)
        self.assertEqual(result.shape[1], self.mock_hidden_size)


class test_ParallelDeepseekMLP(TestCase):
    @patch('omni.adaptors.vllm.distributed.parallel_state._MLP_TP',
    new_callable=lambda: MagicMock(spec=GroupCoordinator))    
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))
    def setUp(self, mock_quant_config, mock_omni_mlp_tp):
        self.mock_hidden_size = 7168
        self.mock_intermediate_size = 18432
        self.mock_hidden_act = "silu"
        self.mock_reduce_results = True
        self.mock_prefix = "model.layers.0.mlp"
        self.mock_bsz = 8
        self.mock_world_size = 8
        self.mock_rank_in_group = 0

        mock_quant_config = None

        mock_omni_mlp_tp.world_size = self.mock_world_size
        mock_omni_mlp_tp.rank_in_group = 0
        mock_omni_mlp_tp.device_group = MagicMock()  

        import omni.adaptors.vllm.distributed.parallel_state as omni_ps
        omni_ps._MLP_TP = mock_omni_mlp_tp

        self.mlp = ParallelDeepseekMLP(hidden_size=self.mock_hidden_size,
                                       intermediate_size=self.mock_intermediate_size,
                                       hidden_act=self.mock_hidden_act,
                                       quant_config=mock_quant_config,
                                       reduce_results=self.mock_reduce_results,
                                       prefix=self.mock_prefix,
                                       comm_group=mock_omni_mlp_tp)

    def tearDown(self):
        import omni.adaptors.vllm.distributed.parallel_state as omni_ps
        omni_ps._MLP_TP = None

    def test_initialization(self):
        self.assertIsNotNone(self.mlp.gate_up_proj)      
        self.assertIsNotNone(self.mlp.down_proj)     
        self.assertIsNotNone(self.mlp.act_fn_obj)      

        self.assertTrue(hasattr(self.mlp, 'prefix'))
        self.assertTrue(hasattr(self.mlp, 'quant_symbol'))
        self.assertTrue(hasattr(self.mlp, 'comm_group'))

        self.assertEqual(self.mlp.comm_group.rank_in_group, self.mock_rank_in_group)
        self.assertEqual(self.mlp.comm_group.world_size, self.mock_world_size)

    def test_act_fn(self):
        mock_input = [torch.randint(low=0, high=self.mock_bsz, size=(self.mock_bsz, 4096), dtype=torch.int32),
                      torch.randn(self.mock_bsz, dtype=torch.float32)]
        mock_quant_symbol = True

        self.mlp.gate_up_proj.weight_scale = MagicMock(return_value=torch.randn(4096, dtype=torch.float32))
        self.mlp.act_fn_obj.forward = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz, 2048), dtype=torch.int8),
                                                                "pertoken_scale":torch.randn(self.mock_bsz, dtype=torch.float32)})
        result = self.mlp.act_fn(x=mock_input, quant_symbol=mock_quant_symbol)

        self.assertIsInstance(result, dict)
        self.assertIn('x_int8', result)
        self.assertIn('pertoken_scale', result)
        self.assertIsInstance(result['x_int8'], torch.Tensor)
        self.assertIsInstance(result['pertoken_scale'], torch.Tensor)

    def test_forward(self):
        self.mock_bsz_gather = self.mock_world_size * self.mock_bsz
        mock_input = torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16)
        self.mlp.quant_symbol = True     

        self.mlp.comm_group.all_gather = MagicMock(side_effect=lambda data, dim: data.repeat(self.mock_world_size, 1))

        self.mlp.gate_up_proj.forward = MagicMock(return_value=[torch.randint(low=0, high=self.mock_bsz_gather, size=(self.mock_bsz_gather, 4608), dtype=torch.int32),
                                                    torch.randn(self.mock_bsz_gather, dtype=torch.float32)])
        self.mlp.act_fn = MagicMock(return_value={"x_int8": torch.randint(low=0, high=128, size=(self.mock_bsz_gather, 2304), dtype=torch.int8),
                                                  "pertoken_scale":torch.randn(self.mock_bsz_gather, dtype=torch.float32)})
        self.mlp.down_proj.forward = MagicMock(return_value=[torch.randn(self.mock_bsz_gather, self.mock_hidden_size, dtype=torch.bfloat16), None])

        self.mlp.comm_group.reduce_scatter = MagicMock(return_value=torch.randn(self.mock_bsz, self.mock_hidden_size, dtype=torch.bfloat16))

        result_1 = self.mlp.forward(x=mock_input, residual=None, attn_metadata=None, layerid=None)           

        self.assertIsInstance(result_1, torch.Tensor)
        self.assertEqual(result_1.shape[0], self.mock_bsz)
        self.assertEqual(result_1.shape[1], self.mock_hidden_size)


        mock_residual = torch.randn(self.mock_bsz_gather, self.mock_hidden_size, dtype=torch.bfloat16)
        result_2, residual = self.mlp.forward(x=mock_input, residual=mock_residual, attn_metadata=None, layerid=None)

        self.assertIsInstance(result_2, torch.Tensor)
        self.assertEqual(result_2.shape[0], self.mock_bsz)
        self.assertEqual(result_2.shape[1], self.mock_hidden_size)

        self.assertIsInstance(residual, torch.Tensor)
        self.assertEqual(residual.shape[0], self.mock_bsz_gather)
        self.assertEqual(residual.shape[1], self.mock_hidden_size)       


class test_DeepseekMoE(TestCase):
    @patch("torch.npu.device_count")
    @patch('vllm.distributed.parallel_state._PP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._WORLD',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._EP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch('vllm.config.QuantizationConfig',
    new_callable=lambda: MagicMock(spec=QuantizationConfig))
    @patch('transformers.PretrainedConfig')
    def setUp(self, 
              mock_pretrain_config,
              mock_quant_config,
              mock_model_extra_config,
              mock_vllm_ep,
              mock_vllm_pp,
              mock_vllm_world,
              mock_device_count,
              ):
        os.environ['ASCEND_PLATFORM'] = 'A3'
        self.layer_id = 17

        self.mock_decode_bsz = 8
        self.mock_prefill_bsz = 256
        self.mock_prefix = "model.layers.{}.mlp".format(self.layer_id)
        self.mock_hidden_size = 7168
        self.mock_ep_size = 8
        self.mock_world_size = 8
        self.mock_rank_in_group = 0

        mock_quant_config = None

        mock_pretrain_config.routed_scaling_factor = 2.5
        mock_pretrain_config.hidden_size = self.mock_hidden_size
        mock_pretrain_config.hidden_act = 'silu'
        mock_pretrain_config.num_experts_per_tok = 8
        mock_pretrain_config.topk_group = 4
        mock_pretrain_config.n_group = 8

        mock_pretrain_config.n_routed_experts = 256
        mock_pretrain_config.num_routed_experts = 256
        mock_pretrain_config.num_experts = 256
        mock_pretrain_config.num_shared_experts = 1
        mock_pretrain_config.n_shared_experts = 1
        mock_pretrain_config.num_dense_layers = 3
        mock_pretrain_config.first_k_dense_replace = 3

        mock_model_extra_config.parall_config.enable_share_expert_tp = False        
        mock_model_extra_config.parall_config.redundancy_shared_expert_num = 0

        mock_model_extra_config.task_config.enable_attn_ffn_disaggregation = False
        mock_model_extra_config.task_config.enable_omni_placement = False
        mock_model_extra_config.task_config.decode_gear_list = [2048]

        mock_model_extra_config.operator_opt_config.gmm_nz = True
        mock_model_extra_config.operator_opt_config.new_w4_op = False
        mock_model_extra_config.operator_opt_config.experts_pruning = False
        mock_model_extra_config.operator_opt_config.prefill_moe_all_to_all = True
        mock_model_extra_config.operator_opt_config.decode_experts_pruning = False
        mock_model_extra_config.operator_opt_config.decode_moe_dispatch_combine = True
        mock_model_extra_config.operator_opt_config.shared_expert_gate_up_prefetch = 28
        mock_model_extra_config.operator_opt_config.shared_expert_down_prefetch = 14

        mock_vllm_ep.world_size = self.mock_ep_size
        mock_vllm_ep.rank_in_group = self.mock_rank_in_group
        mock_vllm_ep.device_group = MagicMock()
        mock_vllm_pp.world_size = self.mock_ep_size
        mock_vllm_pp.rank_in_group = self.mock_rank_in_group
        mock_vllm_pp.device_group = MagicMock()    
        mock_vllm_world.world_size = self.mock_world_size
        mock_vllm_world.rank_in_group = self.mock_rank_in_group
        mock_vllm_world.device_group = MagicMock()  

        mock_device_count.return_value = self.mock_ep_size

        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._EP = mock_vllm_ep
        vllm_ps._PP = mock_vllm_pp
        vllm_ps._WORLD = mock_vllm_world

        self.moe = DeepseekMoE(config=mock_pretrain_config,
                               quant_config=mock_quant_config,
                               prefix=self.mock_prefix)

    def tearDown(self):   
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._EP = None
        vllm_ps._PP = None
        vllm_ps._WORLD = None

    def test_initialization(self):
        self.assertIsNotNone(self.moe.gate)      
        self.assertIsNotNone(self.moe.experts)     
        self.assertIsNotNone(self.moe.shared_experts)      

        self.assertTrue(hasattr(self.moe, 'prefix'))
        self.assertTrue(hasattr(self.moe, 'ep_size'))
        self.assertTrue(hasattr(self.moe, 'routed_scaling_factor'))
        self.assertTrue(hasattr(self.moe, 'device_count'))
        self.assertTrue(hasattr(self.moe, 'node_rank'))
        self.assertTrue(hasattr(self.moe, 'which_half'))
        self.assertTrue(hasattr(self.moe, 'n_routed_experts'))
        self.assertTrue(hasattr(self.moe, 'redundancy_shared_expert_num'))

        self.assertTrue(hasattr(self.moe, 'top_k'))
        self.assertTrue(hasattr(self.moe, 'use_grouped_topk'))
        self.assertTrue(hasattr(self.moe, 'renormalize'))
        self.assertTrue(hasattr(self.moe, 'topk_group'))
        self.assertTrue(hasattr(self.moe, 'num_expert_group'))
        self.assertTrue(hasattr(self.moe, 'custom_routing_function'))
        self.assertTrue(hasattr(self.moe, 'scoring_func'))
        self.assertTrue(hasattr(self.moe, 'n_shared_experts'))
        self.assertTrue(hasattr(self.moe, 'first_k_dense_replace'))
        self.assertTrue(hasattr(self.moe, 'n_redundant_experts'))
        self.assertTrue(hasattr(self.moe, 'shared_experts'))

        self.assertTrue(hasattr(self.moe, 'experts'))
        self.assertTrue(hasattr(self.moe, 'fake_experts'))
        self.assertTrue(hasattr(self.moe, 'global_rank'))
        self.assertTrue(hasattr(self.moe, 'planner'))
        self.assertTrue(hasattr(self.moe, 'moe_layer_idx'))
        self.assertTrue(hasattr(self.moe, 'expert_mapping'))
        self.assertTrue(hasattr(self.moe, 'attn_prefetch'))
        self.assertTrue(hasattr(self.moe, 'is_attn_die'))

        self.assertEqual(self.moe.experts_pruning, False)
        self.assertEqual(self.moe.decode_experts_pruning, False)

    def _create_hidden_and_residual(self, batch_size):
        hidden_states = torch.randn(batch_size, self.mock_hidden_size, dtype=torch.bfloat16)
        residual = torch.randn(batch_size, self.mock_hidden_size, dtype=torch.bfloat16)
        return hidden_states, residual

    def _setup_expert_selection(self, batch_size, fused_moe_mock):
        fused_moe_mock.select_experts.return_value = [
            torch.randn(batch_size, self.mock_ep_size, dtype=torch.float32),
            torch.randint(low=0, high=255, size=(batch_size, self.mock_ep_size), dtype=torch.int32),
            None,
        ]
        self.moe.experts_pruning_threshold = torch.tensor(
            [0, 0.01, 0.01, 0.01, 0.0665, 0.086, 0.125, 0.135]
        )


    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_decode(self, 
                     mock_attn_metadata,
                     mock_model_extra_config,
                    ):
        os.environ['ASCEND_PLATFORM'] = 'A3'
        mock_attn_metadata.prefill = None

        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_decode_bsz)

        mock_model_extra_config.task_config.enable_attn_ffn_disaggregation = False
        mock_model_extra_config.operator_opt_config.enable_round_pipeline_comm = False

        self.moe.is_init_gate = True

        self.moe._forward_decode_norm = MagicMock(return_value=[mock_hidden_states, mock_residual])

        result = self.moe.forward(hidden_states=mock_hidden_states,
                                  residual=mock_residual,
                                  attn_metadata=mock_attn_metadata,
                                  layer_id=self.layer_id,
                                  next_attention_weights=None)

        self.moe._forward_decode_norm.assert_called_once()
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertEqual(result[0].shape[0], self.mock_decode_bsz)
        self.assertEqual(result[0].shape[1], self.mock_hidden_size)
        self.assertEqual(result[1].shape[0], self.mock_decode_bsz)
        self.assertEqual(result[1].shape[1], self.mock_hidden_size)

    @patch("torch_npu.npu_format_cast")
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_prefill(self, 
                     mock_attn_metadata,
                     mock_model_extra_config,
                     mock_npu_format_cast,
                    ):
        os.environ['ASCEND_PLATFORM'] = 'A3'

        mock_attn_metadata.prefill = True

        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_prefill_bsz)

        mock_model_extra_config.task_config.enable_attn_ffn_disaggregation = False

        self.moe.is_init_gate = True
        self.moe._forward_prefill_norm = MagicMock(return_value=[mock_hidden_states, mock_residual])

        mock_npu_format_cast = MagicMock(side_effect=lambda data, dim: data)

        result = self.moe.forward(hidden_states=mock_hidden_states,
                                  residual=mock_residual,
                                  attn_metadata=mock_attn_metadata,
                                  layer_id=self.layer_id,
                                  next_attention_weights=None)

        self.moe._forward_prefill_norm.assert_called_once()
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertEqual(result[0].shape[0], self.mock_prefill_bsz)
        self.assertEqual(result[0].shape[1], self.mock_hidden_size)
        self.assertEqual(result[1].shape[0], self.mock_prefill_bsz)
        self.assertEqual(result[1].shape[1], self.mock_hidden_size)

    @patch('vllm.distributed.parallel_state._EP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch("torch_npu.npu_moe_finalize_routing")
    @patch('omni.layers.moe.deepseek_moe.FusedMoE',
    new_callable=lambda: MagicMock(spec=FusedMoE))
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_prefill_norm(self,
                                  mock_attn_metadata,
                                  mock_model_extra_config,
                                  mock_fused_moe,
                                  mock_npu_moe_finalize_routing,
                                  mock_vllm_ep
                                  ):
 
        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_prefill_bsz)
        mock_model_extra_config.operator_opt_config.prefill_moe_all_to_all = True

        self.moe.shared_experts.forward = MagicMock(side_effect=lambda data: data)

        self.moe.gate.forward = MagicMock(return_value=[torch.randn(self.mock_prefill_bsz, 256, dtype=torch.float32), None])

        self._setup_expert_selection(self.mock_prefill_bsz, mock_fused_moe)

        self.moe.experts.apply_expert_load_balance = MagicMock(side_effect=lambda topk_ids: topk_ids)

        self.moe.experts.forward = MagicMock(return_value=[torch.randn(self.mock_prefill_bsz, self.mock_hidden_size, dtype=torch.bfloat16),
                                                           torch.randn(self.mock_prefill_bsz*self.mock_ep_size, self.mock_hidden_size, dtype=torch.bfloat16),
                                                           None,
                                                           torch.randint(low=0, high=255, size=(self.mock_prefill_bsz*self.mock_ep_size,), dtype=torch.int32)])

        mock_npu_moe_finalize_routing.return_value = torch.randn(self.mock_prefill_bsz, self.mock_hidden_size, dtype=torch.bfloat16)

        result = self.moe._forward_prefill_norm(hidden_states=mock_hidden_states,
                                                residual=mock_residual,
                                                attn_metadata=mock_attn_metadata)

        self.moe.gate.forward.assert_called_once()
        self.moe.experts.apply_expert_load_balance.assert_called_once()
        self.moe.experts.forward.assert_called_once()

        mock_npu_moe_finalize_routing.assert_called_once()
        mock_fused_moe.select_experts.assert_called_once()

        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertEqual(result[0].shape[0], self.mock_prefill_bsz)
        self.assertEqual(result[0].shape[1], self.mock_hidden_size)
        self.assertEqual(result[1].shape[0], self.mock_prefill_bsz)
        self.assertEqual(result[1].shape[1], self.mock_hidden_size)

    @patch('vllm.distributed.parallel_state._EP',
    new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch("torch_npu.npu_moe_finalize_routing")
    @patch('omni.layers.moe.deepseek_moe.FusedMoE',
    new_callable=lambda: MagicMock(spec=FusedMoE))
    @patch('omni.models.config_loader.loader.model_extra_config',
    new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_decode_norm(self,
                                  mock_attn_metadata,
                                  mock_model_extra_config,
                                  mock_fused_moe,
                                  mock_npu_moe_finalize_routing,
                                  mock_vllm_ep
                                  ):

        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_decode_bsz)

        mock_model_extra_config.operator_opt_config.moe_multi_stream_tune = False

        self.moe.shared_experts.forward = MagicMock(side_effect=lambda data: data)

        self.moe.gate.forward = MagicMock(return_value=[torch.randn(self.mock_decode_bsz, 256, dtype=torch.float32), None])

        self._setup_expert_selection(self.mock_decode_bsz, mock_fused_moe)

        self.moe.experts.apply_expert_load_balance = MagicMock(side_effect=lambda topk_ids, best_topk_ids: topk_ids)

        self.moe.experts.forward = MagicMock(return_value=[torch.randn(self.mock_hidden_size, dtype=torch.bfloat16),
                                                           torch.randn(self.mock_hidden_size, dtype=torch.bfloat16),
                                                           None,
                                                           torch.randn(self.mock_hidden_size, dtype=torch.bfloat16),])

        mock_npu_moe_finalize_routing.return_value = torch.randn(self.mock_decode_bsz, self.mock_hidden_size, dtype=torch.bfloat16)


        result = self.moe._forward_decode_norm(hidden_states=mock_hidden_states,
                                                residual=mock_residual,
                                                attn_metadata=mock_attn_metadata,
                                                layer_id=self.layer_id,
                                                next_attention_weights=None)

        self.moe.gate.forward.assert_called_once()
        self.moe.experts.apply_expert_load_balance.assert_called_once()
        self.moe.experts.forward.assert_called_once()

        mock_npu_moe_finalize_routing.assert_called_once()
        mock_fused_moe.select_experts.assert_called_once()

        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertEqual(result[0].shape[0], self.mock_decode_bsz)
        self.assertEqual(result[0].shape[1], self.mock_hidden_size)
        self.assertEqual(result[1].shape[0], self.mock_decode_bsz)
        self.assertEqual(result[1].shape[1], self.mock_hidden_size)

    @patch("omni.layers.moe.deepseek_moe.tng.scope.npu_stream_switch", new_callable=lambda: MagicMock(side_effect=lambda *_, **__: nullcontext()))
    @patch("omni.layers.moe.deepseek_moe.tng.scope.npu_wait_tensor", new_callable=lambda: MagicMock(side_effect=lambda x, *_, **__: x))
    @patch("torch_npu.npu_prefetch")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_moe_distribute_combine_v2")
    @patch("torch_npu.npu_moe_distribute_dispatch_v2")
    @patch('omni.layers.moe.deepseek_moe.FusedMoE', new_callable=lambda: MagicMock(spec=FusedMoE))
    @patch('omni.models.config_loader.loader.model_extra_config', new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch('vllm.distributed.parallel_state._WORLD', new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch('vllm.distributed.parallel_state._EP', new_callable=lambda: MagicMock(spec=GroupCoordinatorGPU))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_decode_dispatch_combine(self,
                                             mock_attn_metadata,
                                             mock_vllm_ep,
                                             mock_vllm_world,
                                             mock_model_extra_config,
                                             mock_fused_moe,
                                             mock_dispatch,
                                             mock_combine,
                                             mock_swiglu,
                                             mock_grouped_matmul,
                                             mock_prefetch,
                                             mock_npu_wait_tensor,
                                             mock_stream_switch,
                                             ):
        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_decode_bsz)

        mock_attn_metadata.decode.best_topk = None
        mock_attn_metadata.decode.mc2_mask = torch.ones((self.mock_decode_bsz, self.moe.experts.top_k), dtype=torch.bool)

        mock_model_extra_config.operator_opt_config.moe_multi_stream_tune = False
        mock_model_extra_config.operator_opt_config.attn_prefetch = 0

        mock_vllm_ep.world_size = self.mock_ep_size
        mock_vllm_ep.rank_in_group = self.mock_rank_in_group
        mock_vllm_world.world_size = self.mock_world_size
        mock_vllm_world.rank_in_group = self.mock_rank_in_group

        topk_weights = torch.rand((self.mock_decode_bsz, self.moe.experts.top_k), dtype=torch.float32)
        topk_ids = torch.randint(low=0, high=self.moe.n_routed_experts, size=(self.mock_decode_bsz, self.moe.experts.top_k))
        mock_fused_moe.select_experts.return_value = (topk_weights, topk_ids, None)

        dispatch_expand_x = torch.randn((self.mock_decode_bsz * self.moe.experts.top_k, self.mock_hidden_size), dtype=torch.bfloat16)
        dispatch_expand_idx = torch.arange(self.mock_decode_bsz * self.moe.experts.top_k, dtype=torch.int32)
        expert_token_nums = torch.ones((self.mock_decode_bsz,), dtype=torch.int32)
        ep_recv_counts = torch.ones((self.mock_ep_size,), dtype=torch.int32)
        tp_recv_counts = torch.ones((self.mock_ep_size,), dtype=torch.int32)
        mock_dispatch.return_value = [dispatch_expand_x, torch.ones_like(dispatch_expand_x), dispatch_expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts]

        matmul_outputs = [torch.randn_like(dispatch_expand_x), torch.randn_like(dispatch_expand_x)]
        mock_grouped_matmul.side_effect = [[matmul_outputs[0]], [matmul_outputs[1]]]
        mock_swiglu.return_value = matmul_outputs[0]

        combined_output = torch.randn((self.mock_decode_bsz, self.mock_hidden_size), dtype=torch.bfloat16)
        mock_combine.return_value = combined_output

        self.moe.shared_experts.gate_up_proj.forward = MagicMock(return_value=(dispatch_expand_x, None))
        self.moe.shared_experts.act_fn = MagicMock(return_value=torch.randn_like(dispatch_expand_x))
        self.moe.shared_experts.down_proj.forward = MagicMock(return_value=(torch.randn(self.mock_decode_bsz, self.mock_hidden_size, dtype=torch.bfloat16), None))

        self.moe.experts.apply_expert_load_balance = MagicMock(side_effect=lambda topk_ids, best_topk_ids=None: topk_ids)

        result = self.moe._forward_decode_dispatch_combine(hidden_states=mock_hidden_states,
                                                           residual=mock_residual,
                                                           attn_metadata=mock_attn_metadata,
                                                           layer_id=self.layer_id,
                                                           next_attention_weights=None)

        mock_fused_moe.select_experts.assert_called_once()
        mock_dispatch.assert_called_once()
        mock_grouped_matmul.assert_called()
        mock_combine.assert_called_once()
        self.moe.shared_experts.down_proj.forward.assert_called_once()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result[0][0].shape, combined_output.shape)
        self.assertEqual(result[0][1].shape, combined_output.shape)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertEqual(result[1].shape, mock_residual.shape)

    @patch("torch_npu.npu_prefetch")
    @patch("omni.layers.moe.deepseek_moe.fused_experts_moe_dispatch_combine")
    @patch('omni.layers.moe.deepseek_moe.FusedMoE', new_callable=lambda: MagicMock(spec=FusedMoE))
    @patch('omni.models.config_loader.loader.model_extra_config', new_callable=lambda: MagicMock(spec=model_extra_config))
    @patch("vllm.attention.backends.abstract.AttentionMetadata")
    def test_forward_separate_expert_decode(self,
                                            mock_attn_metadata,
                                            mock_model_extra_config,
                                            mock_fused_moe,
                                            mock_dispatch_combine,
                                            mock_prefetch,
                                            ):
        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_decode_bsz)

        mock_model_extra_config.operator_opt_config.best_ep = False

        topk_weights = torch.rand((self.mock_decode_bsz, self.moe.experts.top_k), dtype=torch.float32)
        topk_ids = torch.randint(low=0, high=self.moe.n_routed_experts, size=(self.mock_decode_bsz, self.moe.experts.top_k))
        mock_fused_moe.select_experts.return_value = (topk_weights, topk_ids, None)

        dispatched_output = torch.randn_like(mock_hidden_states)
        mock_dispatch_combine.return_value = dispatched_output

        self.moe.w13_prefetch_size = 0
        self.moe.w2_prefetch_size = 0
        self.moe.gate_up_prefetch_size = 0
        self.moe.down_prefetch_size = 0

        self.moe.gate.forward = MagicMock(return_value=(torch.randn_like(mock_hidden_states), None))

        result = self.moe.forward_separate_expert_decode(hidden_states=mock_hidden_states,
                                                         residual=mock_residual,
                                                         attn_metadata=mock_attn_metadata,
                                                         next_attention_weights=None)

        self.moe.gate.forward.assert_called_once()
        mock_fused_moe.select_experts.assert_called_once()
        mock_dispatch_combine.assert_called_once()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], dispatched_output)
        self.assertIs(result[1], mock_residual)

    @patch('omni.layers.moe.deepseek_moe.get_ep_group')
    def test_forward_separate_expert_prefill(self, mock_ep_group):
        mock_hidden_states, mock_residual = self._create_hidden_and_residual(self.mock_prefill_bsz)
        mock_global_states = torch.randn(self.mock_prefill_bsz * 2, self.mock_hidden_size, dtype=torch.float32)

        reduce_shared = torch.randn_like(mock_hidden_states)
        reduce_final = torch.randn_like(mock_hidden_states)

        mock_ep_group.return_value.all_gather.return_value = mock_global_states
        mock_ep_group.return_value.reduce_scatter.side_effect = [reduce_shared, reduce_final]

        self.moe.shared_experts = None
        self.moe.experts = None

        output, residual = self.moe.forward_separate_expert_prefill(
            hidden_states=mock_hidden_states,
            residual=mock_residual,
            attn_metadata=MagicMock(),
        )

        mock_ep_group.return_value.all_gather.assert_called_once_with(mock_hidden_states, dim=0)
        self.assertEqual(mock_ep_group.return_value.reduce_scatter.call_count, 2)
        torch.testing.assert_close(output, reduce_final + reduce_shared)
        torch.testing.assert_close(residual, mock_residual)

    def test_chunked_gmm(self):
        mock_chunk_size = 2
        hidden_states = torch.randn(5, self.mock_hidden_size, dtype=torch.float32)
        topk_weights = torch.randn(5, self.mock_ep_size, dtype=torch.float32)
        topk_ids = torch.randint(low=0, high=self.mock_ep_size, size=(5, self.mock_ep_size), dtype=torch.int32)
        pertoken_scale = torch.randn(5, dtype=torch.float32)
        attn_metadata = MagicMock()

        def _expert_side_effect(hidden_states, **_):
            return hidden_states + 1

        self.moe.experts.forward = MagicMock(side_effect=_expert_side_effect)

        output = self.moe.chunked_gmm(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=pertoken_scale,
            attn_metadata=attn_metadata,
            chunk_size=mock_chunk_size,
        )

        expected = torch.cat([
            hidden_states[:2] + 1,
            hidden_states[2:4] + 1,
            hidden_states[4:] + 1,
        ])

        self.assertEqual(self.moe.experts.forward.call_count, 3)
        torch.testing.assert_close(output, expected)

    def test_chunked_gmm_no_split(self):
        hidden_states = torch.randn(2, self.mock_hidden_size, dtype=torch.float32)
        topk_weights = torch.randn(2, self.mock_ep_size, dtype=torch.float32)
        topk_ids = torch.randint(low=0, high=self.mock_ep_size, size=(2, self.mock_ep_size), dtype=torch.int32)
        pertoken_scale = torch.randn(2, dtype=torch.float32)
        attn_metadata = MagicMock()

        expected = torch.randn_like(hidden_states)
        self.moe.experts.forward = MagicMock(return_value=expected)

        output = self.moe.chunked_gmm(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=pertoken_scale,
            attn_metadata=attn_metadata,
            chunk_size=hidden_states.shape[0],
        )

        self.moe.experts.forward.assert_called_once_with(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=pertoken_scale,
            attn_metadata=attn_metadata,
        )

if __name__ == "__main__":
    unittest.main()