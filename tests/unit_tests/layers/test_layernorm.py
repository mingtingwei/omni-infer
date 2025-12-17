# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
import torch
# import torch_npu
from unittest.mock import Mock, patch, MagicMock
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import get_tp_group

from omni.layers.layernorm import RMSNorm


class TestRMSNorm(unittest.TestCase):

    def setUp(self):
        '''initialize the test environment'''
        self.mock_hidden_size = 1024
        self.mock_tp_size = 8

        self.mock_tp_group = MagicMock()
        self.mock_tp_group.all_gather = MagicMock(side_effect=lambda x, dim: x)
        self.mock_tp_group.world_size = self.mock_tp_size
        self.mock_tp_group.rank_in_group = 0
        
        '''mock the Tensor parallel test environment'''
        self.tp_size_mock = patch('vllm.distributed.parallel_state.get_tensor_model_parallel_world_size', 
                                  return_value=self.mock_tp_size)
        self.tp_rank_mock = patch('vllm.distributed.parallel_state.get_tensor_model_parallel_rank', return_value=0)
        self.tp_group_mock = patch('vllm.distributed.get_tp_group', return_value=self.mock_tp_group)
        
        self.tp_size_mock.start()
        self.tp_rank_mock.start()
        self.tp_group_mock.start()
        
        import vllm.distributed.parallel_state as ps
        ps._TP = self.mock_tp_group
        
        self.rms_norm = RMSNorm(self.mock_hidden_size, eps=1e-6)
        
        self.npu_add_rms_norm_mock = MagicMock()
        self.npu_add_rms_norm_mock.return_value = (torch.randn(self.mock_tp_size, self.mock_hidden_size), None, torch.randn(self.mock_tp_size, self.mock_hidden_size))
        self.npu_rms_norm_mock = MagicMock()
        self.npu_rms_norm_mock.return_value = (torch.randn(self.mock_tp_size, self.mock_hidden_size),)
        self.npu_dynamic_quant_mock = MagicMock()
        self.npu_dynamic_quant_mock.return_value = (torch.randn(self.mock_tp_size, self.mock_hidden_size), torch.randn(self.mock_tp_size))
        
        self.patch1 = patch('torch_npu.npu_add_rms_norm', new=self.npu_add_rms_norm_mock)
        self.patch2 = patch('torch_npu.npu_rms_norm', new=self.npu_rms_norm_mock)
        self.patch3 = patch('torch_npu.npu_dynamic_quant', new=self.npu_dynamic_quant_mock)
        
        self.patch1.start()
        self.patch2.start()
        self.patch3.start()

    def tearDown(self):
        '''clear test environment'''
        self.tp_size_mock.stop()
        self.tp_rank_mock.stop()
        self.tp_group_mock.stop()
        self.patch1.stop()
        self.patch2.stop()
        self.patch3.stop()
        
        import vllm.distributed.parallel_state as ps
        ps._TP = None

    def test_initialization(self):
        '''Test RMSNorm initialization'''
        self.assertEqual(self.rms_norm.hidden_size, self.mock_hidden_size)
        self.assertEqual(self.rms_norm.variance_epsilon, 1e-6)
        
        self.assertTrue(hasattr(self.rms_norm, 'weight'))
        self.assertEqual(self.rms_norm.weight.shape, (self.mock_hidden_size,))

        self.assertTrue(hasattr(self.rms_norm, 'variance_epsilon'))
        self.assertEqual(self.rms_norm.variance_epsilon, 1e-6)
        custom_eps = 1e-5
        custom_norm = RMSNorm(self.mock_hidden_size, eps=custom_eps)
        self.assertEqual(custom_norm.variance_epsilon, custom_eps)

        custom_norm = RMSNorm(self.mock_hidden_size, eps=1e-6, has_weight=False, dtype=torch.bfloat16)
        self.assertEqual(custom_norm.weight.dtype, torch.bfloat16)

    def test_forward_with_residual_basic(self):
        '''Test forward propagation with residual connections'''
        self.npu_add_rms_norm_mock.reset_mock()
        x = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        residual = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        
        result = self.rms_norm(x, residual=residual)
        self.npu_add_rms_norm_mock.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_forward_with_residual_quant(self):
        '''Test forward propagation with residual connections and quantization methods'''
        self.npu_add_rms_norm_mock.reset_mock()
        self.npu_dynamic_quant_mock.reset_mock()
        x = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        residual = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        
        result = self.rms_norm(x, residual=residual, quant_symbol=True)
        self.npu_add_rms_norm_mock.assert_called_once()
        self.npu_dynamic_quant_mock.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIn('x_int8', result[0])
        self.assertIn('pertoken_scale', result[0])

    def test_forward_with_residual_all_gather(self):
        '''Test forward propagation with residual connections and all_gather'''
        self.npu_add_rms_norm_mock.reset_mock()
        x = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        residual = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        
        result = self.rms_norm(x, residual=residual, quant_symbol="AG")
        self.npu_add_rms_norm_mock.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_forward_without_residual_basic(self):
        '''Test forward propagation without residual connections'''
        self.npu_rms_norm_mock.reset_mock()
        x = torch.randn(self.mock_tp_size, self.mock_hidden_size)
        
        result = self.rms_norm(x)
        self.npu_rms_norm_mock.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)

if __name__ == "__main__":
    unittest.main()