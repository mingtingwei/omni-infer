import os
import unittest
import torch

import torch_npu
import torchair
from unittest.mock import Mock, patch, MagicMock
from abc import abstractmethod
from typing import Optional
from vllm.model_executor.layers.quantization.base_config import (QuantizationConfig, 
                                                                 QuantizeMethodBase)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
    get_tp_group
)
from omni.layers.activation import SiluAndMul
from omni.layers.linear import MergedColumnParallelFlashCommLinear, RowParallelFlashCommLinear
from omni.models.config_loader.loader import model_extra_config
from omni.layers.fused_mlp.layer import FusedMLP, UnquantizedFusedMLPMethod


class TestFusedMLPLayer(unittest.TestCase):

    def setUp(self):
        '''Create unit test environment'''
        self.world_size_patcher = patch(
            'omni.layers.fused_mlp.layer.get_tensor_model_parallel_world_size',
            return_value=1
        )
        self.rank_patcher = patch(
            'omni.layers.fused_mlp.layer.get_tensor_model_parallel_rank',
            return_value=0
        )
        self.merged_linear_patcher = patch(
            'omni.layers.fused_mlp.layer.MergedColumnParallelFlashCommLinear'
        )
        self.row_linear_patcher = patch(
            'omni.layers.fused_mlp.layer.RowParallelFlashCommLinear'
        )
        self.all_reduce_patcher = patch(
            'omni.layers.fused_mlp.layer.tensor_model_parallel_all_reduce'
        )
        self.reduce_scatter_patcher = patch(
            'omni.layers.fused_mlp.layer.tensor_model_parallel_reduce_scatter'
        )
        self.all_gather_patcher = patch(
            'omni.layers.fused_mlp.layer.tensor_model_parallel_all_gather'
        )

        self.world_size_mock = self.world_size_patcher.start()
        self.rank_mock = self.rank_patcher.start()
        self.merged_linear_mock = self.merged_linear_patcher.start()
        self.row_linear_mock = self.row_linear_patcher.start()
        self.all_reduce_mock = self.all_reduce_patcher.start()
        self.reduce_scatter_mock = self.reduce_scatter_patcher.start()
        self.all_gather_mock = self.all_gather_patcher.start()

        self.gate_up_proj = MagicMock(return_value=(torch.ones(2, 2), None))
        self.gate_up_proj.tp_size = 1
        self.gate_up_proj.tp_rank = 0
        self.down_proj = MagicMock(return_value=(torch.ones(2, 2), None))
        self.down_proj.tp_size = 1
        self.down_proj.tp_rank = 0

        self.merged_linear_mock.return_value = self.gate_up_proj
        self.row_linear_mock.return_value = self.down_proj

        self.sample_input = torch.randn(2, 2)

    def tearDown(self):
        '''clear unit test environment'''
        self.world_size_patcher.stop()
        self.rank_patcher.stop()
        self.merged_linear_patcher.stop()
        self.row_linear_patcher.stop()
        self.all_reduce_patcher.stop()
        self.reduce_scatter_patcher.stop()
        self.all_gather_patcher.stop()

    def test_init_with_unsupported_activation(self):
        '''Hidden activation other than silu should raise ValueError'''
        with self.assertRaises(ValueError):
            FusedMLP(4, 8, hidden_act="relu")

    def test_forward_calls_all_reduce_when_needed(self):
        '''When tp_size>1 and reduce_type is AR, all_reduce should be called'''
        self.world_size_mock.return_value = 2
        self.down_proj.tp_size = 2
        mlp = FusedMLP(4, 8, hidden_act="silu")

        mocked_output = torch.ones(2, 2)
        quant_method_mock = MagicMock()
        quant_method_mock.apply.return_value = mocked_output
        mlp.quant_method = quant_method_mock

        self.all_reduce_mock.return_value = mocked_output * 2

        result = mlp(self.sample_input, reduce_type="AR")

        quant_method_mock.apply.assert_called_once_with(
            mlp,
            self.sample_input,
            x_transform=None,
            is_prefill=True
        )
        self.all_reduce_mock.assert_called_once_with(mocked_output)
        self.reduce_scatter_mock.assert_not_called()
        self.assertTrue(torch.equal(result, mocked_output * 2))

    def test_forward_calls_reduce_scatter(self):
        '''When tp_size>1 and reduce_type is RS, reduce_scatter should be called'''
        self.world_size_mock.return_value = 2
        self.down_proj.tp_size = 2
        mlp = FusedMLP(4, 8, hidden_act="silu")

        quant_method_mock = MagicMock()
        quant_method_mock.apply.return_value = torch.ones(2, 2)
        mlp.quant_method = quant_method_mock

        reduced = torch.full((2, 2), 3.0)
        self.reduce_scatter_mock.return_value = reduced

        result = mlp(self.sample_input, reduce_type="RS")

        quant_method_mock.apply.assert_called_once()
        self.reduce_scatter_mock.assert_called_once()
        self.all_reduce_mock.assert_not_called()
        self.assertTrue(torch.equal(result, reduced))


class TestUnquantizedFusedMLPMethod(unittest.TestCase):

    def setUp(self):
        '''Create unit test environment'''
        self.all_gather_patcher = patch(
            'omni.layers.fused_mlp.layer.tensor_model_parallel_all_gather'
        )
        self.all_gather_mock = self.all_gather_patcher.start()
        self.all_gather_mock.return_value = "gathered_input"

    def tearDown(self):
        '''clear unit test environment'''
        self.all_gather_patcher.stop()

    def test_apply_with_all_gather(self):
        '''AG transform should trigger tensor parallel all gather'''
        layer = MagicMock()
        layer.act_fn = MagicMock(return_value="activated")
        layer.gate_up_proj = MagicMock(return_value=("gate_up", None))
        layer.down_proj = MagicMock(return_value=("down_out", None))

        method = UnquantizedFusedMLPMethod()
        result = method.apply(layer, "input_tensor", x_transform="AG")

        self.all_gather_mock.assert_called_once_with("input_tensor", dim=0)
        layer.gate_up_proj.assert_called_once_with("gathered_input", x_transform=None)
        layer.act_fn.assert_called_once_with("gate_up")
        layer.down_proj.assert_called_once_with("activated", reduce_type=None)
        self.assertEqual(result, "down_out")


if __name__ == "__main__":
    unittest.main()