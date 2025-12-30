import torch
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch


def _build_model_extra_config():
    return SimpleNamespace(
        task_config=SimpleNamespace(
            enable_attn_ffn_disaggregation=False,
            enable_omni_placement=False,
        ),
        parall_config=SimpleNamespace(
            attn_dies=0,
            redundancy_shared_expert_num=0,
        ),
        operator_opt_config=SimpleNamespace(
            best_ep=False,
            moe_multi_stream_tune=False,
            decode_moe_dispatch_combine=False,
        ),
    )


class TestFusedMoELayer(TestCase):
    def setUp(self):
        self.mock_ep_group = MagicMock()
        self.mock_ep_group.world_size = 1
        self.mock_ep_group.rank_in_group = 0
        self.mock_ep_group.device_group = MagicMock()

        self.mock_world_group = MagicMock()
        self.mock_world_group.world_size = 1
        self.mock_world_group.rank_in_group = 0
        self.mock_world_group.device_group = MagicMock()

        self.mock_pp_group = MagicMock()
        self.mock_pp_group.world_size = 1
        self.mock_pp_group.rank_in_group = 0
        self.mock_pp_group.device_group = MagicMock()

        self.model_extra_config = _build_model_extra_config()

        self.patchers = [
            patch("omni.layers.moe.fused_moe.layer.get_ep_group", return_value=self.mock_ep_group),
            patch("omni.layers.moe.fused_moe.layer.get_world_group", return_value=self.mock_world_group),
            patch("omni.layers.moe.fused_moe.layer.get_pp_group", return_value=self.mock_pp_group),
            patch("omni.layers.moe.fused_moe.layer.model_extra_config", self.model_extra_config),
            patch("omni.layers.moe.fused_moe.layer.current_platform", SimpleNamespace(device_type="cpu")),
        ]

        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    def _build_quant_config(self, quant_method):
        quant_config = MagicMock()
        quant_config.get_quant_method.return_value = quant_method
        return quant_config

    def test_apply_expert_load_balance_uses_planner(self):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        planned_topk = torch.randint(low=0, high=4, size=(2, 2))
        planner = MagicMock()
        planner.plan.return_value = (None, planned_topk, None)

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_config = self._build_quant_config(quant_method)

        layer = FusedMoE(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            quant_config=quant_config,
            planner=planner,
            moe_layer_idx=0,
        )

        topk_ids = torch.zeros(2, 2, dtype=torch.int32)
        balanced = layer.apply_expert_load_balance(topk_ids, best_topk_ids=None)

        planner.plan.assert_called_once()
        self.assertTrue(torch.equal(balanced, planned_topk))

    def test_apply_expert_load_balance_best_ep_prefill(self):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        self.model_extra_config.operator_opt_config.best_ep = True

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_config = self._build_quant_config(quant_method)

        layer = FusedMoE(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            quant_config=quant_config,
        )
        layer.is_prefill_instance = True

        topk_ids = torch.zeros(2, 8, dtype=torch.int32)
        balanced = layer.apply_expert_load_balance(topk_ids, best_topk_ids=None)

        self.assertEqual(balanced.shape, topk_ids.shape)
        self.assertTrue(torch.equal(balanced[0], torch.arange(8, dtype=torch.int32)))

    @patch("omni.layers.moe.fused_moe.layer.get_forward_context")
    def test_select_experts_uses_custom_function(self, mock_forward_ctx):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        mock_forward_ctx.return_value = SimpleNamespace(attn_metadata=None)

        expected_weights = torch.randn(2, 2)
        expected_ids = torch.ones(2, 2, dtype=torch.int32)
        expected_rows = torch.zeros(2, 2, dtype=torch.int32)

        def _custom(hidden_states, gating_output, topk, renormalize):
            return expected_weights, expected_ids, expected_rows

        weights, ids, rows = FusedMoE.select_experts(
            hidden_states=torch.randn(2, 2),
            router_logits=torch.randn(2, 2),
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            custom_routing_function=_custom,
            scoring_func="softmax",
        )

        self.assertTrue(torch.equal(weights, expected_weights))
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(rows, expected_rows))

    @patch("omni.layers.moe.fused_moe.layer.get_forward_context")
    @patch("omni.layers.moe.fused_moe.layer.grouped_topk")
    def test_select_experts_grouped_topk_scales_weights(self, mock_grouped_topk, mock_forward_ctx):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        mock_forward_ctx.return_value = SimpleNamespace(attn_metadata=None)
        routed_scaling = torch.tensor(0.5)
        expected_weights = torch.ones(2, 2)
        expected_ids = torch.full((2, 2), 3, dtype=torch.int32)
        expected_rows = torch.arange(4, dtype=torch.int32).view(2, 2)
        mock_grouped_topk.return_value = (expected_weights, expected_ids, expected_rows)

        weights, ids, rows = FusedMoE.select_experts(
            hidden_states=torch.randn(2, 2),
            router_logits=torch.randn(2, 2),
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            topk_group=1,
            num_expert_group=1,
            routed_scaling_factor=routed_scaling,
        )

        mock_grouped_topk.assert_called_once()
        self.assertTrue(torch.equal(weights, expected_weights * routed_scaling))
        self.assertTrue(torch.equal(ids, expected_ids))
        expected_arange_rows = torch.arange(expected_ids.numel(), dtype=torch.int32).view(-1, expected_ids.shape[0]).transpose(0, 1)
        self.assertTrue(torch.equal(rows, expected_arange_rows))

    @patch("omni.layers.moe.fused_moe.layer.get_forward_context")
    @patch("omni.layers.moe.fused_moe.layer.torch_npu.npu_moe_gating_top_k")
    def test_select_experts_grouped_topk_with_bias_uses_npu_path(self, mock_topk, mock_forward_ctx):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        mock_forward_ctx.return_value = SimpleNamespace(attn_metadata=None)
        mock_weights = torch.full((2, 1), 0.3)
        mock_ids = torch.arange(2, dtype=torch.int32).view(2, 1)
        mock_topk.return_value = (mock_weights, mock_ids, None)

        weights, ids, rows = FusedMoE.select_experts(
            hidden_states=torch.randn(2, 3),
            router_logits=torch.randn(2, 3),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=2,
            num_expert_group=4,
            e_score_correction_bias=torch.tensor(0.1),
            routed_scaling_factor=torch.tensor(1.0),
        )

        mock_topk.assert_called_once()
        self.assertTrue(torch.equal(weights, mock_weights))
        self.assertTrue(torch.equal(ids, mock_ids))
        self.assertTrue(torch.equal(rows, torch.tensor([[0], [1]], dtype=torch.int32)))

    def test_forward_reduces_results_when_needed(self):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        self.mock_ep_group.world_size = 1
        reduced = torch.full((3, 4), 5.0)
        self.mock_ep_group.all_reduce = MagicMock(return_value=reduced)

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_method.apply = MagicMock(return_value=torch.ones(3, 4))
        quant_config = self._build_quant_config(quant_method)

        layer = FusedMoE(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            quant_config=quant_config,
            reduce_results=True,
            tp_size=2,
        )

        out = layer.forward(
            hidden_states=torch.randn(3, 8),
            topk_weights=torch.randn(3, 2),
            topk_ids=torch.ones(3, 2, dtype=torch.int32),
            pertoken_scale=None,
            attn_metadata=MagicMock(),
        )

        quant_method.apply.assert_called_once()
        self.mock_ep_group.all_reduce.assert_called_once()
        self.assertTrue(torch.equal(out, reduced))

    @patch("omni.layers.moe.fused_moe.layer.tng.scope.npu_wait_tensor")
    def test_apply_expert_load_balance_waits_for_best_topk(self, mock_wait):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        self.model_extra_config.operator_opt_config.best_ep = True
        self.model_extra_config.operator_opt_config.moe_multi_stream_tune = True
        self.model_extra_config.operator_opt_config.decode_moe_dispatch_combine = False

        balanced = torch.arange(4, dtype=torch.int32).view(2, 2)
        mock_wait.return_value = balanced

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_config = self._build_quant_config(quant_method)

        layer = FusedMoE(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            quant_config=quant_config,
        )
        layer.is_prefill_instance = False

        topk_ids = torch.zeros(2, 2, dtype=torch.int32)
        best_topk_ids = torch.full_like(topk_ids, 7)

        result = layer.apply_expert_load_balance(topk_ids, best_topk_ids=best_topk_ids)

        mock_wait.assert_called_once_with(best_topk_ids, topk_ids)
        self.assertTrue(torch.equal(result, balanced))

    def test_init_grouped_topk_missing_args_raises(self):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_config = self._build_quant_config(quant_method)

        with self.assertRaises(RuntimeError):
            FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=8,
                intermediate_size=16,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=None,
                topk_group=None,
            )

    def test_init_non_softmax_scoring_requires_grouped_topk(self):
        from omni.layers.moe.fused_moe.layer import FusedMoE

        quant_method = MagicMock()
        quant_method.create_weights = MagicMock()
        quant_config = self._build_quant_config(quant_method)

        with self.assertRaises(ValueError):
            FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=8,
                intermediate_size=16,
                quant_config=quant_config,
                scoring_func="sigmoid",
            )

if __name__ == "__main__":
    unittest.main()