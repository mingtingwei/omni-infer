import torch
import torch_npu
import unittest
from unittest.mock import patch, MagicMock, DEFAULT
from torch.nn.parameter import Parameter
from vllm.platforms import current_platform
from omni.layers.vocab_parallel_embedding  import ParallelLMHead, VocabParallelEmbedding

DEFAULT_VOCAB_PADDING_SIZE = 64

def divide(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)
    return numerator // denominator

class TestParallelLMHead(unittest.TestCase):
    def setUp(self):
        '''initialize the test environment'''

        self.hidden_dim = 8
        self.num_embeddings = 100
        self.embedding_dim = 8
        self.params_dtype = torch.float32
        self.org_num_embeddings = 80
        self.padding_size = DEFAULT_VOCAB_PADDING_SIZE
        self.parallel_lmhead = True

        '''Use patch to mock distributed functions.'''
        self.vllm_embedding_patcher = patch.multiple(
            'vllm.model_executor.layers.vocab_parallel_embedding',
            get_tensor_model_parallel_rank=DEFAULT,
            get_tensor_model_parallel_world_size=DEFAULT,
            divide=DEFAULT
        )
        self.omni_embedding_patcher = patch.multiple(
            'omni.layers.vocab_parallel_embedding',
            get_tensor_model_parallel_rank = DEFAULT,
            get_tensor_model_parallel_world_size = DEFAULT,
            get_world_group = DEFAULT,
            get_local_world_group = DEFAULT,
            tensor_model_parallel_all_reduce = DEFAULT,
        )
        
        self.vllm_embedding_mocks = self.vllm_embedding_patcher.start()
        self.omni_embedding_mocks = self.omni_embedding_patcher.start()

        self.vllm_embedding_mocks["get_tensor_model_parallel_rank"].side_effect = lambda : self.tp_rank
        self.vllm_embedding_mocks["get_tensor_model_parallel_world_size"].side_effect = lambda : self.tp_size
        self.vllm_embedding_mocks["divide"].side_effect = divide
        self.omni_embedding_mocks["get_tensor_model_parallel_rank"].side_effect = lambda : self.tp_rank
        self.omni_embedding_mocks["get_tensor_model_parallel_world_size"].side_effect = lambda : self.tp_size
        self.omni_embedding_mocks["get_world_group"].side_effect = lambda : type('tmp', (), {'local_rank':self.tp_rank})
        self.omni_embedding_mocks["get_local_world_group"].side_effect = lambda : type('tmp', (), {'world_size':self.tp_size})
        self.omni_embedding_mocks["tensor_model_parallel_all_reduce"].side_effect = lambda x : x

    def tearDown(self):
        '''clear unit test environment'''
        self.vllm_embedding_patcher.stop()
        self.omni_embedding_patcher.stop()

    def test_initialization(self):
        """unit test initialization"""
        for tp_size in [1, 2, 4, 8, 16, 32]:
            self.tp_size = tp_size
            for tp_rank in range(tp_size):
                self.tp_rank = tp_rank
                for bias in [True, False]:

                    lm_head = ParallelLMHead(
                        num_embeddings = self.num_embeddings,
                        embedding_dim = self.embedding_dim,
                        params_dtype = self.params_dtype,
                        org_num_embeddings = self.org_num_embeddings,
                        padding_size = self.padding_size,
                        quant_config = None,
                        prefix="test",
                        parallel_lmhead = self.parallel_lmhead,
                        bias = bias
                    )

                    self.assertEqual(lm_head.num_embeddings, self.num_embeddings)
                    self.assertEqual(lm_head.embedding_dim, self.embedding_dim)
                    self.assertEqual(lm_head.org_vocab_size, self.org_num_embeddings)
                    self.assertEqual(lm_head.padding_size, self.padding_size)                  
                    self.assertTrue(lm_head.parallel_lmhead)
                    
                    add_vocab_size = self.num_embeddings - lm_head.org_vocab_size
                    expected_shard_size = (lm_head.org_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                    expected_shard_size += (add_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                    expected_shard_size = expected_shard_size // self.tp_size
                    self.assertEqual(lm_head.num_embeddings_per_partition, expected_shard_size)

                    self.assertIsNotNone(lm_head.weight)
                    self.assertEqual(lm_head.weight.shape, (lm_head.num_embeddings_per_partition, self.hidden_dim))      

if __name__ == '__main__':
    unittest.main()