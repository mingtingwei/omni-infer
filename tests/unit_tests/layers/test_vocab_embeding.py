import unittest
from unittest.mock import patch, DEFAULT
import torch
import torch_npu
from torch.nn.parameter import Parameter
from vllm.platforms import current_platform
from omni.layers.vocab_parallel_embedding import  VocabParallelEmbedding

device = "npu"

def divide(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)
    return numerator // denominator

def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

class TestVocabParallelEmbedding(unittest.TestCase):

    def setUp(self):
        '''initialize the test environment'''
        self.tp_rank = 0 
        self.tp_size = 2 
        
        '''initialize input parameters'''
        self.num_embeddings = 100
        self.embedding_dim = 8
        self.org_vocab_size = 80
        self.params_dtype = torch.float32
        self.padding_size = 64  

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
            tensor_model_parallel_all_reduce = DEFAULT
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
        '''Test the initialization of VocabParallelEmbedding'''
        for tp_size in [1, 2, 4, 8, 16, 32]:
            self.tp_size = tp_size
            for tp_rank in range(tp_size):
                self.tp_rank = tp_rank
                for parallel_lmhead in [True, False]:
                    
                    embedding_layer = VocabParallelEmbedding(
                        num_embeddings = self.num_embeddings,
                        embedding_dim = self.embedding_dim,
                        params_dtype = self.params_dtype,
                        org_num_embeddings = self.org_vocab_size,
                        padding_size = self.padding_size,
                        quant_config = None,
                        prefix="test",
                        parallel_lmhead = parallel_lmhead
                    )

                    self.assertEqual(embedding_layer.num_embeddings, self.num_embeddings)
                    self.assertEqual(embedding_layer.embedding_dim, self.embedding_dim)
                    self.assertEqual(embedding_layer.org_vocab_size, self.org_vocab_size)
                    self.assertEqual(embedding_layer.tp_size, self.tp_size)

                    add_vocab_size = self.num_embeddings - self.org_vocab_size
                    expected_shard_size = (self.org_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                    expected_shard_size += (add_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                    expected_shard_size = expected_shard_size // self.tp_size

                    self.assertEqual(embedding_layer.num_embeddings_per_partition, expected_shard_size)
                    self.assertTrue(hasattr(embedding_layer, 'weight'))
                    self.assertEqual(embedding_layer.weight.shape, (expected_shard_size, self.embedding_dim))

    def test_weight_loader(self):
        for tp_size in [1, 2, 4, 8, 16, 32]:
            self.tp_size = tp_size
            for tp_rank in range(tp_size):
                self.tp_rank = tp_rank
                embedding_layer = VocabParallelEmbedding(
                    num_embeddings = self.num_embeddings,
                    embedding_dim = self.embedding_dim,
                    params_dtype = self.params_dtype,
                    org_num_embeddings = self.org_vocab_size,
                    padding_size = self.padding_size,
                    quant_config = None,
                    prefix = "test"
                )

                embedding_layer.weight = Parameter(torch.zeros(embedding_layer.num_embeddings_per_partition, self.embedding_dim))
                loaded_weight = torch.rand(self.org_vocab_size, self.embedding_dim, dtype = self.params_dtype)  
                embedding_layer.weight.output_dim = 0 
                embedding_layer.weight_loader(embedding_layer.weight, loaded_weight.clone())

                start_index = embedding_layer.shard_indices.org_vocab_start_index
                end_index = embedding_layer.shard_indices.org_vocab_end_index
                
                padding_org_num = (self.org_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                per_partition_org_vocab_num = divide(padding_org_num, self.tp_size)
                padded_org_vocab_start_index = per_partition_org_vocab_num * self.tp_rank
                padded_org_vocab_end_index = padded_org_vocab_start_index + per_partition_org_vocab_num
                org_vocab_start_index = min(padded_org_vocab_start_index,
                                            self.org_vocab_size)
                org_vocab_end_index = min(padded_org_vocab_end_index, self.org_vocab_size)
                self.assertEqual(org_vocab_start_index, start_index)
                self.assertEqual(org_vocab_end_index, end_index)
                
                expected_loaded_chunk = loaded_weight.narrow(0, start_index, end_index - start_index)

                self.assertTrue(torch.allclose(embedding_layer.weight.data[: expected_loaded_chunk.size(0)], expected_loaded_chunk))

                self.assertTrue(torch.all(embedding_layer.weight.data[expected_loaded_chunk.size(0) :] == 0))

    def test_embedding(self):
        input_size = 256
        for tp_size in [1, 2, 4, 8, 16, 32]:
            self.tp_size = tp_size
            for tp_rank in range(tp_size):
                self.tp_rank = tp_rank
                embedding_layer = VocabParallelEmbedding(
                    num_embeddings = self.num_embeddings,
                    embedding_dim = self.embedding_dim,
                    params_dtype = self.params_dtype,
                    org_num_embeddings = self.org_vocab_size,
                    padding_size = self.padding_size,
                    quant_config = None,
                    prefix = "test"
                ).to(device)
                
                padding_org_num = (self.org_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
                added_num = self.num_embeddings - self.org_vocab_size
                padding_added_num = (added_num + self.padding_size - 1) // self.padding_size * self.padding_size
                padding_num = padding_org_num + padding_added_num
                
                embedding_layer.weight = Parameter(torch.zeros(divide(padding_num, self.tp_size), self.embedding_dim).to(device), requires_grad=False)
                loaded_weight = torch.rand(self.org_vocab_size, self.embedding_dim, dtype = self.params_dtype, device = device)

                embedding_layer.weight.output_dim = 0

                embedding_layer.weight_loader(embedding_layer.weight, loaded_weight.clone())

                test_input = (torch.rand(input_size, device=device) * self.org_vocab_size).to(torch.long)
                result = embedding_layer.forward(test_input)

                start_index = embedding_layer.shard_indices.org_vocab_start_index
                end_index = embedding_layer.shard_indices.org_vocab_end_index
                expected_weight = loaded_weight.narrow(0, start_index, end_index - start_index)
                
                masked_input, input_mask = get_masked_input_and_mask(
                        test_input, embedding_layer.shard_indices.org_vocab_start_index,
                        embedding_layer.shard_indices.org_vocab_end_index,
                        embedding_layer.shard_indices.num_org_vocab_padding,
                        embedding_layer.shard_indices.added_vocab_start_index,
                        embedding_layer.shard_indices.added_vocab_end_index)
                
                expected_output = torch.nn.functional.embedding(masked_input, expected_weight)
                
                expected_output.masked_fill_(input_mask.unsqueeze(-1), 0)
                
                self.assertTrue(torch.allclose(expected_output, result))
    
    def test_embedding_outlier(self):

        input_size = 256
        self.tp_size = 4
        self.tp_rank = 1
        for outlier in [self.org_vocab_size, -2 * self.org_vocab_size]:
            embedding_layer = VocabParallelEmbedding(
                num_embeddings = self.num_embeddings,
                embedding_dim = self.embedding_dim,
                params_dtype = self.params_dtype,
                org_num_embeddings = self.org_vocab_size,
                padding_size = self.padding_size,
                quant_config = None,
                prefix = "test"
            ).to(device)
            
            padding_org_num = (self.org_vocab_size + self.padding_size - 1) // self.padding_size * self.padding_size
            added_num = self.num_embeddings - self.org_vocab_size
            padding_added_num = (added_num + self.padding_size - 1) // self.padding_size * self.padding_size
            padding_num = padding_org_num + padding_added_num
            
            embedding_layer.weight = Parameter(torch.zeros(divide(padding_num, self.tp_size), self.embedding_dim).to(device), requires_grad=False)
            loaded_weight = torch.rand(self.org_vocab_size, self.embedding_dim, dtype = self.params_dtype, device = device)  # 模拟从外部加载的权重

            embedding_layer.weight.output_dim = 0 
            embedding_layer.weight_loader(embedding_layer.weight, loaded_weight.clone())

            test_input = (torch.rand(input_size, device=device) * self.org_vocab_size + outlier).to(torch.long)
            result = embedding_layer.forward(test_input)

            self.assertTrue(torch.all(result == 0))


if __name__ == '__main__':
    unittest.main()