## 分布式测试概述

我们实现了持久化工作进程池来实现单节点双die高性能分布式测试。

不再为每个测试用例初始化环境，而是在模块开始时启动两个工作进程，分别对应两个die。这些工作进程持续运行，接受任务，并在已初始化的分布式环境中执行。这样能够显著提高性能。

## 如何添加新测试

添加分布式测试需要将代码分为两部分：

测试payload函数：在工作进程上运行（Rank 0、Rank 1 等）。

测试驱动：在主机（Pytest）上运行，将任务分发给工作进程。

注：如果不需要分布式测试，不需要引入该框架。

### 步骤 1：定义逻辑函数
包含实际的PyTorch/Omni代码。

黄金法则：
1. 必须定义在顶层：函数必须在全局作用域定义（以便可以序列化）。

```python
def _logic_my_new_feature(device, local_rank, world_size, input_size, dtype):
    """
    此函数在持久化工作进程内部运行。
    """
    # [关键] 1. 重置随机种子：工作进程长期存活；防止RNG漂移。
    torch.manual_seed(0)
    from omni.layers.linear import AscendRowParallelLinear
    # [关键] 2. 局部import：确保层获取最新的配置。
    # 如果依赖顶层导入，工作进程可能会使用带有错误TP大小的缓存类版本。

    # 指定die进行测试
    device = torch.device(f"npu:{device}")
    
    # 创建层
    layer = AscendRowParallelLinear(input_size, ...).to(device)
    
    # 创建输入
    x = torch.randn(input_size, dtype=dtype, device=device)
    
    # 运行，可能涉及die间通信
    out, _ = layer(x)
    
    # 验证（例如，与本地ground truth计算进行对比）
    expected = ... 
    assert torch.allclose(out, expected)
```

### 步骤 2：编写Pytest驱动程序
此函数在主进程中运行。它生成测试数据（如果需要）并将逻辑分发给进程池。

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_my_new_feature_distributed(distributed_worker_pool, dtype):
    """
    'distributed_worker_pool' fixture管理这些工作进程。
    """
    input_size = 64
    world_size = 2
    
    distributed_worker_pool(
        _logic_my_new_feature,  # 上面定义的函数
        input_size,             # 参数 1
        dtype                   # 参数 2
    )
```

## 深入探讨：常见陷阱

### 1. "局部导入"要求
1. 测试因大小不匹配而失败。你期望all_gather后的激活值（大小 60），但得到的是local shard（大小 30）。  
通过在测试payload函数`_logic_...` 内部导入对应的类，确保工作进程使用重新加载的模块和正确的配置。

### 2. 序列化错误
症状：`AttributeError: Can't pickle local object...`  
修复：将 `_logic_...` 移动到文件的全局作用域。

### 3. 随机性偏差 
测试逻辑函数的第一行调用 `torch.manual_seed(TEST_SEED)`。

## 调试

如果工作进程崩溃，主进程将收到一个包含来自特定失败rank的堆栈跟踪信息的RuntimeError。

示例错误：

```
RuntimeError: Rank 1 失败：
Traceback (most recent call last):
  File ...
    assert out.shape == expected.shape
AssertionError: ...
```

如果池无限期挂起，通常意味着一个集合操作（如`all_gather`）在一个rank上被调用，但在另一个rank上没有调用（例如，在 `if local_rank == 0:` 块）。
