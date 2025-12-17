# 概述

基于pytest框架构建模型端到端测试，主要进行模型的初始化和foward（prefill、decode）接口测试，测试模型脚本的基本功能、精度和性能。



**主要运行依赖项有：**

- 模型的config.json
- vllm_config（包括：通信域配置、extra_args、additional_config、模型相关的调度配置、投机推理配置等）
- 模型配置文件
- 简化的model_runner



**约定：**

- 在A3单机内完成所有模型测试用例
- 由于HCCL的限制，不能在同一个进程内多次初始化模型



# 如何运行

可以单独在```tests/unit_tests/models```目录下执行```pytest -v -s```执行所有的模型测试用例。

也可以跑单个用例：```pytest -v -s test_e2e_models.py::Test_e2e_models::test_model[DeepseekV3ForCausalLM-1-False]```，[]中的格式需要随着函数```test_model```参数列表的变化而做相应的修改。



# 文件说明

- base.py：该文件中存放一些用于测试模型的基础类实现，比如用于运行模型的简化model runner，命名为MockRunner
- registry.py：该文件中用于注册要测试的模型配置信息，比如模型轻量化后的config信息、输入的token id等
- test_e2e_models.py：模型端到端测试用例实现文件，扩展或者新增的测试用例实现放在这个文件中
- utils.py：改文件用于存放独立功能函数实现



# 添加新用例

## 添加新模型

在registry.py文件中添加，可参考当前已有的```DeepseekV3ForCausalLM```类实现。

新增的模型需要完成```_HfExamplesInfo```类信息的填充，包括对应模型的类名、模型的config信息、输入的token id等，将信息放入字典```__TRANSFORMERS_MODELS```中。



## 新增测试条件

对应代码实现请放到test_e2e_models.py文件中。

比如对每个模型要进行量化和非量化场景的测试，则可以在```test_model```的参数列表中增加量化的参数，使用pytest的```@pytest.mark.parametrize```方式生成多个测试用例。

