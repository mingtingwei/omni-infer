## 1. 前置操作

环境准备：
模型mock确保下载完成，如果不，进入accelerators目录下执行以下指令预设model相关mock信息

```shell
cd ../accelerations
bash setup_vllm_mock.sh
```

## 2. 快速执行

执行命令：
静态unitest用例，可直接执行

```shell
pytest test_scheduler.py -s -v
pytest test_sche_v1.py -s -v
```