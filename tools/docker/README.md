## 安装前准备
安装前需要提供以下本地包（aarch64环境下）：
0.6.0版本、0.7.0版本需要的包：
- [msgpack-c-6.1.0.tar.gz](https://github.com/msgpack/msgpack-c/releases/download/c-6.1.0/msgpack-c-6.1.0.tar.gz)
- [torchvision-0.21.0-cp311-cp311-linux_aarch64.whl](https://download.pytorch.org/whl/torchvision/)
- [torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl](https://download.pytorch.org/whl/cu126/torch/)
- Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run
- Ascend-cann-toolkit_8.3.T1_linux-aarch64.run

**注意：** 
1. aarch64环境下只需要提供对应版本的torch_npu，安装时自动下载对应版本的torch包，而x86_64必须自行提供对应版本的torch包；
2. torchvision包和torch包有依赖关系，请下载对应版本的包到本地，当前torchvision-0.21.0依赖torch-2.6.0。
3. 本地包统一放在/omniifer/tools/copy_data路径下面，需新建文件夹。

## 构建镜像
1. 构建镜像前，需要修改下面变量
```bash
ARCH="aarch64"
PROXY="http://username:passward@hostIP:port/"
HUGGING_FACE_PROXY="http://username:passward@hostIP:port/"
PIP_INDEX_URL="https://mirrors.huaweicloud.com/repository/pypi/simple"
PIP_TRUSTED_HOST="mirrors.huaweicloud.com"
MODEL_NAME="Qwen/Qwen2.5-0.5B"
BASE_IMAGE=test-infer-base:0.1
DEV_IMAGE=test-infer-dev:0.1
USER_IMAGE=test-infer-apiserver:0.1
OMNI_VERSION_NUM=master ## 按需修改
```
2. 按需修改上述内容且保证在/omniifer/tools/copy_data路径下有所需的本地包之后，执行命令
```shell
bash docker_build_run.sh
```