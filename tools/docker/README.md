# OmniInfer Docker 镜像构建指南

本文档介绍如何使用 Docker 构建 OmniInfer 推理服务镜像，支持 Ascend NPU 硬件加速。

## 目录

1. [安装前准备](#安装前准备)
2. [镜像分层](#镜像分层)
3. [镜像一键构建](#镜像一键构建)
   - [脚本参数说明](#docker_build_run脚本参数说明)
   - [命令执行](#命令执行)
4. [Dockerfile 详细说明](#dockerfile-详细说明)
5. [容器启动](#容器启动)

## 快速开始

### 全量构建（推荐首次使用）
```bash
bash docker_build_run.sh \
    --build-target all \
    --system-image openeuler:22.03 \
    --base-image my-infer-system:0.1 \
    --L1-image my-infer-meddle:0.1 \
    --L2-image my-infer-omniinfer:0.1 \
    --model-name "Qwen/Qwen2.5-0.5B"
```

### 增量构建（已有 L0 镜像）
```bash
bash docker_build_run.sh \
    --build-target both \
    --base-image my-infer-system:0.1 \
    --L1-image my-infer-meddle:0.1 \
    --L2-image my-infer-omniinfer:0.1 \
    --model-name "Qwen/Qwen2.5-0.5B"
```

---

## 安装前准备

安装前需要提供以下本地包（aarch64环境下）：

### 必需的基础包
- [msgpack-c-6.1.0.tar.gz](https://github.com/msgpack/msgpack-c/releases/download/c-6.1.0/msgpack-c-6.1.0.tar.gz) - 用于 L0 镜像构建

### PyTorch 相关包（0.6.0版本、0.7.0版本需要）
- [torchvision-0.21.0-cp311-cp311-linux_aarch64.whl](https://download.pytorch.org/whl/torchvision/) - 用于 L1 镜像构建
- [torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl](https://download.pytorch.org/whl/cu126/torch/) - 用于 L1 镜像构建

### CANN 包（用于 L1 镜像构建）

**整包安装模式（cann-install-mode=whole）**：
- Ascend-cann-toolkit_8.3.T1_linux-aarch64.run
- Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run
- Ascend-cann-nnal_8.3.T1_linux-aarch64.run（可选）

**分包安装模式（cann-install-mode=split）**：
- CANN-runtime-*.run
- CANN-opp-*.run
- CANN-toolkit-*.run
- CANN-compiler-*.run
- CANN-hccl-*.run
- CANN-aoe-*.run（可选）
- Ascend*-opp_kernel-*.run
- Ascend-cann-nnal-*.run（可选）

### 自定义算子包和其他代码
- 所需的自定义算子包代码及其构建脚本

**注意**： 

1. **架构差异**：aarch64 环境下只需要提供对应版本的 torch_npu，安装时会自动下载对应版本的 torch 包；而 x86_64 必须自行提供对应版本的 torch 包。
2. **版本依赖**：torchvision 包和 torch 包有依赖关系，请下载对应版本的包到本地。当前 torchvision-0.21.0 依赖 torch-2.6.0。
3. **文件路径**：whl 包、run 包等本地安装包统一放在 `/omniifer/tools/docker/copy_data` 路径下面。
4. **自定义算子**：自定义算子包和对应的编包脚本放在 `/omniifer/tools/docker/codes` 路径下，其他无法 git 需要自行下载的代码也统一放到该路径下。
5. **Python 版本**：如需自定义 Python 版本，确保 torch_npu 和 torchvision 等包的 Python 版本（如 cp311）与 `--python-version` 参数指定的版本一致。
6. **SGLang 框架**：如需使用 SGLang 框架（`--framework-type sglang`），需要额外准备 Mooncake 相关代码和 Go 语言环境（go1.24.1.linux-arm64.tar.gz）。
## 镜像分层

下面简要说明脚本中涉及的镜像分层（层次越高依赖越低层），便于理解各层构建的职责和包含的软件包。

- 基础系统镜像 (SYSTEM_IMAGE)
	- 说明：系统级基础镜像，通常由用户预先提供或基于官方发行版构建（如 openEuler）。
	- 作用：提供操作系统基础环境。

- L0 镜像（由 Dockerfile.system 生成，脚本中变量名为 BASE_IMAGE）
	- 说明：系统基础镜像，在 SYSTEM_IMAGE 基础上安装 Python 环境和 msgpack 等基础依赖。
	- 作用：为后续镜像提供统一的 Python 运行环境，支持自定义 Python 版本。

- L1 镜像（由 Dockerfile.base 生成，脚本中变量名为 L1_IMAGE）
	- 说明：开发/设备层镜像，在 L0 镜像基础上安装 CANN、torch_npu 和内核驱动相关组件。
	- 作用：为推理与硬件加速提供完整的运行时环境（例如 Ascend runtime、驱动、torch_npu 等），并打包为可复用的中间层镜像供 L2 依赖。

- L2 镜像（由 Dockerfile.omniinfer 或 Dockerfile.sglang 生成，脚本中变量名为 L2_IMAGE）
	- 说明：应用层镜像／apiserver 镜像，基于 L1 镜像构建，包含 omniinfer 服务、Python 依赖、模型服务与用户级工具。
	- 作用：打包并暴露模型推理服务（apiserver/omniinfer）、管理脚本、模型下载配置、自定义算子以及运行时需要的第三方 Python 包，直接用于运行容器并对外提供推理服务。
	- 框架支持：支持 vLLM（使用 Dockerfile.omniinfer）和 SGLang（使用 Dockerfile.sglang）两种推理框架。

- Roma 镜像（由 Dockerfile.roma 生成，脚本中变量名为 ROMA_IMAGE，可选）
	- 说明：Roma 环境专用镜像，基于 L2 镜像构建，添加 Roma 环境特定的用户权限配置。
	- 作用：为 Roma 平台环境提供符合安全规范的镜像，包含 ma-user 用户配置和权限管理。


## 镜像一键构建

该脚本目前支持执行脚本一键构建，具备以下特性：
- 支持灵活的镜像层级构建（L0/L1/L2 单独或组合构建）
- 支持多种推理框架（vLLM、SGLang）
- 支持 CANN 包的整包/分包安装
- 支持自定义 Python 版本
- 支持自定义算子包安装（需提供自定义算子包和对应的 build 脚本）
- 支持 Roma 环境专用镜像构建
- 支持多架构（aarch64、x86_64）

### docker_build_run脚本参数说明

`docker_build_run.sh` 中参数的各个字段说明如下：
| 字段                            | 含义                                                                                                                               |
| :-------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------
| `--arch <arch>`             | 目标构建平台架构（例如: aarch64 或 x86_64）。默认值: aarch64
| `--proxy <proxy>`           | 构建时使用的 HTTP 代理（用于在构建镜像过程中访问外部资源）。示例: http://user:pass@host:port/
| `--hugging-face-proxy <proxy>` | 运行时容器用于模型下载的 HTTP 代理（传递给启动容器的环境变量）。默认同 --proxy
| `--pip-index-url <url>`     | Docker 构建中使用的 pip 索引地址（用于安装 pip 包）。示例: https://mirrors.huaweicloud.com/repository/pypi/simple
| `--pip-trusted-host <host>` | pip 的 trusted host，用于允许在受信任的索引上安装包（避免证书或主机校验失败）。
| `--model-name <name>`       | 运行时要下载或启动的模型名字（例如: "Qwen/Qwen2.5-0.5B"）。此参数会被传递到容器运行时。
| `--python-version <version>` | 在构建 Dockerfile.system 时，指定要安装的 Python 版本（例如: 3.11.12）。默认: 3.11.12
| `--cann-install-mode <split\|whole>`| 在构建 Dockerfile.base 时，指定 CANN 包的安装方式：`whole`（使用完整安装包）或 `split`（分包安装）。默认: whole
| `--system-image <image>`    | 指定系统级基础镜像标签（作为 Dockerfile.system 的输入镜像）。默认: test-infer-base:0.1
| `--base-image <image>`      | 指定 L0 镜像标签（由 Dockerfile.system 生成，作为 Dockerfile.base 的输入镜像）。默认: test-infer-system:0.1
| `--L1-image <image>`        | L1 开发镜像（base/dev）构建完成后打的镜像 tag。默认: test-infer-meddle:0.1
| `--L2-image <image>`        | L2 用户/服务镜像（apiserver/omniinfer）构建完成后打的镜像 tag。默认: test-infer-omniinfer:0.1
| `--roma-image <image>`      | Roma 镜像构建完成后打的镜像 tag。默认: test-infer-ROMA:0.1
| `--branch <tag>`  | 要包含到镜像中的 omni 源码版本或分支（例如: master 或具体 tag）。默认: master
| `--framework-type <type>`   | 指定推理框架类型：`vllm` 或 `sglang`。决定使用 Dockerfile.omniinfer 还是 Dockerfile.sglang。默认: vllm
| `--custom-ops <ops>`        | 需要加入镜像的自定义算子包（逗号分隔的包名或构建脚本路径）。默认空，即不加入额外自定义算子。
| `--npu-platform <platform>` | 在构建自定义算子包时，指定当前机器硬件平台类型：910B 或 910C。默认: 910C
| `--start-server <True\|False>` | 指定是否启动容器执行 start_server.sh，默认为 True
| `--build-for-roma <True\|False>` | 指定是否构建 Roma 环境镜像（基于 L2 镜像）。默认为 false
| `--build-target <L0\|L1\|L2\|both\|all\|skip>` | 选择要构建的目标：<br>- `L0`：只构建 Dockerfile.system<br>- `L1`：只构建 Dockerfile.base<br>- `L2`：只构建 Dockerfile.omniinfer 或 Dockerfile.sglang<br>- `both`：先构建 L1 再构建 L2（默认，假设 L0 已存在）<br>- `all`：依次构建 L0、L1、L2<br>- `skip`：跳过所有构建

### 命令执行

下面给出若干常见的执行示例——把示例中的占位符替换为实际值：镜像tag、模型名、自定义算子包以及构建目标等。

**示例 1 — 全量构建所有层级（build-target=all）**：

```bash
bash docker_build_run.sh \
	--build-target all \
	--arch aarch64 \
	--system-image openeuler:22.03 \
	--base-image new-infer-system:0.1 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image new-infer-omniinfer:0.1 \
	--python-version 3.11.12 \
	--model-name "Qwen/Qwen2.5-0.5B" \
	--branch master
```
在这个示例中，将会依次构建 L0、L1 和 L2 镜像：
1. 执行 `Dockerfile.system`：输入 `openeuler:22.03`，输出 L0 镜像 `new-infer-system:0.1`，安装 Python 3.11.12
2. 执行 `Dockerfile.base`：输入 `new-infer-system:0.1`，输出 L1 镜像 `new-infer-meddle:0.1`，安装 CANN 和 PyTorch
3. 执行 `Dockerfile.omniinfer`：输入 `new-infer-meddle:0.1`，输出 L2 镜像 `new-infer-omniinfer:0.1`，安装 omniinfer 服务

**示例 2 — 默认构建（build-target=both，假设 L0 已存在）**：

```bash
bash docker_build_run.sh \
	--arch aarch64 \
	--base-image new-infer-system:0.1 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image new-infer-omniinfer:0.1 \
	--model-name "Qwen/Qwen2.5-0.5B" \
	--branch master
```
在这个示例中，将会串行构建 L1 和 L2 镜像，跳过 L0 镜像构建（假设 L0 镜像 `new-infer-system:0.1` 已存在）。

**示例 3 — 仅构建 L0（Python 基础镜像）**：

```bash
bash docker_build_run.sh \
	--build-target L0 \
	--system-image openeuler:22.03 \
	--base-image new-infer-system:0.1 \
	--python-version 3.11.12 \
	--pip-index-url "https://mirrors.huaweicloud.com/repository/pypi/simple" \
	--pip-trusted-host "mirrors.huaweicloud.com"
```
在这个示例中，只会构建 L0 镜像，安装指定版本的 Python 环境。

**示例 4 — 仅构建 L1（CANN + PyTorch），并使用 split 模式安装 CANN**：

```bash
bash docker_build_run.sh \
	--build-target L1 \
	--cann-install-mode split \
	--base-image new-infer-system:0.1 \
	--L1-image new-infer-meddle:0.1 \
	--pip-index-url "https://mirrors.huaweicloud.com/repository/pypi/simple" \
	--pip-trusted-host "mirrors.huaweicloud.com"
```
在这个示例中，会跳过 L0 和 L2 的构建，只执行 `Dockerfile.base` 输出 L1 镜像（`new-infer-meddle:0.1`）。注意，必须确保 `--base-image` 指定的 L0 镜像已经存在。

**示例 5 — 仅构建 L2（应用层），加入自定义算子包**：

```bash
bash docker_build_run.sh \
	--build-target L2 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image test-infer-omniinfer:latest \
	--branch release_0.7.0 \
	--custom-ops build_cann_recipes_ops,build_omni_ops \
	--start-server False
```
在这个示例中，会跳过 L0 和 L1 的构建，只执行 `Dockerfile.omniinfer` 输出 L2 镜像（`test-infer-omniinfer:latest`）。注意，这里必须提供对应的 `--L1-image` 入参，否则会使用默认的 `test-infer-meddle:0.1`。镜像构建完之后不再启动容器执行 start_server.sh。

**示例 6 — 使用 SGLang 框架构建 L2 镜像**：

```bash
bash docker_build_run.sh \
	--build-target L2 \
	--framework-type sglang \
	--L1-image new-infer-meddle:0.1 \
	--L2-image test-infer-sglang:0.1 \
	--custom-ops build_cann_recipes_ops \
	--branch master
```
在这个示例中，将使用 `Dockerfile.sglang` 构建 L2 镜像，而不是默认的 `Dockerfile.omniinfer`。

**示例 7 — 构建 Roma 环境镜像**：

```bash
bash docker_build_run.sh \
	--build-target both \
	--L1-image new-infer-meddle:0.1 \
	--L2-image new-infer-omniinfer:0.1 \
	--roma-image new-infer-roma:0.1 \
	--build-for-roma True \
	--branch master
```
在这个示例中，将先构建 L1 和 L2 镜像，然后基于 L2 镜像构建 Roma 环境专用镜像，添加 ma-user 用户和相关权限配置。

**关于 `--custom-ops` 的使用说明**：

自定义算子的传参格式为 `ops1,ops2,...`，其中 ops1 指的是所要加入的自定义算子，命名需要对应 `/omniifer/tools/docker/codes` 路径下的 sh 脚本文件名（不含 .sh 后缀）。Dockerfile 会自动执行传入脚本名对应的自定义算子构建脚本。

例如，如果要安装 `build_cann_recipes_ops` 和 `build_omni_ops` 两个自定义算子，需要确保以下文件存在：
- `/omniifer/tools/docker/codes/build_cann_recipes_ops.sh`
- `/omniifer/tools/docker/codes/build_omni_ops.sh`

## Dockerfile 详细说明

### Dockerfile.system（L0 镜像）
- **输入**：SYSTEM_IMAGE（系统基础镜像，如 openEuler）
- **输出**：BASE_IMAGE（L0 镜像）
- **主要功能**：
  - 编译安装指定版本的 Python
  - 安装 msgpack-c 库
  - 配置 Python 环境变量和 pip
  - 为后续镜像提供统一的 Python 基础环境

### Dockerfile.base（L1 镜像）
- **输入**：BASE_IMAGE（L0 镜像）
- **输出**：L1_IMAGE（L1 镜像）
- **主要功能**：
  - 安装 PyTorch 和 torch_npu（根据架构自动选择安装方式）
  - 安装 CANN 工具包（支持整包和分包两种安装模式）
  - 安装 NNAL（可选）
  - 配置 Ascend 运行时环境变量
  - 提供完整的 NPU 推理运行时环境

### Dockerfile.omniinfer（L2 镜像 - vLLM 框架）
- **输入**：L1_IMAGE（L1 镜像）
- **输出**：L2_IMAGE（L2 镜像）
- **主要功能**：
  - 构建 omniinfer 和 vLLM whl 包
  - 安装 omniinfer 服务和依赖包
  - 安装自定义算子包（如果指定）
  - 安装 omni-proxy 和 global-proxy RPM 包
  - 配置 openai 兼容的 API 服务
  - 设置启动脚本 start_server.sh
- **多阶段构建**：
  - whl_builder：构建 whl 包
  - develop_image：开发环境（包含开发工具）
  - user_image：用户镜像（仅包含运行时依赖）
  - omininfer_openai：最终镜像（添加 API 服务）

### Dockerfile.sglang（L2 镜像 - SGLang 框架）
- **输入**：L1_IMAGE（L1 镜像）
- **输出**：L2_IMAGE（L2 镜像）
- **主要功能**：
  - 安装 SGLang 框架
  - 构建和安装 Mooncake（分布式 KV 缓存）
  - 安装 Go 语言环境
  - 安装自定义算子包（如果指定）
  - 配置 SGLang 推理服务环境

### Dockerfile.roma（Roma 镜像）
- **输入**：L2_IMAGE（L2 镜像）
- **输出**：ROMA_IMAGE（Roma 镜像）
- **主要功能**：
  - 创建 ma-user 用户（UID 1000）和 ma-group 组（GID 100）
  - 配置用户权限和 sudo 权限
  - 设置目录权限以符合 Roma 平台安全规范
  - 复制 root 的环境配置到 ma-user
  - 切换到 ma-user 用户运行

## 容器启动

### 启动说明
当前的镜像制作默认将 ENTRYPOINT 设置为 `start_server.sh`（对于 L2 omniinfer 镜像），因此：
- 如果需要进入容器交互式操作，起容器时需要加上 `--entrypoint=bash` 指令
- ENTRYPOINT 也支持自行覆盖

### 启动示例

**使用默认 ENTRYPOINT 启动推理服务**：
```bash
docker run --rm -it --shm-size=500g \
    --net=host --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -e PORT=8301 \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e HTTP_PROXY="http://proxy:port" \
    -e MODEL_NAME="Qwen/Qwen2.5-0.5B" \
    new-infer-omniinfer:0.1 \
    --model "Qwen/Qwen2.5-0.5B"
```

**进入容器进行调试**：
```bash
docker run --rm -it --shm-size=500g \
    --net=host --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    new-infer-omniinfer:0.1
```

**启动 Roma 镜像**：
```bash
docker run --rm -it --shm-size=500g \
    --net=host --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -u ma-user \
    new-infer-roma:0.1
```

### 环境变量说明
| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| PORT | API 服务监听端口 | 8301 |
| ASCEND_RT_VISIBLE_DEVICES | 可见的 NPU 设备 ID | 0 |
| HTTP_PROXY | 模型下载代理 | - |
| MODEL_NAME | 要加载的模型名称 | - |

## 常见问题与故障排除

### 1. 镜像构建失败

**问题**：构建过程中找不到依赖包
```
ERROR: Could not find a version that satisfies the requirement torch_npu
```

**解决方案**：
- 检查 `/omniifer/tools/docker/copy_data` 目录下是否包含所需的 whl 包
- 确保包的命名格式正确（例如：`torch_npu-2.6.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl`）
- 确保 Python 版本与 whl 包的版本一致（cp311 对应 Python 3.11）

---

**问题**：CANN 包安装失败
```
ERROR: CANN toolkit installation failed
```

**解决方案**：
- 检查 CANN 包的完整性和版本匹配
- 如果使用整包模式（whole），确保存在 `Ascend-cann-toolkit_*.run` 和 `Atlas-*-cann-kernels*.run`
- 如果使用分包模式（split），确保所有必需的 CANN 组件包都已准备
- 查看构建日志中的详细错误信息

---

**问题**：自定义算子构建失败
```
Error: no matching script found for custom op: build_xxx_ops
```

**解决方案**：
- 检查 `/omniifer/tools/docker/codes` 目录下是否存在对应的 `.sh` 脚本
- 确保脚本文件名与 `--custom-ops` 参数传入的名称一致（不含 .sh 后缀）
- 检查脚本是否有执行权限

### 2. 容器启动失败

**问题**：NPU 设备不可用
```
ERROR: NPU device not found
```

**解决方案**：
- 确保主机上已正确安装 Ascend 驱动
- 检查设备文件是否存在：`ls -l /dev/davinci*`
- 确保容器启动时添加了必要的设备映射参数：
  ```bash
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm
  ```
- 使用 `npu-smi info` 命令检查 NPU 设备状态

---

**问题**：权限不足
```
ERROR: Permission denied
```

**解决方案**：
- 确保容器以 root 用户或正确的用户启动
- 对于 Roma 镜像，使用 `-u ma-user` 参数
- 检查文件和目录权限设置


### 4. 构建目标选择

| 场景 | 推荐的 build-target | 说明 |
| :--- | :--- | :--- |
| 首次构建，没有任何镜像 | `all` | 依次构建 L0、L1、L2 |
| 已有 L0 镜像，需要更新 L1/L2 | `both` | 只构建 L1 和 L2 |
| 只需要更新 Python 版本 | `L0` | 只构建 L0 镜像 |
| 只需要更新 CANN 版本 | `L1` | 只构建 L1 镜像 |
| 只需要更新应用代码 | `L2` | 只构建 L2 镜像 |
| 需要切换推理框架 | `L2` | 使用 `--framework-type` 指定框架 |


## 版本兼容性

| 组件 | 推荐版本 | 说明 |
| :--- | :--- | :--- |
| Python | 3.11.12 | 支持 3.10+ |
| CANN | 8.3.T1 | 支持 8.0+ |
| PyTorch | 2.6.0 | 需与 torch_npu 版本匹配 |
| torch_npu | 2.6.0.post2 | 需与 PyTorch 版本匹配 |
| torchvision | 0.21.0 | 需与 PyTorch 版本匹配 |
| openEuler | 22.03 | 支持其他 Linux 发行版 |
