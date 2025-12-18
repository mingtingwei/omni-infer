#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

BASE_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

# 代码下载需要网络代理
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}
# The incoming value is a branch name (e.g. 'release_v0.6.0' or 'master').
# Use it directly if provided, otherwise default to 'master'.
branch="${BRANCH:-master}"

# 新增：is_branch_after_050作为传入参数，默认为True
is_branch_after_050="${IS_BRANCH_AFTER_050:-True}"


# 需要预先准备 omniinfer 代码，可根据CI流程优化，在容器中下载会导致构建时间过长
git config --global http.sslVerify false
# git clone -b "${branch}" https://gitee.com/omniai/omniinfer.git
# cd omniinfer/infer_engines
# git clone https://github.com/vllm-project/vllm.git

# Check if omniinfer exists in codes directory, copy if exists, otherwise clone
if [ -d "${BASE_DIR}/codes/omniinfer" ]; then
    echo "Directory ${BASE_DIR}/codes/omniinfer already exists. Copying to current path..."
    cp -r "${BASE_DIR}/codes/omniinfer" "${BASE_DIR}/"
    cd ${BASE_DIR}/omniinfer
    git checkout "${branch}" || echo "Warning: failed to checkout branch ${branch} in omniinfer."
else
    echo "Cloning omniinfer repo (branch: ${branch})..."
    if ! git clone -b "${branch}" https://gitee.com/omniai/omniinfer.git; then
        echo "ERROR: Failed to clone omniinfer from remote repository."
        echo "Please manually download omniinfer to ./codes/omniinfer and run this script again."
        exit 1
    fi
fi

cd ${BASE_DIR}/omniinfer/infer_engines

if [ -d "${BASE_DIR}/omniinfer/infer_engines/vllm" ]; then
    echo "vllm already exists in infer_engines. Skipping clone."
else
    # Check if vllm exists in codes directory, copy if exists, otherwise clone
    if [ -d "${BASE_DIR}/codes/vllm" ]; then
        echo "Directory ${BASE_DIR}/codes/vllm already exists. Copying to infer_engines..."
        cp -r "${BASE_DIR}/codes/vllm" "${BASE_DIR}/omniinfer/infer_engines/"
    else
        echo "Cloning vllm repo from remote..."
        if ! git clone https://github.com/vllm-project/vllm.git; then
            echo "ERROR: Failed to clone vllm from remote repository."
            echo "Please manually download vllm to ./codes/vllm and run this script again."
            exit 1
        fi
    fi
fi

# 构建 whl 包
cd ${BASE_DIR}/omniinfer
chmod +x build/build.sh
chmod +x infer_engines/bash_install_code.sh

# 根据 is_branch_after_050 参数的值决定构建方式
if [ "${is_branch_after_050}" = "True" ] || [ "${is_branch_after_050}" = "true" ]; then
    # For 'master' branch or versions >= 0.5.0 we use the cached build path
    bash -xe build/build.sh --cache 1
    cd ${BASE_DIR}/omniinfer/tools/quant/python
    python setup.py bdist_wheel
else
    bash -xe build/build.sh
fi

pip cache purge