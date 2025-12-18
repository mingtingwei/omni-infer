#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
dependencies_file=/workspace/Mooncake/dependencies.sh

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help "
	  echo "  -a, --arch 架构"
	  echo "  -g, --go_version 版本"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--arch)
            arch=$2
            shift 2
            ;;
        -g|--go_version)
            go_version=$2
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            show_help
            exit 1
            ;;
    esac
done

#注释GOVER
sed -i '/^GOVER=[0-9]\+\./s/^/# /' $dependencies_file

sed -i '/NC="\\033\[0m" # No Color/r /dev/stdin' $dependencies_file <<EOF
##TODO 环境参数
ENV=\$1
if [ "\$ENV" != "aarch64" ]; then
	ENV="x86_64"
else
	ENV="aarch64"
fi

##TODO 传入go_version 版本
GOVER=\$2
EOF


#清除多余的repo文件
#openEuler_file=/etc/yum.repos.d/openEuler.repo
#if [ -f $openEuler_file ];then
#  rm -rf $openEuler_file
#fi

# 把待插入的脚本块整体转义后，通过 here-document 传给 sed
sed -i '/^apt-get install -y \$SYSTEM_PACKAGES$/{
r /dev/stdin
}' "/workspace/Mooncake/dependencies.sh" <<'EOF'
rm -rf /etc/yum.repos.d/openEuler.repo
if [ "$ENV" == "x86_64" ]; then
    yum install -y $SYSTEM_PACKAGES
else
    yum install -y cmake git wget libibverbs-devel glog-devel gtest-devel jsoncpp-devel libunwind-devel numactl-devel python3-devel boost-devel openssl-devel grpc-devel protobuf-devel yaml-cpp-devel grpc-plugins libcurl-devel hiredis-devel pkgconfig grpc pybind11-devel patchelf
fi
EOF

## 调整为openeuler的安装方案，采用yum安装
#注释掉下面一行
#apt-get install -y $SYSTEM_PACKAGES
sed -i '/apt-get install -y \$SYSTEM_PACKAGES/s/^/# /' $dependencies_file

# 将apt-get 替换成yum
sed -i 's/apt-get/yum/g' $dependencies_file
sed -i 's/\byum update\b/yum -y update/g' $dependencies_file
#sed -i 's/^[[:space:]]*yum update[[:space:]]*$/# &/' $dependencies_file
## 删除下载go gz包
#wget -q --show-progress https://go.dev/dl/go$GOVER.linux-$ARCH.tar.gz
#check_success "Failed to download Go $GOVER"
sed -i '/wget -q --show-progress https:\/\/go.dev\/dl\/go$GOVER.linux-$ARCH.tar.gz/s/^/# /' $dependencies_file
sed -i '/check_success "Failed to download Go $GOVER"/s/^/# /' $dependencies_file

awk '
/^#/ {print; next}                          # 跳过已注释行
/^[[:space:]]*if \[[[:space:]]*-f[[:space:]]+"\${REPO_ROOT}\/\.gitmodules"[[:space:]]*\]; then[[:space:]]*$/ {
    in_block = 1
    depth    = 0
}
in_block {
    # 计算当前行对 if/fi 的深度贡献
    cnt_if  = gsub(/(^|[[:space:]])if[[:space:]]+\[/,  "&")
    cnt_fi  = gsub(/(^|[[:space:]])fi[[:space:]]*$/ , "&")
    depth  += cnt_if - cnt_fi

    # 给本行加注释（保留缩进）
    match($0, /^[[:space:]]*/)
    printf "%s# %s\n", substr($0, RSTART, RLENGTH), substr($0, RLENGTH+1)

    if (depth == 0) { in_block = 0 }         # 最外层 fi 已出现
    next
}
{ print }
'  $dependencies_file > $dependencies_file.tmp && mv $dependencies_file.tmp $dependencies_file

awk '
/^#/ {print; next}                          # 跳过已注释行
/^[[:space:]]*if \[[[:space:]]*-d[[:space:]]+"yalantinglibs"[[:space:]]*\]; then[[:space:]]*$/ {
    in_block = 1
    depth    = 0
}
in_block {
    cnt_if  = gsub(/(^|[[:space:]])if[[:space:]]+\[/,  "&")
    cnt_fi  = gsub(/(^|[[:space:]])fi[[:space:]]*$/ , "&")
    depth  += cnt_if - cnt_fi

    # 保留缩进并加注释
    match($0, /^[[:space:]]*/)
    printf "%s# %s\n", substr($0, RSTART, RLENGTH), substr($0, RLENGTH+1)

    if (depth == 0) { in_block = 0 }
    next
}
{ print }
' $dependencies_file > $dependencies_file.tmp && mv $dependencies_file.tmp $dependencies_file


sed -i '/^[[:space:]]*git clone \${GITHUB_PROXY}\/alibaba\/yalantinglibs\.git[[:space:]]*$/s/^[[:space:]]*/&# /' $dependencies_file
sed -i '/^[[:space:]]*git checkout 0\.5\.5[[:space:]]*$/s/^[[:space:]]*/&# /' $dependencies_file

cmake_list_file=Mooncake/mooncake-store/CMakeLists.txt
#注释 add_subdirectory(tests)
sed -i '/^[[:space:]]*add_subdirectory(tests)[[:space:]]*$/s/^[[:space:]]*/&#/' $cmake_list_file


master_client_file=Mooncake/mooncake-store/src/master_client.cpp
#c语言 注释
##include <source_location>
#auto location = std::source_location::current
#auto name = location.function_name();
#LOG(INFO) << "Connecting to master at " << master_addr << " from " << name;
sed -i '/^[[:space:]]*#include <source_location>/s/^/\/\//' $master_client_file
sed -i '/^\s*auto location = std::source_location::current/s/^\s*/&\/\//' $master_client_file
sed -i '/^\s*auto name = location\.function_name();/s/^\s*/&\/\//' $master_client_file
sed -i '/^\s*LOG(INFO) << "Connecting to master at " << master_addr << " from " << name;/s/^\s*/&\/\//' $master_client_file

#到install_go 中添加cd /workspace
sed -i -e '
/^install_go() {/,/^}/ {          # 只在这对花括号之间操作
  /^install_go() {$/ {            # 情况1：{ 与函数名同一行
    n                              # 移到下一行（函数体第一行）
    i\    cd /workspace
  }
  /^install_go()$/ {               # 情况2：{ 单独在下一行
    N                              # 把 { 那一行也读进来
    /\n{$/ {
      s/\n$/\ncd \/workspace\n/    # 在 { 后面插一行
    }
  }
}
' $dependencies_file

cd Mooncake

#export http_proxy=$proxy
#export https_proxy=$proxy

sh /workspace/Mooncake/dependencies.sh $arch $go_version -y yes

mkdir /workspace/Mooncake/build
cd /workspace/Mooncake/build

#默认开启ascend kdirect transport
cmake -DUSE_ASCEND_DIRECT=ON ..
make -j

cd .. # 这里一定要到Mooncake目录下才不会报错！
sh scripts/build_wheel.sh #构造whl包，默认构造完的路径在	-wheel/dist下

#直接安装wheel
pip install /workspace/Mooncake/mooncake-wheel/dist/mooncake*.whl


#设置pip url
#pip config set global.index-url "http://mirrors.tools.huawei.com/pypi/simple"
#pip config set global.trusted-host "mirrors.tools.huawei.com"
source ~/.bashrc

#容器内部pip
pip install numpy==1.24.0 absl-py decorator jinja2 attrs psutil cloudpickle ml-dtypes scipy==1.15.3 tornado dataflow sympy pyyaml cffi pathlib2 protobuf requests wheel typing_extensions ray==2.49.0 numba setuptools_scm pybase64 IPython pydantic orjson uvicorn uvloop setproctitle==1.3.6 torchao==0.9.0 compressed-tensors==0.9.3 fastapi aiohttp pyzmq Pillow einops sentencepiece msgspec partial_json_parser python-multipart dill xgrammar pybind11 cachetools==6.1.0 blake3==1.0.5 build
pip install vllm==v0.8.5 openai_harmony

cd /workspace/omniinfer
python -m build
cd /
pip install /workspace/omniinfer/dist/omni_i*.whl
cd /workspace/omniinfer/build
bash build_sglang_whl.sh
cd /
pip install /workspace/omniinfer/build/sglang_dist/sglang-*.whl
cd /workspace/omniinfer/omni/accelerators/sched/global_proxy/build/
bash build.sh

cd /workspace/omniinfer/omni/accelerators/sched/omni_proxy/build/
bash build.sh

cd /workspace/omniinfer/build/dist

if [ -f ./omni-proxy*.rpm ];then
yum install -y omni-proxy*.rpm
fi

if [ -f ./global-proxy*.rpm ];then
rpm -ivh --replacefiles global-proxy*.rpm;
fi

rm -rf Mooncake
rm -rf /tmp/*