#!/bin/bash
################################################################################
# 脚本名称: collect_logs.sh
# 功能描述: 服务器日志收集与解析自动化脚本
################################################################################
#
# 功能概述:
# ==========
# 1. 连接到配置的服务器列表
# 2. 收集 trace_output_directory 中的日志
# 3. 收集 nginx 错误日志
# 4. 将所有日志保存到本地目录
# 5. 调用 Python 解析脚本处理日志
# 6. 根据 先P后D 或 先D后P 特性生成 端到端流程图 并标识阶段平均耗时
#
# 配置参数说明:
# ==========
# 1. SERVER_LIST: 拉起服务的所有目标机器 IP
# 2. REMOTE_FOLDER: 目标机上 trace日志 存放目录，默认为 /tmp/trace_output_directory/
# 3. PROXY_FOLDER: nginx_error.log 所在目录，根据自己的服务日志目录填写
# 4. TARGET_FOLDER: 执行机上存放 trace日志 的目录
# 5. PRIVATE_KEY: 私钥 存放目录
# 6. CONNECTOR_TYPE: 先P后D——p2d / 先D后P——d2p
################################################################################

SERVER_LIST="
            10.11.123.1
            10.11.123.2
            10.11.123.3
            10.11.123.4
        "
REMOTE_FOLDER="/your/trace/output/directory/"
PROXY_FOLDER="/your/nginx_error/directory/"
TARGET_FOLDER="/your/trace/collect/save/directory/"
PRIVATE_KEY="/your/private/key/path"

if [ -d "$TARGET_FOLDER" ]; then
    echo "Error: $TARGET_FOLDER already exists."
    exit 1
fi

mkdir $TARGET_FOLDER

CONNECTOR_TYPE="d2p"

# collect trace logs
for IP in $(echo $SERVER_LIST); do
    echo "Collecting logs from $IP..."
    scp -i "$PRIVATE_KEY" -r "root@$IP:$REMOTE_FOLDER" "./logs_$IP"
    mv "./logs_$IP" $TARGET_FOLDER

    # copy logs in proxy
    if ssh -i "$PRIVATE_KEY" root@$IP "test -f '$PROXY_FOLDER/nginx_error.log'"; then
        echo "nginx_error.log found on $IP, copying..."
        scp -i "$PRIVATE_KEY" "root@$IP:$PROXY_FOLDER/nginx_error.log" "$TARGET_FOLDER/nginx_${IP}.log"
    fi
done

# parse trace logs
# current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"
# path of script parse_logs.py
PARSE_SCRIPT="${SCRIPT_DIR}/parse_logs.py"

python "$PARSE_SCRIPT" "$TARGET_FOLDER" "$CONNECTOR_TYPE"