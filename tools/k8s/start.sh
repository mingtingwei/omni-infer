#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=${CUR_DIR}}"
: "${CONFIG_DIR:=${ROOT_DIR}/config}"
: "${ROLE_DIR:=${ROOT_DIR}/role}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/env/env_tools.sh
source "${TOOL_DIR}"/grace/cleanup.sh
source "${TOOL_DIR}"/health/npu_checker.sh

# set SIGTERM trap
trap 'stop_all_processes_and_check_npus' SIGTERM

echo_with_time "------------> begin the process on A3 Omni 0.4.2"

# set entrypoint env
echo_with_time "------------> set entrypoint env"
unset RANK_TABLE_FILE
export GLOBAL_RANK_TABLE_FILE_PATH_KEY="GLOBAL_RANK_TABLE_FILE_PATH"
export TRANS_GLOBAL_RANK_TABLE_FILE_PATH="${CONFIG_DIR}"/ranktable/global_rank_table.json
echo_with_time "------------> set entrypoint env finished"

# 适配云道场景获取SERVER_GROUP_ID
if [ ${VC_WORKER_HOSTS} ]; then
    export GLOBAL_RANK_TABLE_FILE_PATH="${TRANS_GLOBAL_RANK_TABLE_FILE_PATH}"
    if [ ! -e "/usr/local/Ascend/latest" ]; then
        mkdir -p /usr/local/Ascend/latest
        ln -sf /usr/local/Ascend/ascend-toolkit/latest/* /usr/local/Ascend/latest
        echo "Link created successfully."
    else
        echo "Link already exists or target missing"
    fi

    if [[ -e "/usr/local/Ascend/ascend-toolkit" ]]; then
        python ${CODE_PATH}/tools/scripts/process_nz_config.py /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json
    else
        python ${CODE_PATH}/tools/scripts/process_nz_config.py /usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json
    fi

    if [[ ${ASCEND_PLATFORM} == "A2" ]]; then
        export device_list="0,1,2,3,4,5,6,7"
        export device_collection="01234567"
        export local_device_size=8
    else
        export device_list="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
        export device_collection="0123456789101112131415"
        export local_device_size=16
    fi

    export RANKTABLE_SAVE_PATH="/tmp/ranktable_save_path"
    export PREFILL_RANKTABLE_SAVE_PATH="${PREFILL_RANKTABLE_SAVE_PATH}"
    export DECODE_RANKTABLE_SAVE_PATH="${DECODE_RANKTABLE_SAVE_PATH}"
    export RANK_TABLE_PATH="${RANKTABLE_SAVE_PATH}/test/global"
    export RANK_TABLE_PATH_P="${RANK_TABLE_PATH}/p"
    export RANK_TABLE_PATH_D="${RANK_TABLE_PATH}/d"
    export GLOBAL_RANK_TABLE_FILE_PATH_KEY="GLOBAL_RANK_TABLE_FILE_PATH"
    export TRANS_GLOBAL_RANK_TABLE_FILE_PATH="${RANK_TABLE_PATH}/global_ranktable_merge.json"
    export GLOBAL_RANK_TABLE_FILE_PATH="${TRANS_GLOBAL_RANK_TABLE_FILE_PATH}"

    # XXXXXXXXXXXXXXXXXXXXXXXX 调试用 XXXXXXXXXXXXXXXXXXXXXXXXXXX
    export ASCEND_LAUNCH_BLOCKING="0"
    export LCCL_DETERMINISTIC="0"
    export LCCL_PARALLEL="0"
    export USING_LCCL_COM="0"

    # 提前创建所有必要目录
    mkdir -p \
      "$RANK_TABLE_PATH" "$RANK_TABLE_PATH_P" "$RANK_TABLE_PATH_D"\
      "$PREFILL_RANKTABLE_SAVE_PATH" \
      "$DECODE_RANKTABLE_SAVE_PATH"
      # "$PROMETHEUS_MULTIPROC_DIR"
    # parse ip
    export MA_CURRENT_IP="${MA_CURRENT_IP}"
    export NUM_PER_PREFILL_POD="${NUM_PER_PREFILL_POD}"
    export NUM_PER_DECODE_POD="${NUM_PER_DECODE_POD}"
    export VC_WORKER_HOSTS="${VC_WORKER_HOSTS}"
    export NODE_IPS=$(python3 "${TOOL_DIR}"/parse_domain.py)
    echo "-----------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
    echo "XXXXXXXXXXXXXXXXXXXX NODE_IPS is "$NODE_IPS
    echo "-----------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
    echo_with_time "------------> current_ip:"$MA_CURRENT_IP
    echo_with_time "------------> ip_list:"$NODE_IPS
    echo_with_time "------------> NUM_PER_PREFILL_POD:"$NUM_PER_PREFILL_POD
    echo_with_time "------------> NUM_PER_DECODE_POD:"$NUM_PER_DECODE_POD
    echo_with_time "------------> PREFILL_POD_NUM:"$PREFILL_POD_NUM
    echo_with_time "------------> DECODE_POD_NUM:"$DECODE_POD_NUM
    # 网络相关环境变量
    VPC_PREFIX=$(echo "$MA_CURRENT_IP" | cut -d'.' -f1-2)
    POD_INET_IP=$(hostname -I | tr ' ' '\n' | grep -o "^$VPC_PREFIX\.[0-9]\+\.[0-9]\+" | head -n 1)
    export SOCKET_IFNAME=$(ifconfig | grep -B 1 "$POD_INET_IP" | head -n 1 | awk '{print $1}' | sed 's/://')
    export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
    export TP_SOCKET_IFNAME=$SOCKET_IFNAME

    IFS=',' read -r -a arr <<< "$NODE_IPS"
    for i in "${!arr[@]}";do
        echo_with_time "------------> ip in array is "${arr[$i]}
        if [[ "${arr[$i]}" == "$MA_CURRENT_IP" ]]; then
            echo_with_time "------------> position is "$i
            PREFILL_TOTAL_NUM=$(($NUM_PER_PREFILL_POD * ${PREFILL_POD_NUM}))
            # 处理prefill节点的情况
            if [[ $i -lt  ${PREFILL_TOTAL_NUM} ]]; then
                export SERVER_GROUP_ID=$(($i / $NUM_PER_PREFILL_POD))
                export PREFILL_RANK=$i
                export PREFILL_ADDITIONAL_CONFIG=${PREFILL_ADDITIONAL_CONFIG}
                if [ $i -eq $((${SERVER_GROUP_ID} * ${NUM_PER_PREFILL_POD})) ]; then
                    export role_head_ip="${arr[$i]}"
                    echo "role_head_ip is ${role_head_ip}"
                fi
            # 处理decode节点的情况
            else
                DECODE_NUM_RNAK=$(($i - ${PREFILL_TOTAL_NUM}))
                export SERVER_GROUP_ID=$(($((${DECODE_NUM_RNAK} / $NUM_PER_DECODE_POD)) + ${PREFILL_POD_NUM}))
                export DECODE_RANK=${DECODE_NUM_RNAK}
                export DECODE_ADDITIONAL_CONFIG=${DECODE_ADDITIONAL_CONFIG}
                if [ ${DECODE_RANK} -eq $(($((${SERVER_GROUP_ID} - ${PREFILL_POD_NUM})) * ${NUM_PER_DECODE_POD})) ]; then
                    export role_head_ip="${arr[$i]}"
                    echo "role_head_ip is ${role_head_ip}"
                fi
            fi
            break
        fi
    done
    # set base env
    source "$CONFIG_DIR"/env/set_base_env.sh "$@"
fi


# 云道场景没有安装kubeinfer，无法先生成全局ranktable
if [ -z "${VC_WORKER_HOSTS}" ]; then
    # check ranktable
    echo_with_time "------------> start to check global rank table"
    python3 "${TOOL_DIR}"/ranktable/rank_table_checker.py &
    RANK_TABLE_CHECKER_PID=$!
    wait "${RANK_TABLE_CHECKER_PID}"
    echo "global rank table is as below:"
    cat "${!GLOBAL_RANK_TABLE_FILE_PATH_KEY}"
    echo_with_time "------------> check global rank table finished"

    # gen global ranktable
    echo_with_time "------------> start to gen global rank table"
    python3 "${TOOL_DIR}"/ranktable/global_rank_table_generator.py
    export GLOBAL_RANK_TABLE_FILE_PATH="${TRANS_GLOBAL_RANK_TABLE_FILE_PATH}"
    export SERVER_GROUP_ID=$((SERVER_GROUP_ID - 1))
    echo_with_time "------------> gen global rank table finished"
    # set base env
    source "$CONFIG_DIR"/env/set_base_env.sh "$@"
    # prepare for pre-stop
    {
        echo "export DECODE_SERVERS=${DECODE_SERVERS}"
        echo "export PREFILL_SERVERS=${PREFILL_SERVERS}"
        echo "export PRE_STOP_LOG_PATH=${PRE_STOP_LOG_PATH}"
    } >> "$CONFIG_DIR"/env/set_pre_stop_env.sh
fi




# prepare for log
echo_with_time "------------> prepare log path"
mkdir -p "${BASE_LOG_DIR}"
sed -i "s|{{PROBE_LOG_PATH}}|${PROBE_LOG_PATH}|g" "${ROOT_DIR}"/health.sh
echo_with_time "------------> prepare log path finished"


echo_with_time "------------> current group: ${SERVER_GROUP_ID}"
# 适配云道场景，proxy和prefill共用一个节点
echo "----------------------------------------------------------------"
echo "VC_WORKER_HOSTS is "$VC_WORKER_HOSTS
echo "----------------------------------------------------------------"
echo "----------------------------------------------------------------"
echo "SERVER_GROUP_ID is "$SERVER_GROUP_ID
echo "----------------------------------------------------------------"
if [ ${VC_WORKER_HOSTS} ] && [ "${SERVER_GROUP_ID}" -eq 0 ] ; then
    echo_with_time "XXXXXXXXXXXXXXXX PROXY XXXXXXXXXXXXXXXXX"$@
    echo_with_time "------------> yundao sophere: current role: proxy and prefill"
    if [ "${PROXY_BACKEND}" = "global-proxy" ]; then
        bash "${ROLE_DIR}"/start_global_proxy.sh "$@"
    else
        bash "${ROLE_DIR}"/start_route_server.sh "$@"
    fi
    bash "${ROLE_DIR}"/start_prefill.sh "$@"
elif [ "${SERVER_GROUP_ID}" -eq -1 ]; then
    echo_with_time "------------> current role: proxy"
    if [ "${PROXY_BACKEND}" = "global-proxy" ]; then
        bash "${ROLE_DIR}"/start_global_proxy.sh "$@"
    else
        bash "${ROLE_DIR}"/start_route_server.sh "$@"
    fi
elif [ "${SERVER_GROUP_ID}" -ge 0 ] && [ "${SERVER_GROUP_ID}" -lt "${PREFILL_POD_NUM}" ]; then
    echo_with_time "------------> current role: prefill"
    echo_with_time "XXXXXXXXXXXXXXXX PREFILL XXXXXXXXXXXXXXXXX"$@
    bash "${ROLE_DIR}"/start_prefill.sh "$@"
elif [ "${SERVER_GROUP_ID}" -ge "${PREFILL_POD_NUM}" ] && [ "${SERVER_GROUP_ID}" -lt "$((PREFILL_POD_NUM + DECODE_POD_NUM))" ]; then
    echo_with_time "------------> current role: decode"
    echo_with_time "XXXXXXXXXXXXXXXX DECODE XXXXXXXXXXXXXXXXX"$@
    bash "${ROLE_DIR}"/start_decode.sh "$@"
fi

echo "hold process..."
loop_count=0
while true; do
    ((loop_count++))
    if [ ${loop_count} -ge "${MAIN_THREAD_LOOP_TASK_CYCLE}" ]; then
        python3 "${TOOL_DIR}"/ray/clean_ray_logs.py
        loop_count=0
    fi
    sleep "${MAIN_THREAD_SLEEP_INTERVAL}"
done
