#!/bin/bash

function set_env_from_arg_or_default() {
    local env_key="$1"
    local arg_key="$2"
    local default_value="$3"
    local data_source="by ENV"

    # 判断是否是路径（包含 /）
    if [[ "$default_value" == */* ]]; then
        echo "路径格式正确"
        if [[ "$default_value" == *.* ]]; then
            # 处理路径带文件的
            dir=$(dirname "$default_value")
            # 判断文件是否存在
            if [ ! -d "$dir" ]; then
                echo "目录不存在，准备创建目录：$dir"
                mkdir -p "$dir"
            else
                echo "目录已存在"
            fi
        else
            if [ ! -d "$default_value" ]; then
                echo "目录不存在，准备创建目录：$dir"
                mkdir -p "$default_value"
            else
                echo "目录已存在"
            fi
        fi
    else
        echo "不是有效路径，跳过"
    fi

    for arg in "$@"; do
        case "$arg" in
            ${arg_key}=*)
                export "$env_key"="${arg#*=}"
                data_source="by ARG $arg_key"
                ;;
        esac
    done

    if [ -z "${!env_key}" ] && [ -n "$default_value" ]; then
        export "$env_key"="$default_value"
        data_source="by DEFAULT"
    fi

    if [ -z "${!env_key}" ]; then
        unset "$env_key"
        echo "$env_key is UNSET"
    else
        echo "$env_key=${!env_key} ($data_source)"
    fi
}

function set_env() {
    local env_key="$1"
    local env_value="$2"
    local description="by BOOT Script"
    if [ -n "$3" ]; then
        local description="$description: $3"
    fi

    if [ -n "$env_value" ]; then
        export "$env_key"="$env_value"
        echo "$env_key=${!env_key} ($description)"
    else
        unset "$env_key"
        echo "$env_key is UNSET"
    fi
}
