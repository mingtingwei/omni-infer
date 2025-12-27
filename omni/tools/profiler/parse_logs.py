# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import csv
import re
import os
from collections import defaultdict
import sys
import pandas as pd
import openpyxl
import traceback

FILL_VALUE = "-"

def parse_trace_logs(root_dir):
    pattern = (
        r"<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([a-z0-9-]+)(?:; Role:(\S+))?"
    )
    data_by_request = defaultdict(dict)
    request_role = defaultdict(dict)
    action_timestamps = {}
    engine_step_lines = []
    decode_engine_step_lines = []
    engine_core_info = {}

    time_analysis_path = os.path.join(root_dir, "time_analysis.xlsx")
    engine_step_path = os.path.join(root_dir, "engine_step.xlsx")
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                _get_step_line(
                    pattern,
                    data_by_request,
                    request_role,
                    action_timestamps,
                    engine_step_lines,
                    decode_engine_step_lines,
                    engine_core_info,
                    dirpath,
                    filename,
                )

        # process time analysis
        if data_by_request:
            df_final = _get_final_df(data_by_request, request_role)

            with pd.ExcelWriter(time_analysis_path, engine="openpyxl") as writer:
                df_final.to_excel(writer, sheet_name="time_analysis", index=False)
                summary_data = {
                    "RequestID": list(data_by_request.keys()),
                    "ActionCount": [
                        len(actions) for actions in data_by_request.values()
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

            print(
                f"Successfully parsed time analysis files."
            )

        else:
            print("No valid action record found in any log files.")

        # Process engine_step_lines
        engine_step_headers = _get_engine_step_headers()
        with pd.ExcelWriter(engine_step_path, engine="openpyxl") as writer:
            # engine_step_sheet
            _engine_step_sheet(
                engine_step_lines, engine_step_path, writer, engine_step_headers
            )

            _decode_engine_step_sheet(
                decode_engine_step_lines, engine_step_path, writer, engine_step_headers
            )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.print_exc())


def _get_engine_step_headers():
    return [
        "node",
        "engine_step start",
        "engine_step end",
        "execute time(ms)",
        "running_reqs_num_after_step",
        "total_tokens",
        "waiting_reqs_num_after_step",
        "reqs_ids",
        "bs_tokens",
        "execute_model_start_time",
        "execute_model_end_time",
        "execute_model_cost_time(ms)",
        "kv_cache_usage",
        "kv_blocks_num",
        "start_free_block_num",
        "end_free_block_num",
        "cost_blocks_num",
        "engine_core_str",
    ]


def _get_final_df(data_by_request, request_role):
    origin_action_map = _get_action_map()
    action_map = dict(map(lambda k, v: (k, v[0]), origin_action_map.keys(), origin_action_map.values()))
    fieldnames = ["RequestID", "P_NODE", "D_NODE"] + list(action_map.keys())
    data = []
    for request_id, actions in data_by_request.items():
        decode = request_role[request_id].get("decode")
        prefill = request_role[request_id].get("prefill")
        if decode is None or prefill is None:
            print(
                f'request_id: {request_role[request_id].get("request_id")} decode or prefill is None'
            )
            continue
        row = {"RequestID": request_id, "P_NODE": prefill, "D_NODE": decode}
        # Add timestamps for each action, FILL_VALUE for missing actions
        for action in action_map.keys():
            row[action] = actions.get(action, FILL_VALUE)
        data.append(row)

    df = pd.DataFrame(data, columns=fieldnames)
    # chinese_row
    chinese_row = {"RequestID": "", "P_NODE": "", "D_NODE": ""}
    chinese_row.update(action_map)
    df_cn = pd.DataFrame([chinese_row], columns=fieldnames)
    df_final = pd.concat([df.iloc[:0], df_cn, df.iloc[0:]], ignore_index=True)
    return df_final


def _get_step_line(
    pattern,
    data_by_request,
    request_role,
    action_timestamps,
    engine_step_lines,
    decode_engine_step_lines,
    engine_core_info,
    dirpath,
    filename,
):
    if filename.endswith(".log"):
        log_file_path = os.path.join(dirpath, filename)
        print(f"Processing log file: {log_file_path}")
        try:
            with open(log_file_path, "r", encoding="latin1") as file:
                for line in file:
                    # main model info
                    if "profile_mainmodel:" in line:
                        _get_main_model_info(engine_core_info, line)
                        continue
                    # mtp model info
                    if "profile_mtpmodel:" in line:
                        _get_mtp_model_info(engine_core_info, line)
                        continue
                    # for engine step
                    if "profile: " in line:
                        st_idx = line.find("profile:") + len("profile: ")
                        line = line[st_idx:]
                        # if "prefill" in line:
                        if not "[]" in line:
                            engine_step_lines.append(line)
                        else:
                            line = _set_decode_info(
                                decode_engine_step_lines, engine_core_info, line
                            )
                        continue
                    # for time analysis
                    if "<<<Action" in line:
                        st_idx = line.find("<<<Action")
                        line = line[st_idx:]  # skip prefix if any
                        match = re.match(pattern, line.strip())
                        if match:
                            action, timestamp, request_id, role = match.groups()
                            role, ip = role.split("_")
                            action = action.strip()
                            timestamp = float(timestamp)
                            # min value
                            if (
                                action not in data_by_request[request_id]
                                or timestamp < data_by_request[request_id][action]
                            ):
                                data_by_request[request_id][action] = timestamp
                            request_role[request_id][role] = ip
                            if (
                                action not in action_timestamps
                                or timestamp < action_timestamps[action]
                            ):
                                action_timestamps[action] = timestamp
        except Exception as e:
            print(f"Error reading {log_file_path}: {str(e)}")
            print(traceback.print_exc())


def _decode_engine_step_sheet(
    decode_engine_step_lines, engine_step_path, writer, engine_step_headers
):
    if len(decode_engine_step_lines) != 0:
        mtp_model_main_model_headers = _get_mtp_model_main_model_headers()
        decode_data = []
        decode_engine_step_headers = engine_step_headers + mtp_model_main_model_headers
        for line in decode_engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(decode_engine_step_headers, values))
            decode_data.append(row)

        df_decode = pd.DataFrame(decode_data, columns=decode_engine_step_headers)
        df_decode["prefix"] = (
            df_decode["node"]
            + "_"
            + df_decode["engine_core_str"].str.extract(r"(\d+)", expand=False)
        )
        df_decode.to_excel(writer, sheet_name="decode_engine_step", index=False)

        print(
            f"Successfully parsed decode engine step logs. "
            f"Added 'decode_engine_step' sheet to {engine_step_path}."
        )

        # dump die load and die time
        _decode_die_load_sheet(engine_step_path, writer, df_decode)

    else:
        print("No valid decode engine step record found in log files.")


def _get_mtp_model_main_model_headers():
    return [
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
        "mtp_model_start_time",
        "mtp_model_end_time",
        "execute_mtp_model_cost_time",
    ]


def _engine_step_sheet(
    engine_step_lines, engine_step_path, writer, engine_step_headers
):
    if len(engine_step_lines) != 0:
        engine_data = []
        for line in engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(engine_step_headers, values))
            engine_data.append(row)

        df_engine = pd.DataFrame(engine_data, columns=engine_step_headers)
        df_engine.to_excel(writer, sheet_name="engine_step", index=False)

        print(
            f"Successfully parsed engine step logs. Added 'engine_step' {engine_step_path}."
        )
    else:
        print("No valid engine step record found in log files.")


def _decode_die_load_sheet(engine_step_path, writer, df_decode):
    decode_die_load_columns = _get_decode_die_load_columns()
    grouped = df_decode.groupby("prefix")
    wide_blocks = []

    for prefix, group in grouped:
        group = group.reset_index(drop=True)
        filtered = group[decode_die_load_columns].copy()

        # Rename columns with prefix
        filtered.columns = [f"{prefix}_{col}" for col in filtered.columns]

        # Reset index for alignment and add to list
        wide_blocks.append(filtered.reset_index(drop=True))
    final_df = pd.concat(wide_blocks, axis=1)
    final_df.to_excel(writer, sheet_name="decode_die_load", index=False)
    print(
        f"Successfully parsed decode die load. "
        f"Added 'decode_die_load' sheet to {engine_step_path}."
    )


def _get_decode_die_load_columns():
    return [
        "execute_model_start_time",
        "total_tokens",
        "running_reqs_num_after_step",
        "waiting_reqs_num_after_step",
        "execute_model_cost_time(ms)",
        "start_free_block_num",
        "cost_blocks_num",
    ]


def _get_action_map() -> dict:
    """
        the tuple's number means different phase:
        proxy -- 0
        P side:
            api server -- 1
            engine core -- 2
        D side:
            api server -- 3
            engine core -- 4
            worker -- 5
    """
    return {
        "PD api server get request": ("P侧api server收到请求", 1),
        "Get prefill engine request and start pickle": ("触发engine处理请求", 1),
        "Finish process request in prefill engine": ("engine结束tokennizer", 1),
        "Start process request in prefill engine": ("engine准备开始处理输入请求", 2),
        "Prefill add waiting queue": ("P侧请求添加到waiting队列", 2),
        "try to schedule in waiting queue": ("首次尝试加入running队列", 2),
        "fail to add result of kv insufficient": ("首次kv不足加入失败", 2),
        "Prefill get new_blocks": ("P侧申请完成KV", 2),
        "success add to seq groups": ("P侧请求成功加入running队列", 2),
        "Prefill start execute_model": ("P侧开始execute model", 2),
        "Prefill start execute main model": ("P侧开始execute main model", 2),
        "Prefill done execute main model": ("P侧完成execute main model", 2),
        "Prefill start execute mtp model": ("P侧开始execute mtp model", 2),
        "Prefill done execute mtp model": ("P侧完成execute mtp model", 2),
        "Prefill done execute_model": ("P侧完成execute model", 2),
        "Start to send output in prefill stage": ("P侧engine异步发送输出", 2),
        "Client get prefill output": ("client收到输出并入队", 1),
        # "Pop output queues": "client出队",
        "Finish prefill pickle and start response": ("P侧api server收到请求准备返回", 0),
        "Enter decode to generate": ("D侧api server收到请求", 3),
        "Start to dispatch decode request": ("进入engine分发请求", 4),
        "Add need pulling sequence": ("D侧请求添加到need pulling队列", 4),
        "Start pull kv": ("开始pull kv", 5),
        "Finish pull kv": ("结束pull kv", 5),
        "Prefill free kv blocks": ("P侧释放KV(和前后列时间戳可能存在时钟误差)", 2),
        "Start append running sequece for decode": ("pull kv结束添加到running队列", 4),
        "Start to send output": ("触发首个decode token执行", 4),
        "Decode done execute_model": ("D侧完成execute model", 4),
        "First decode output token": ("返回第一个token", 3),
        "Second decode output token": ("返回第二个token", 3),
        "Third decode output token": ("返回第三个token", 3),
        "Finish decode pickle and start response": ("D侧api server收到推理结果", 3),
        "Decode send ip, rank": ("D侧发送ip, rank", 4),
        "Prefill receive and record decode ip, rank": ("P侧收到ip, rank", 2),
        "Prefill send metadata": ("P侧发送metadata", 2),
        "Decode receive metadata": ("D侧收到metadata", 4),
    }


def _set_decode_info(decode_engine_step_lines, engine_core_info, line):
    core_match = re.search(r"(\d+-\d+\.\d+)", line)
    if core_match:
        core_str = core_match.group(1)
        info = engine_core_info.get(
            core_str,
            {
                "main_model_start_time": 0.0,
                "main_model_end_time": 0.0,
                "execute_main_model_cost_time": 0.0,
                "mtp_model_start_time": 0.0,
                "mtp_model_end_time": 0.0,
                "execute_mtp_model_cost_time": 0.0,
            },
        )
        line = (
            line.strip()
            + f"|{info.get('main_model_start_time')}|{info.get('main_model_end_time')}|{info.get('execute_main_model_cost_time')}|{info.get('mtp_model_start_time')}|{info.get('mtp_model_end_time')}|{info.get('execute_mtp_model_cost_time')}\n"
        )
    decode_engine_step_lines.append(line)
    return line


def _get_mtp_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        mtp_start = float(parts[1])
        mtp_end = float(parts[2])
        mtp_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "mtp_model_start_time": mtp_start,
                "mtp_model_end_time": mtp_end,
                "execute_mtp_model_cost_time": mtp_cost,
            }
        )


def _get_main_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        main_start = float(parts[1])
        main_end = float(parts[2])
        main_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "main_model_start_time": main_start,
                "main_model_end_time": main_end,
                "execute_main_model_cost_time": main_cost,
            }
        )


def _calculate_time_difference(df_time, connector_type):
    # calculate time stream end to end of p2d/d2p connector
    try:
        time_record = pd.read_excel(df_time)
        print("time_analysis.xlsx文件读取成功")
    except Exception as e:
        print(f"time_analysis.xlsx文件读取失败: {e}")
        return

    time_pairs = _get_time_pairs(connector_type)

    time_difference = pd.DataFrame()
    first_col_name = time_record.columns[0]
    first_col_data = time_record.iloc[:, 0].copy()
    time_difference[first_col_name] = first_col_data

    for start_col, end_col, col_name in time_pairs:
        time_difference[col_name] = time_record.apply(
            lambda row: safe_subtract(row[start_col], row[end_col]), axis=1
        )

    # calculate the average time of all requests traced for each phase
    averages = {}
    for col in time_difference.columns[1:]:
        column_data = time_difference[col].iloc[2:]

        valid_values = []
        for val in column_data:
            if val != FILL_VALUE:
                try:
                    valid_values.append(float(val))
                except:
                    pass

        if valid_values:
            avg_value = sum(valid_values) / len(valid_values)
            averages[col] = round(avg_value, 6)
        else:
            averages[col] = FILL_VALUE
    
    time_difference.iloc[0, 0] = "average_time_difference(s)"
    for col in time_difference.columns[1:]:
        time_difference.at[0, col] = averages.get(col, FILL_VALUE)

    with pd.ExcelWriter(df_time, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        time_difference.to_excel(writer, sheet_name="time_difference", index=False)

    print(f"Successfully calculate time difference. Check {df_time} for details.")


def safe_subtract(a, b):
    if a == FILL_VALUE or b == FILL_VALUE:
        return FILL_VALUE
    try:
        return float(b) - float(a)
    except (ValueError, TypeError):
        return FILL_VALUE


def _get_time_pairs(connector_type):
    commen_time_pairs = [
        ("PD api server get request", "Get prefill engine request and start pickle", 
         "PD api server get request -> Get prefill engine request and start pickle"),
        ("Get prefill engine request and start pickle", "Finish process request in prefill engine", 
         "Get prefill engine request and start pickle -> Finish process request in prefill engine"),
        ("Finish process request in prefill engine", "Start process request in prefill engine", 
         "Finish process request in prefill engine -> Start process request in prefill engine"),
        ("Start process request in prefill engine", "Prefill add waiting queue", 
         "Start process request in prefill engine -> Prefill add waiting queue"),
        ("Prefill add waiting queue", "try to schedule in waiting queue", 
         "Prefill add waiting queue -> try to schedule in waiting queue"),
        ("try to schedule in waiting queue", "Prefill get new_blocks", 
         "try to schedule in waiting queue -> Prefill get new_blocks"),
        ("Prefill get new_blocks", "success add to seq groups", 
         "Prefill get new_blocks -> success add to seq groups"),
        ("success add to seq groups", "Prefill start execute_model", 
         "success add to seq groups -> Prefill start execute_model"),
        ("Prefill start execute_model", "Prefill done execute_model", 
         "Prefill start execute_model -> Prefill done execute_model"),
        ("Enter decode to generate", "Start to dispatch decode request", 
         "Enter decode to generate -> Start to dispatch decode request"),
        ("Start to dispatch decode request", "Add need pulling sequence", 
         "Start to dispatch decode request -> Add need pulling sequence"),
        ("Add need pulling sequence", "Start pull kv", 
         "Add need pulling sequence -> Start pull kv"),
        ("Start pull kv", "Finish pull kv", 
         "Start pull kv -> Finish pull kv"),
        ("Finish pull kv", "Start append running sequece for decode", 
         "Finish pull kv -> Start append running sequece for decode"),
        ("Finish pull kv", "Prefill free kv blocks", 
         "Finish pull kv -> Prefill free kv blocks"),
        ("Start append running sequece for decode", "Start to send output", 
         "Start append running sequece for decode -> Start to send output"),
        ("Start to send output", "First decode output token", 
         "Start to send output -> First decode output token"),
        ("First decode output token", "Second decode output token", 
         "First decode output token -> Second decode output token"),
        ("Second decode output token", "Third decode output token", 
         "Second decode output token -> Third decode output token"),
        ("Third decode output token", "Finish decode pickle and start response", 
         "Third decode output token -> Finish decode pickle and start response"),    
    ]

    p2d_time_pairs = [
        ("Prefill done execute_model", "Start to send output in prefill stage", 
         "Prefill done execute_model -> Start to send output in prefill stage"),
        ("Start to send output in prefill stage", "Finish prefill pickle and start response", 
         "Start to send output in prefill stage -> Finish prefill pickle and start response"),
        ("Finish prefill pickle and start response", "Client get prefill output", 
         "Finish prefill pickle and start response -> Client get prefill output"),
        ("Client get prefill output", "Enter decode to generate", 
         "Client get prefill output -> Enter decode to generate"),
    ]

    d2p_time_pairs = [
        ("Start to dispatch decode request", "Decode send ip, rank", 
         "Start to dispatch decode request -> Decode send ip, rank"),
        ("Decode send ip, rank", "Prefill receive and record decode ip, rank", 
         "Decode send ip, rank -> Prefill receive and record decode ip, rank"),
        ("Prefill done execute_model", "Prefill send metadata", 
         "Prefill done execute_model -> Prefill send metadata"),
        ("Prefill send metadata", "Decode receive metadata", 
         "Prefill send metadata -> Decode receive metadata"),
        ("Decode receive metadata", "Start pull kv", 
         "Decode receive metadata -> Start pull kv"),
    ]

    if connector_type == "p2d":
        return commen_time_pairs + p2d_time_pairs
    return commen_time_pairs + d2p_time_pairs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # connector type: "p2d" -- default, "d2p" -- when using feature "First D Then P"
        print("Please input log directory. e.g.: python parse_logs.py path/to/all_pd_logs_direcotry p2d/d2p")
        exit()
    root_dir = sys.argv[1]
    if len(sys.argv) == 3:
        connector_type = sys.argv[2]
    else:
        connector_type = "p2d"
    parse_trace_logs(root_dir)
    _calculate_time_difference(os.path.join(root_dir, "time_analysis.xlsx"), connector_type)