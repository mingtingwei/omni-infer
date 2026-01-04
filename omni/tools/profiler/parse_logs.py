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
import datetime

FILL_VALUE = "-"

def parse_trace_logs(root_dir, time_pairs, action_maps):
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
            summary_data = {
                    "RequestID": list(data_by_request.keys()),
                    "ActionCount": [
                        len(actions) for actions in data_by_request.values()
                    ],
                }
            df_summary = pd.DataFrame(summary_data)
            df_final = _get_final_df(data_by_request, request_role, action_maps)
            df_diff = _calculate_time_difference(df_final, time_pairs)

            with pd.ExcelWriter(time_analysis_path, engine="openpyxl") as writer:
                df_final.to_excel(writer, sheet_name="time_analysis", index=False)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)
                df_diff.to_excel(writer, sheet_name="time_difference", index=False)

            print(
                f"Successfully parsed time analysis files. Check {time_analysis_path} for details."
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


def _calculate_time_difference(time_record, time_pairs):
    """Calculate time stream end to end of p2d/d2p connector"""
    
    time_difference = pd.DataFrame()
    first_col_name = time_record.columns[0]
    first_col_data = time_record.iloc[:, 0].copy()
    time_difference[first_col_name] = first_col_data

    for start_col, end_col in time_pairs:
        col_name = start_col + " -> " + end_col
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

    return time_difference


def generate_connected_graphviz_flowchart(time_difference, time_pairs, action_maps):
    """Generate Graphviz DOT code of flowchart"""
    all_nodes = set()
    for start_node, end_node in time_pairs:
        all_nodes.add(start_node)
        all_nodes.add(end_node)

    # classify nodes to different types
    action_class = dict(map(lambda k, v: (k, v[1]), action_maps.keys(), action_maps.values()))
    
    # node_categories: {class1: [node1, node2], class2: [node3, node4]}
    classes = [
        "proxy", 
        "prefill_api_server", 
        "prefill_engine", 
        "decode_api_server", 
        "decode_engine"
    ]

    node_categories = {key: [] for key in classes}
    for node in all_nodes:     
        if node not in action_class:
            print(f"Warning: {node} is not defined in action_maps")
            continue
            
        node_class = action_class[node]
        if node_class == 0:
            node_categories["proxy"].append(node)
        elif node_class == 1:
            node_categories["prefill_api_server"].append(node)
        elif node_class == 2:
            node_categories["prefill_engine"].append(node)
        elif node_class == 3:
            node_categories["decode_api_server"].append(node)
        else:
            node_categories["decode_engine"].append(node)
    
    # define node colors
    node_colors = {
        "proxy": {"fill": "#FFEBEE", "stroke": "#C62828", "bg": "#FFCDD2"},
        "prefill_api_server": {"fill": "#E8F5E9", "stroke": "#2E7D32", "bg": "#C8E6C9"},
        "prefill_engine": {"fill": "#E3F2FD", "stroke": "#1565C0", "bg": "#BBDEFB"},
        "decode_api_server": {"fill": "#FFF3E0", "stroke": "#EF6C00", "bg": "#FFE0B2"},
        "decode_engine": {"fill": "#F3E5F5", "stroke": "#7B1FA2", "bg": "#E1BEE7"}
    }
    
    # generate code
    dot_code = 'digraph G {\n'
    dot_code += '    // Graph settings\n'
    dot_code += '    rankdir=TB;           // Top to Bottom layout\n'
    dot_code += '    compound=true;        // Allow edges between clusters\n'
    dot_code += '    nodesep=0.5;          // Horizontal spacing between nodes\n'
    dot_code += '    ranksep=0.8;          // Vertical spacing between ranks\n'
    dot_code += '    fontname="Arial";\n'
    dot_code += '    fontsize=16;\n\n'
    
    dot_code += '    // Default node style\n'
    dot_code += '    node [\n'
    dot_code += '        shape=box,\n'
    dot_code += '        style="rounded,filled",\n'
    dot_code += '        fontname="Arial",\n'
    dot_code += '        fontsize=18,\n'
    dot_code += '        penwidth=2\n'
    dot_code += '    ];\n\n'
    
    dot_code += '    // Default edge style\n'
    dot_code += '    edge [\n'
    dot_code += '        color="#666666",\n'
    dot_code += '        penwidth=3,\n'
    dot_code += '        fontname="Arial",\n'
    dot_code += '        fontsize=14\n'
    dot_code += '    ];\n\n'
    
    # make subgraph
    node_id_map = {}
    cluster_id = 0
    
    for category, nodes in node_categories.items():
        if not nodes:
            continue
            
        cluster_name = f"cluster_{cluster_id}"
        readable_title = category.replace('_', ' ').title()
        colors = node_colors.get(category, {"fill": "#FFFFFF", "stroke": "#000000", "bg": "#F5F5F5"})
        
        dot_code += f'    // {readable_title}\n'
        dot_code += f'    subgraph {cluster_name} {{\n'
        dot_code += f'        label="{readable_title}";\n'
        dot_code += f'        bgcolor="{colors["bg"]}";\n'
        dot_code += f'        color="{colors["stroke"]}";\n'
        dot_code += f'        penwidth=3;\n'
        dot_code += f'        fontname="Arial";\n'
        dot_code += f'        fontsize=20;\n\n'
        
        for node in nodes:
            node_id = node.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            node_label = node
            node_id_map[node] = node_id
            
            dot_code += f'        {node_id} [\n'
            dot_code += f'            label="{node_label}",\n'
            dot_code += f'            fillcolor="{colors["fill"]}",\n'
            dot_code += f'            color="{colors["stroke"]}"\n'
            dot_code += f'        ];\n'
        
        dot_code += '    }\n\n'
        cluster_id += 1
    
    # add edges
    dot_code += '    // Connections between nodes\n'
    
    for start_node, end_node in time_pairs:
        if start_node not in node_id_map or end_node not in node_id_map:
            continue
            
        start_id = node_id_map[start_node]
        end_id = node_id_map[end_node]
        
        # get time difference
        phase = start_node + " -> " + end_node
        if phase in time_difference.columns:
            time_val = time_difference.iloc[0][phase]
            if pd.isna(time_val) or time_val == "-":
                avg_time = "N/A"
            else:
                try:
                    phase_time = pd.to_numeric(time_val)
                    if phase_time < 1:
                        avg_time = f"{phase_time*1000:.3f}ms"
                    else:
                        avg_time = f"{phase_time:.3f}s"
                except:
                    avg_time = str(time_val)
        else:
            avg_time = "N/A"
        
        dot_code += f'    {start_id} -> {end_id} [label="{avg_time}"];\n'
    
    dot_code += '}\n'
    
    return dot_code


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


def _get_final_df(data_by_request, request_role, origin_action_map):
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
            engine -- 2
        D side:
            api server -- 3
            engine -- 4
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
        "Start pull kv": ("开始pull kv", 4),
        "Finish pull kv": ("结束pull kv", 4),
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


def safe_subtract(a, b):
    if a == FILL_VALUE or b == FILL_VALUE:
        return FILL_VALUE
    try:
        return float(b) - float(a)
    except (ValueError, TypeError):
        return FILL_VALUE


def _get_time_pairs(connector_type):
    commen_time_pairs = [
        ("PD api server get request", "Get prefill engine request and start pickle"),
        ("Get prefill engine request and start pickle", "Finish process request in prefill engine"),
        ("Finish process request in prefill engine", "Start process request in prefill engine"),
        ("Start process request in prefill engine", "Prefill add waiting queue"),
        ("Prefill add waiting queue", "try to schedule in waiting queue"),
        ("try to schedule in waiting queue", "Prefill get new_blocks"),
        ("Prefill get new_blocks", "success add to seq groups"),
        ("success add to seq groups", "Prefill start execute_model"),
        ("Prefill start execute_model", "Prefill done execute_model"),
        ("Enter decode to generate", "Start to dispatch decode request"),
        ("Start to dispatch decode request", "Add need pulling sequence"),
        ("Start pull kv", "Finish pull kv"),
        ("Finish pull kv", "Start append running sequece for decode"),
        ("Finish pull kv", "Prefill free kv blocks"),
        ("Start append running sequece for decode", "Start to send output"),
        ("Start to send output", "First decode output token"),
        ("First decode output token", "Second decode output token"),
        ("Second decode output token", "Third decode output token"),
        ("Third decode output token", "Finish decode pickle and start response"),    
    ]

    p2d_time_pairs = [
        ("Prefill done execute_model", "Start to send output in prefill stage"),
        ("Start to send output in prefill stage", "Finish prefill pickle and start response"),
        ("Finish prefill pickle and start response", "Client get prefill output"),
        ("Add need pulling sequence", "Start pull kv"),
        ("Client get prefill output", "Enter decode to generate"),
    ]

    d2p_time_pairs = [
        ("Finish process request in prefill engine", "Prefill receive and record decode ip, rank"),
        ("Start to dispatch decode request", "Decode send ip, rank"),
        ("Decode send ip, rank", "Prefill receive and record decode ip, rank"),
        ("Prefill done execute_model", "Prefill send metadata"),
        ("Prefill send metadata", "Decode receive metadata"),
        ("Add need pulling sequence", "Decode receive metadata"),
        ("Decode receive metadata", "Start pull kv"),
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

    time_pairs = _get_time_pairs(connector_type)
    action_maps = _get_action_map()
    
    parse_trace_logs(root_dir, time_pairs, action_maps)
    
    time_diff_path = os.path.join(root_dir, "time_analysis.xlsx")
    time_difference_df = pd.read_excel(time_diff_path, sheet_name="time_difference")
    
    stream_gv_code = generate_connected_graphviz_flowchart(time_difference_df, time_pairs, action_maps)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gv_file_name = f"flowchart_{connector_type}_{timestamp}.gv"
    output_dir = os.path.join(root_dir, "graphviz_output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    gv_file_path = os.path.join(output_dir, gv_file_name)
    
    with open(gv_file_path, "w", encoding="utf-8") as f:
        f.write(stream_gv_code)
    
    print(f"✅ Graphviz DOT codes have been save in: {gv_file_path},",
          "check the flowchart through https://dreampuf.github.io/GraphvizOnline,",
          "or using VSCode plugin - Graphviz Interactive Preview.")