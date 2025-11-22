import re
import os
import ast
import sys
import uuid
import json
from collections import defaultdict

action_dict = {
    'Start to schedule': '调度进程收到请求',   # 0
    'Finish to choose device and add request': '选择设备，准备传递请求',
    'Start to send request to pd api server': '准备发送请求给prefill api server',

    'p_1 prefill api server收到请求': 'p_1 prefill api server收到请求',  # 3
    'p_2 触发engine处理请求': 'p_2 触发engine处理请求',
    'p_3 engine开始tokennizer': 'p_3 engine开始tokennizer',
    'p_3 engine结束tokennizer': 'p_3 engine结束tokennizer',             # 6
    'p_4 tokennizer to sche': 'p_4 tokennizer to sche',
    'p_5 P侧添加到bootstrap队列之后': 'p_5 P侧添加到bootstrap队列之后',
    'P侧握手完成': 'P侧握手完成',
    'p_6 握手完成加到waiting队列': 'p_6 握手完成加到waiting队列',  # 10
    'p_7 组batch完成 开始run batch': 'p_7 组batch完成 开始run batch',
    'p_8 Push a new batch to the input queue': 'p_8 Push a new batch to the input queue',
    'p_9 开始执行Run forward': 'p_9 开始执行Run forward', # 13
    'p_9 结束执行Run forward': 'p_9 结束执行Run forward',
    'p_13 开始发送kv cache': 'p_13 开始发送kv cache',
    'p_14 完成发送kv cache': 'p_14 完成发送kv cache',
    'p_15 P侧释放KV': 'p_15 P侧释放KV',                  #17
    'p_16 client收到输出并入队': 'p_16 client收到输出并入队', 
    'p_17 client出队': 'p_17 client出队',
    'p_18 api server收到请求准备返回': 'p_18 api server收到请求准备返回',  # 20

    'Get response from pd api server': '调度进程收到api server返回的响应',
    'Finish to chosse device and start decode generate': '调度进程选择decoder device 准备发送请求',
    'Finish to decode': '调度进程完成发送',

    'd_1 decode api server收到请求': 'd_1 decode api server收到请求',  # 24
    'd_2 触发engine处理请求': 'd_2 触发engine处理请求',
    'd_3 engine开始tokennizer': 'd_3 engine开始tokennizer',
    'd_3 engine结束tokennizer': 'd_3 engine结束tokennizer',           # 27
    'd_4 tokennizer to sche': 'd_4 tokennizer to sche',
    'd_5 scheduler开始处理请求': 'd_5 scheduler开始处理请求', 
    'd_6 D侧添加到prealloc_queue队列之后': 'd_6 D侧添加到prealloc_queue队列之后', # 30
    'd_8 开始分配kv缓存': 'd_8 开始分配kv缓存',
    'd_9 d侧开始握手': 'd_9 d侧开始握手',                                        
    'd_7 Add need pullling sequence': 'd_7 Add need pullling sequence',          # 33
    'd_11 d侧收到kv传输完成通知': 'd_11 d侧收到kv传输完成通知',
    'd_12 轮询到状态 kv cache传输完成 开始加入到waiting队列': 'd_12 轮询到状态 kv cache传输完成 开始加入到waiting队列',
    'd_18 触发首个decode token执行': 'd_18 触发首个decode token执行',
    'd_19 decoder返回第一个token': 'd_19 decoder返回第一个token',  # 37
    'd_20 decoder返回第二个token': 'd_20 decoder返回第二个token',
    'd_21 api server收到推理结果': 'd_21 api server收到推理结果',   # 39
}

EXTRA_SPANS = [
    (0, 39, "sgLang trace", "big_single"),   # root span
    (0, 37, "TTFT", "big_single"),
    (3, 20, "Prefill", "big_single"),

    (10, 20, "P running", "big"),
    (10, 13, "Add to running for prefill", "big"),

    (15, 18, "Prefill free kv", "special"),
    (18, 20, "Send output in prefill stage", "big"),

    (29, 35, "D schdule", "big"),
    (36, 39, "D running", "big")
]

SMALL_SPANS = [
    (0, 1, "Proxy process the request and choose device"),
    (1, 2, "Proxy prepare the request needed sent"),
    (2, 3, "Request transmitting"),

    (3, 4, "Prefill processes the request"),
    (4, 5, "P Engine processes the request"),
    (5, 6, "P Engine tokenizer"),
    (6, 7, "P Tokenizer to schduler"),
    (7, 8, "P added to the bootstrap queue"),
    (8, 9, "P handshaking"),
    (9, 10, "P added to the 'Wait' queue"),
    (10, 11, "P grouping the batch and trying to add to 'Run'"),
    (11, 12, "P pushing the batch into input queue"),
    (12, 13, "P Execute engine step"),
    (13, 14, "P execute Run forward"),
    (14, 15, "Preparing to send KV cache"),
    (15, 16, "Sending KV cache"),
    (16, 17, "P Releasing KV cache"),
    (17, 18, "Engine sends output to client, client enqueues."),
    (18, 19, "Client waiting in the queue"),
    (19, 20, "Sending request to api server"),
    (20, 21, "Api server return request to the proxy"),

    (21, 22, "Proxy choosing D device"),
    (22, 23, "Proxy sending request to the chosen D device"),
    (23, 24, "Request in progress"),

    (24, 25, "D processing the request"),
    (25, 26, "D Engine processing the request"),
    (26, 27, "D Engine tokenizer"),
    (27, 28, "D Tokenizer to schduler"),
    (28, 29, "D Request in progress"),
    (29, 30, "D Schduler processing the request and adding D into the prealloc queue"),
    (30, 31, "D waiting in the prealloc queue"),
    (31, 32, "D allocating KV cache"),
    (32, 33, "D handshaking"),
    (33, 34, "D waiting in the 'need pulling' queue"),
    (34, 35, "D adding into the 'Wait' queue"),
    (35, 36, "D adding into the 'Run' queue"),
    (36, 37, "D executing the first token"),
    (37, 38, "D executing the second token"),
    (38, 39, "D Execute infer until completion and return the result")
]

def normalize_reqid(reqid):
    if reqid.startswith("chatcmpl-"):
        return reqid[len("chatcmpl-"):]
    return reqid

def decode_worker_step(node, engine_core_str, lines_decode_worker_step):
    node_key = node + '|' + engine_core_str
    worker_step: int = None
    if node_key in lines_decode_worker_step.keys():
        worker_step = lines_decode_worker_step[node_key]
    return worker_step

def parse_files(folder_path):
    lines = []
    lines_decode_step = []
    lines_decode_worker_step = {}
    step_data = []  # req_ids : start engine time, end engine time, 同一个batch总的tokens
    step_data_decode = {}
    metric = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):  # 过滤文件
            print(item_path)  # 解析文件
            with open(item_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    if 'CompletionMetric' in line:
                        metric.append(line)
                        continue
                    if 'profile' not in line:
                        continue
                    node_info = None
                    if '_NODE_' in item:
                        idx0 = item.find('_NODE_') + len('_NODE_')
                        idx1 = item.find('_', idx0)
                        node_info = item[idx0:idx1]
                    if 'Times' in line:
                        lines.append(line)
                    if 'engine_step start' in line:  # 含有engine_step start的文件必然有_NODE_信息
                        if node_info.startswith('P'):
                            lines.append(line)
                        else:
                            lines_decode_step.append(line)
                    if node_info != None:
                        if 'Times' in line or node_info.startswith('P'):
                            lines[-1] = lines[-1] + '|NODE=' + node_info + '.'
                        elif 'worker_step start' in line:
                            # profile: worker_step start:4790|1751437716.6964805|model_step=1097
                            id0 = line.find('worker_step start:')
                            id1 = line.find('|', id0)
                            id2 = line.find('|', id1 + 1)
                            d_step_key = node_info + '|' + line[id0 + len('worker_step start:'):id2]
                            worker_step = int(line[id2 + len('|model_step='):])
                            lines_decode_worker_step[d_step_key] = worker_step
                        else:
                            lines_decode_step[-1] = lines_decode_step[-1] + '|NODE=' + node_info + '.'
                            idx0 = item.find('_NODE_') + len('_NODE_')
                            idx1 = item.find('_', idx0) + 1
                            idx0 = item.find('_', idx1)
                            pid = item[idx1:idx0]
                            lines_decode_step[-1] = lines_decode_step[-1] + '|PID=' + pid + '.'

    print(f'read all files done')

    req_datas = {}  # req_id : timstamp actions
    decode_token_nums = {}  # reqid : decoder tokens
    for line in metric:
        if 'CompletionMetric' in line:
            idx0 = line.find('profile REQ_ID[') + len('profile REQ_ID[')
            idx1 = line.find(']', idx0)
            req_id = line[idx0:idx1]
            idx0 = line.find('num_completion_tokens=') + len('num_completion_tokens=')
            idx1 = line.find('.', idx0)
            output_tokens = line[idx0:idx1]
            decode_token_nums[req_id] = int(output_tokens)
    print(f'parse CompletionMetric done')

    # parse decode_step -> step_data_decode
    for line in lines_decode_step:
        if 'engine_step start' in line:
            # profile: engine_step start:1757043250.201461|finish:1757043250.5910044|
            # execute time:389.5432949066162|seqs:17|tokens:52674|waiting_reqs_num_after_step=0|
            # reqs_ids=[4592131703522744441, ..., 8305509393532383720, 6821854389136394445]|
            # bs_tokens=[3425, ..., 3460, 740, 4017, 8107, 2560]|
            # execute_model_start_time=1757043250.201461|execute_model_end_time=1757043250.5910044|
            # execute_model_cost_time=389.5432949066162|kv_cache_usage=0.0166015625|
            # kv_blocks_num=1024|start_free_block_num=1007|end_free_block_num=1007|
            # cost_blocks_num=17|engine_core_str=21382|1757043249.8365471
            id0 = line.find('engine_step start:')
            id1 = line.find('|finish:', id0)
            start_timestamp = line[id0 + len('engine_step start:'):id1]
            start_timestamp = float(start_timestamp)
            id0 = line.find('|finish:')
            id1 = line.find('|execute time:', id0)
            finish_timestamp = line[id0 + len('|finish:'):id1]
            id0 = line.find('|execute time:')
            id1 = line.find('|seqs:', id0)
            execute_time = line[id0 + len('|execute time:'):id1]
            id0 = line.find('|seqs:')
            id1 = line.find('|tokens:', id0)
            seqs = line[id0 + len('|seqs:'):id1]
            id0 = line.find('|tokens:')
            id1 = line.find('|waiting_reqs_num_after_step=', id0)
            tokens = line[id0 + len('|tokens:'):id1]
            if int(tokens) == 0:
                continue
            id0 = line.find('|waiting_reqs_num_after_step=')
            id1 = line.find('|reqs_ids=', id0)
            waiting_num = line[id0 + len('|waiting_reqs_num_after_step='):id1]
            id0 = line.find('|reqs_ids=')
            id1 = line.find('|bs_tokens=', id0)
            reqs = line[id0 + len('|reqs_ids='):id1]
            pattern = r'[0-9]+'
            uuids_reqs = re.findall(pattern, reqs)
            # print(f"{uuids_reqs=}")
            id0 = line.find('|bs_tokens=')
            id1 = line.find('|execute_model_start_time=', id0)
            tokens_per_req = ast.literal_eval(line[id0 + len('|bs_tokens='):id1])
            id0 = line.find('|execute_model_start_time=')
            id1 = line.find('|execute_model_end_time=', id0)
            model_start_timestamp = line[id0 + len('|execute_model_start_time='):id1]
            id0 = line.find('|execute_model_end_time=')
            id1 = line.find('|execute_model_cost_time=', id0)
            model_finish_timestamp = line[id0 + len('|execute_model_end_time='):id1]
            id0 = line.find('|execute_model_cost_time=')
            id1 = line.find('|kv_cache_usage=', id0)
            model_execute_time = line[id0 + len('|execute_model_cost_time='):id1]
            id0 = line.find('|kv_cache_usage=')
            id1 = line.find('|kv_blocks_num=', id0)
            kv_cache_usage = line[id0 + len('|kv_cache_usage='):id1]
            kv_cache_usage = float(kv_cache_usage)
            id0 = line.find('|kv_blocks_num=')
            id1 = line.find('|start_free_block_num=', id0)
            kv_blocks_num = line[id0 + len('|kv_blocks_num='):id1]
            id0 = line.find('|start_free_block_num=')
            id1 = line.find('|end_free_block_num=', id0)
            start_free_block_num = line[id0 + len('|start_free_block_num='):id1]
            id0 = line.find('|end_free_block_num=')
            id1 = line.find('|cost_blocks_num=', id0)
            end_free_block_num = line[id0 + len('|end_free_block_num='):id1]
            id0 = line.find('|cost_blocks_num=')
            id1 = line.find('|engine_core_str=', id0)
            cost_blocks_num = line[id0 + len('|cost_blocks_num='):id1]
            id0 = line.find('|engine_core_str=')
            id1 = line.find('|NODE=', id0)
            engine_core_str = line[id0 + len('|engine_core_str='):id1]
            id0 = line.find('|NODE=')
            id1 = line.find('.', id0)
            node = line[id0 + len('|NODE='):id1]
            id0 = line.find('|PID=', id1)
            id1 = line.find('.', id0)
            pid = line[id0 + len('|PID='):id1]
            worker_step = decode_worker_step(node, engine_core_str, lines_decode_worker_step)
            if worker_step is None:
                print(f'not find worker_step for [{line}]')
            dict_key = node + '_' + pid
            if dict_key not in step_data_decode.keys():
                step_data_decode[dict_key] = []
            step_data_decode[dict_key].append(
                [start_timestamp, finish_timestamp, execute_time, seqs, int(tokens), waiting_num, \
                 float(model_start_timestamp), float(model_finish_timestamp), float(model_execute_time), kv_cache_usage,
                 kv_blocks_num, start_free_block_num, \
                 end_free_block_num, cost_blocks_num, worker_step])
    print(f'parse decode_step done')

    # parse and prefill action timestamp -> step_data, req_data
    for line in lines:
        # engine_step start:1757043627.0652876|finish:1757043627.256134|execute time:190.84644317626953|
        # seqs:1|tokens:3995|waiting_reqs_num_after_step=0|reqs_ids=[5746913226436875047]|bs_tokens=[667]|
        # execute_model_start_time=1757043627.0652876|execute_model_end_time=1757043627.256134|
        # execute_model_cost_time=190.84644317626953|kv_cache_usage=0.0009765625|kv_blocks_num=1024|
        # start_free_block_num=1023|end_free_block_num=1023|cost_blocks_num=1|engine_core_str=21345|1757043626.8809538
        if 'engine_step start' in line:
            # print(f"{line=}")
            id0 = line.find('engine_step start:')
            id1 = line.find('|finish:', id0)
            start_timestamp = line[id0 + len('engine_step start:'):id1]
            start_timestamp = float(start_timestamp)
            id0 = line.find('|finish:')
            id1 = line.find('|execute time:', id0)
            finish_timestamp = line[id0 + len('|finish:'):id1]
            id0 = line.find('|execute time:')
            id1 = line.find('|seqs:', id0)
            execute_time = line[id0 + len('|execute time:'):id1]
            id0 = line.find('|seqs:')
            id1 = line.find('|tokens:', id0)
            seqs = line[id0 + len('|seqs:'):id1]
            id0 = line.find('|tokens:')
            id1 = line.find('|waiting_reqs_num_after_step=', id0)
            tokens = line[id0 + len('|tokens:'):id1]
            if int(tokens) == 0:
                continue
            id0 = line.find('|waiting_reqs_num_after_step=')
            id1 = line.find('|reqs_ids=', id0)
            waiting_num = line[id0 + len('|waiting_reqs_num_after_step='):id1]
            id0 = line.find('|reqs_ids=')
            id1 = line.find('|bs_tokens=', id0)
            reqs = line[id0 + len('|reqs_ids='):id1]
            pattern = r'[0-9]+'
            uuids_reqs = re.findall(pattern, reqs)
            # print(f"{uuids_reqs=}")
            id0 = line.find('|bs_tokens=')
            id1 = line.find('|execute_model_start_time=', id0)
            tokens_per_req = ast.literal_eval(line[id0 + len('|bs_tokens='):id1])
            id0 = line.find('|execute_model_start_time=')
            id1 = line.find('|execute_model_end_time=', id0)
            model_start_timestamp = line[id0 + len('|execute_model_start_time='):id1]
            id0 = line.find('|execute_model_end_time=')
            id1 = line.find('|execute_model_cost_time=', id0)
            model_finish_timestamp = line[id0 + len('|execute_model_end_time='):id1]
            id0 = line.find('|execute_model_cost_time=')
            id1 = line.find('|kv_cache_usage=', id0)
            model_execute_time = line[id0 + len('|execute_model_cost_time='):id1]
            id0 = line.find('|kv_cache_usage=')
            id1 = line.find('|kv_blocks_num=', id0)
            kv_cache_usage = line[id0 + len('|kv_cache_usage='):id1]
            kv_cache_usage = float(kv_cache_usage)
            id0 = line.find('|kv_blocks_num=')
            id1 = line.find('|start_free_block_num=', id0)
            kv_blocks_num = line[id0 + len('|kv_blocks_num='):id1]
            id0 = line.find('|start_free_block_num=')
            id1 = line.find('|end_free_block_num=', id0)
            start_free_block_num = line[id0 + len('|start_free_block_num='):id1]
            id0 = line.find('|end_free_block_num=')
            id1 = line.find('|cost_blocks_num=', id0)
            end_free_block_num = line[id0 + len('|end_free_block_num='):id1]
            id0 = line.find('|cost_blocks_num=')
            id1 = line.find('|engine_core_str=', id0)
            cost_blocks_num = line[id0 + len('|cost_blocks_num='):id1]
            id0 = line.find('|engine_core_str=')
            id1 = line.find('|NODE=', id0)
            engine_core_str = line[id0 + len('|engine_core_str='):id1]
            id0 = line.find('|NODE=')
            id1 = line.find('.', id0)
            node = line[id0 + len('|NODE='):id1]
            step_data.append([start_timestamp, finish_timestamp, execute_time, seqs, tokens, waiting_num, uuids_reqs, \
                              tokens_per_req, model_start_timestamp, model_finish_timestamp, model_execute_time, \
                              kv_cache_usage, kv_blocks_num, start_free_block_num, end_free_block_num, cost_blocks_num,
                              node])
        if 'Times' in line:
            # print(f"{line=}")
            # profile REQ_ID[713430512380037560] action:d_4 tokennizer to sche.Timestamp 1757043070.6599956
            first_index = line.index('profile REQ_ID[')
            second_index = line.index(']', first_index)
            fifth_index = line.index(' action:', second_index)
            sixth_index = line.index('.', fifth_index)
            seventh_index = line.index('Timestamp ', sixth_index)
            last_index = line.index('|', seventh_index)

            # time_str = line[third_index + len('time['):forth_index].strip()
            action = line[fifth_index + len(' action:'):sixth_index].strip()
            timestamp = float(line[seventh_index + len('Timestamp '):last_index].strip())
            first_index = first_index + len('profile REQ_ID[')
            req_id = line[first_index:second_index]
            if req_id == "0":
                continue
            d_node = None
            p_node = None
            if 'NODE=D' in line:
                id0 = line.find('NODE=')
                id1 = line.find('.', id0)
                d_node = line[id0 + len('NODE='):id1]
            elif 'NODE=P' in line:
                id0 = line.find('NODE=')
                id1 = line.find('.', id0)
                p_node = line[id0 + len('NODE='):id1]

            # profile REQ_ID[8783506036407222224] action:d_7 Add need pullling sequence|waiting_pull_len=1.Timestamp 1757043118.0792732
            if 'Add need pullling sequence' in action:
                values = action.split('|')
                action = values[0]
                waiting_pull_len_info = values[1].split('=')
                # print(f"{action=}, {timestamp=}, {waiting_pull_len_info[0]=}, {waiting_pull_len_info[1]=}")
                if req_id not in req_datas:
                    req_datas[req_id] = {action: [timestamp], waiting_pull_len_info[0]: [waiting_pull_len_info[1]]}
                else:
                    req_datas[req_id].update({action: [timestamp]})
                    req_datas[req_id].update({waiting_pull_len_info[0]: [waiting_pull_len_info[1]]})
            # profile REQ_ID[2767562802710561749] action:p_4 tokennizer to sche|Seq len=6434.Timestamp 1757043130.860907
            elif 'p_4 tokennizer to sche' in action:
                values = action.split('|')
                action = values[0]
                seq_len_info = values[1].split('=')
                if req_id not in req_datas:
                    # {p_4 tokennizer to sche: [1757043130.860907], Seq len: 6434}
                    req_datas[req_id] = {action: [timestamp], seq_len_info[0]: [seq_len_info[1]]}
                else:
                    req_datas[req_id].update({action: [timestamp]})
                    req_datas[req_id].update({seq_len_info[0]: [int(seq_len_info[1])]})

            else:
                if action not in action_dict.keys():
                    continue
                if req_id not in req_datas:
                    req_datas[req_id] = {action: []}
                    req_datas[req_id][action].append(timestamp)
                else:
                    if action not in req_datas[req_id]:
                        req_datas[req_id].update({action: []})
                    req_datas[req_id][action].append(timestamp)
                    if d_node is not None:
                        req_datas[req_id].update({'DNode': [d_node]})
                    if p_node is not None:
                        req_datas[req_id].update({'PNode': [p_node]})
    print(f'parse and prefill action timestamp done')

    result = {}
    for req_id, data in req_datas.items():
        result[req_id] = {}
        for miss_key in action_dict.keys() - data.keys():
            result[req_id].update({miss_key: None})
        for action, time_list in data.items():
            result[req_id][action] = min(time_list)
    for req_id, data in result.items():
        if req_id in decode_token_nums.keys():
            result[req_id].update({"decode token number": decode_token_nums[req_id]})
        else:
            result[req_id].update({"decode token number": None})

    return step_data, step_data_decode, result


def build_single_spans(reqid, action_times):
    action_first_ts = []
    for action in action_dict:
        if action in action_times:
            action_first_ts.append(action_times[action])
        else:
            action_first_ts.append(None)
        # print(f"{i}: {action}, action_first_ts: {action_first_ts[i]}")

    action_key = list(action_dict.keys())
    small_spans = []
    for start, end, span_name in SMALL_SPANS:
        if action_first_ts[start] is not None and action_first_ts[end] is not None:
            small_spans.append({
                "start_idx": start,
                "end_idx": end,
                "start_time": action_first_ts[start],
                "end_time": action_first_ts[end],
                "span_name": span_name,
                "span_type": "small",
                "start_action": action_key[start],
                "end_action": action_key[end]
            })
    
    extra_spans = []
    for start, end, span_name, span_type in EXTRA_SPANS:
        if action_first_ts[start] is not None and action_first_ts[end] is not None:
            extra_spans.append({
                "start_idx": start,
                "end_idx": end,
                "start_time": action_first_ts[start],
                "end_time": action_first_ts[end],
                "span_name": span_name,
                "span_type": span_type,
                "start_action": action_key[start],
                "end_action": action_key[end]
            })

    span_objs = []
    all_spans = extra_spans + small_spans
    all_spans.sort(key = lambda x:(x["start_idx"], -x["end_idx"]))
    for i, s in enumerate(all_spans):
        s["span_id"] = str(uuid.uuid4())
        s["children"] = []
        span_objs.append(s)
    name2idx = {s["span_name"]: i for i, s in enumerate(span_objs)}
    idxmap = {(s.get("start_idx"), s.get("end_idx")): i for i, s in enumerate(span_objs)}
    idxmap_backup = idxmap

    parent_of_span = [None] * len(span_objs)
    
    for span in span_objs:
        start_idx = span["start_idx"]
        end_idx = span["end_idx"]
        span_id = span["span_id"]
        candidates = []
        for parent_candidate in span_objs:
            parent_start = parent_candidate["start_idx"]
            parent_end = parent_candidate["end_idx"]
            parent_id = parent_candidate["span_id"]
            if span_id == parent_id:
                continue
            if (parent_start <= start_idx and end_idx <= parent_end):
                parent_duration = parent_end - parent_start
                candidates.append((parent_duration, parent_id))
            if candidates:
                candidates.sort()
                parent_of_span[idxmap_backup[(start_idx, end_idx)]] = candidates[0][1]

    spans_out = []
    for i, s in enumerate(span_objs):
        tags = [
            {"key": "RequestID", "type": "string", "value": reqid},
            {"key": "start_time", "type": "float", "value": s["start_time"]},
            {"key": "end_time", "type": "float", "value": s["end_time"]},
            {"key": "start_action", "type": "string", "value": s["start_action"]},
            {"key": "end_action", "type": "string", "value": s["end_action"]}
        ]
        span_json = {
            "traceID": "TBD",
            "spanID": str(s["span_id"]),
            "operationName": s["span_name"],
            "references": [],
            "startTime": int(s["start_time"] * 1e6),
            "duration": int((s["end_time"] - s["start_time"]) * 1e6),
            "tags": tags,
            "processID": s["span_name"],
            "logs": [
                {"timestamp": int(s["start_time"] * 1e6), "fields": []}
            ]
        }
        if parent_of_span[i] is not None:
            span_json["references"].append({
                "refType": "CHILD_OF",
                "traceID": "TBD",
                "spanID": str(parent_of_span[i])
            })
        spans_out.append(span_json)
    return spans_out

def build_jaeger_trace(reqid, spans):
    trace_id = uuid.uuid4().hex
    for s in spans:
        s["traceID"] = trace_id
        for ref in s["references"]:
            ref["traceID"] = trace_id
    processes = {}
    for s in spans:
        svc = s["processID"]
        if svc not in processes:
            processes[svc] = {
                "serviceName": svc,
                "tags": [
                    {"key": "service", "type": "string", "value": svc}
                ]
            }
    return {
        "traceID": trace_id,
        "spans": spans,
        "processes": processes
    }

def main(log_dir, output_json):
    step_data, step_data_decode, result = parse_files(log_dir)
    # req_action_times, req_roles = parse_log(log_dir)
    traces = []
    for reqid in result:
        spans = build_single_spans(reqid, result[reqid])
        if spans:
            traces.append(build_jaeger_trace(reqid, spans))
    jaeger_data = {"data": traces}
    with open(output_json, "w") as f:
        json.dump(jaeger_data, f, indent=2)
    print(f"Done. Output to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python log_to_jaeger.py /path/to/log_dir trace.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])