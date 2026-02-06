## 0. 指定使用的计算卡编号

测试脚本支持通过环境变量指定运行时可见的计算卡编号。

在启动测试前，可通过设置 `ASCEND_RT_VISIBLE_DEVICES` 来限制测试进程可使用的卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
````
## 1. 测试启动方式

所有测试均通过根目录下的脚本启动：

```bash
bash run_tests.sh [OPTIONS]
````

当前支持以下三种运行模式：

### 1.1 运行全部测试

```bash
bash run_tests.sh
```

• UT用例


  • `tests/unit_tests`

  • `tests/integrated_tests`


---

### 1.2 仅运行单元测试

```bash
bash run_tests.sh --unit
```

• 执行目录：


  ```text
  tests/unit_tests/
  ```

---

### 1.3 仅运行集成测试

```bash
bash run_tests.sh --integrated
```

• 执行目录：


  ```text
  tests/integrated_tests/
  ```

---

## 2. 多进程并发说明

脚本支持通过 `-n` 参数指定测试并发数，例如：

```bash
bash run_tests.sh --unit -n 2
```

### 参数说明

| 参数         | 含义             |
| ---------- | -------------- |
| `-n <num>` | pytest 并发执行进程数 |

### 限制说明

• 当前最大支持并发数为 2

• 超过 2 的并发值可能导致不稳定行为


---

## 3. 覆盖率报告与日志输出

所有测试执行完成后，生成的覆盖率与相关报告统一输出tests/reports目录：

```text
tests/reports/coverage
```
相关日志输出tests/logs目录

---

## 4. 报告类型说明

`coverage` 目录下共包含 四类报告：

### 4.1 vLLM 覆盖率报告

```text
vllm_report/
```

• 覆盖范围：`infer_engines/vllm`

• 提供逐文件、逐行的 HTML 覆盖率高亮展示


---

### 4.2 Omni 覆盖率报告

```text
omni_report/
```

• 覆盖范围：`omni/` 相关代码

• 用于评估 omni 下模块测试覆盖情况


---

### 4.3 Patch 差异覆盖率报告（Diff Coverage）

```text
patch_report
```

• 基于以下输入生成：


  • 合并后的 Patch 文件（`combine.patch`）

  • 覆盖率数据（`coverage.xml`）

• 使用 `diff-cover` 工具生成

• 仅关注 Patch 引入或修改代码的覆盖情况

• 主要用于评估新增代码是否被测试覆盖


---

### 4.4 Proxy 覆盖率报告

```text
proxy_report/
```

• 展示 Proxy 相关代码的覆盖率情况

• test_proxy日志

---

## 5. 多容器并发说明

测试支持多容器并行执行。在宿主机启动多个容器分别运行切片，并汇总覆盖率结果。

注意：用例更新后，请重新生成耗时文件。

### 5.1 生成各用例耗时文件
```bash
cd /path/to/omniinfer/tests
bash generate_test_durations_json.sh
```


### 5.2 多容器并发使用说明

目前，通过人工分析调试、手动定制用例分配策略，得到了一套推荐的定制化并发策略，已预配置在以下脚本中，使用宿主机执行命令如下：
```bash
cd /path/to/omniinfer/tests
bash run_docker.sh <image_name> # 启动容器
bash concurrent_test_run_multi_docker.sh /path/to/omniinfer # 并行运行并合并覆盖率
```

容器与用例分配详情

| 容器      | 所需卡数      | 执行用例说明      |
| -------- | -------- | -------- |
| DT_1  | 1  | `omni-proxy`模块用例 <br> test_api_server.py, test_chunked_prefill_scheduler.py, test_proxy.py |
| DT_2  | 1  | `omni-proxy`模块用例 <br> test_proxy_group.py |
| DT_3  | 1  | `omni-proxy`模块用例 <br> test_proxy_reload.py |
| DT_4  | 1  | `omni-proxy`模块用例 <br> 除DT_1-3外的`omni-proxy`模块用例 |
| DT_5  | 2  | 需要双卡的用例 |
| DT_6  | 1  | 去除DT_1-5外的剩余用例 <br> 按时间切片的第一组 |
| DT_7  | 1  | 去除DT_1-5外的剩余用例 <br> 按时间切片的第二组 |

各容器测试日志保存于`/path/to/omniinfer/tests/install_logs`，汇总覆盖率报告保存于`/path/to/omniinfer/tests/report/coverage`。

如需修改各容器用例分配情况，请修改`run_docker.sh`、`concurrent_test_run_multi_docker.sh`及`multi_docker_collect_coverage.sh`中的相关配置。


## 6. 使用示例

```bash
# 运行全部测试，2 并发
bash run_tests.sh -n 2

# 仅运行单元测试
bash run_tests.sh --unit

# 仅运行集成测试，2 并发
bash run_tests.sh --integrated -n 2

# 使用多容器并发测试
bash run_docker.sh <image_name> && bash concurrent_test_run_multi_docker.sh /path/to/omniinfer

```

---