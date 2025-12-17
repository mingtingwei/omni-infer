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
bash run_tests.sh --all
```

* UT用例

  * `tests/unit_tests`
  * `tests/integrated_tests`

---

### 1.2 仅运行单元测试

```bash
bash run_tests.sh --unit
```

* 执行目录：

  ```text
  tests/unit_tests/
  ```

---

### 1.3 仅运行集成测试

```bash
bash run_tests.sh --integrated
```

* 执行目录：

  ```text
  tests/integrated_tests/
  ```

---

## 2. 并发参数说明

脚本支持通过 `-n` 参数指定测试并发数，例如：

```bash
bash run_tests.sh --unit -n 2
```

### 参数说明

| 参数         | 含义             |
| ---------- | -------------- |
| `-n <num>` | pytest 并发执行进程数 |

### 限制说明

* 当前**最大支持并发数为 2**
* 超过 2 的并发值可能导致不稳定行为

---

## 3. 覆盖率与报告输出

所有测试执行完成后，生成的覆盖率与相关报告统一输出至项目根目录：

```text
coverage_html/
```

---

## 4. 报告类型说明

`coverage_html` 目录下共包含 **四类报告**：

### 4.1 vLLM 覆盖率报告

```text
coverage_html/vllm/
```

* 覆盖范围：`infer_engines/vllm`
* 提供逐文件、逐行的 HTML 覆盖率高亮展示

---

### 4.2 Omni 覆盖率报告

```text
coverage_html/omni/
```

* 覆盖范围：`omni/` 相关代码
* 用于评估 omni 下模块测试覆盖情况

---

### 4.3 Patch 差异覆盖率报告（Diff Coverage）

```text
coverage_html/diff.html
```

* 基于以下输入生成：

  * 合并后的 Patch 文件（`combined.patch`）
  * 覆盖率数据（`coverage.xml`）
* 使用 `diff-cover` 工具生成
* 仅关注 **Patch 引入或修改代码的覆盖情况**
* 主要用于评估新增代码是否被测试覆盖

---

### 4.4 Proxy 覆盖率报告

```text
coverage_html/proxy_report/
```

* 展示 Proxy 相关代码的覆盖率情况
* test_proxy日志
---

## 5. 使用示例

```bash
# 运行全部测试，2 并发
bash run_tests.sh --all -n 2

# 仅运行单元测试
bash run_tests.sh --unit

# 仅运行集成测试，2 并发
bash run_tests.sh --integrated -n 2
```

---
