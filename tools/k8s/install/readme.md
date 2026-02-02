# 部署配置与操作步骤
## 一、环境配置
### 1. 配置网络代理
```bash
# 插入相应代理地址
export http_proxy=""
export https_proxy=""
export HTTP_PROXY="http_proxy"
export HTTPS_PROXY="https_proxy"
export no_proxy=""
export NO_PROXY="$no_proxy"
```

### 2. 配置DNS
编辑DNS配置文件：
```bash
vi /etc/resolv.conf
```
在文件中添加/修改以下内容：
```
search openstacklocal
nameserver 8.8.8.8
nameserver 8.8.4.4
......
```

## 二、部署k8s v1.31版本集群（主节点单节点）
执行以下步骤完成完整部署：
1. 一键部署k8s基础环境：
   ```bash
   bash K8s_Oneclick_Setup
   ```
2. 部署kuberinfer组件：
   ```bash
   cd kuberinfer ; bash kuberinfer_install
   ```
3. 部署Volcano组件：
   ```bash
   # 搭建新集群时：所有节点先执行基础安装
   cd Volcano ; bash volcano_install.sh
   # 在最终稳定的control-plane（主节点）上执行重启生效
   cd Volcano ; DO_RESTART=1 bash volcano_install.sh
   # 后续添加工作节点时：仅在新增节点执行基础安装即可
   # cd Volcano ; bash volcano_install.sh
   ```
4. 部署Device_Plugin组件：
   ```bash
   # 部署device_plugin/升级版本时（所有节点）
   cd Device_Plugin ; DO_RESTART=1 bash dp_install.sh
   # 修复dp,不想影响原有工作集群或者现有modelarts-device-plugin-xxxxx的running_pod时
   # cd Device_Plugin ; bash dp_install.sh
   ```

## 三、部署组件所需依赖包
### 1. kuberinfer组件依赖
```
modelarts-infers-operator-*-aarch64-*.tar
modelarts-infers-operator-*-*.tgz
```

### 2. Volcano组件依赖
```
vc-webhook-manager-v1.12.2.tar
vc-scheduler-v1.12.2.tar
vc-controller-manager-v1.12.2.tar
```

### 3. Device_Plugin组件依赖
```
modelarts-device-plugin-*.tgz
```

## 四、后续操作说明
### 1. 清除安装文件
若需要清除相应的安装文件，**从安装步骤的倒序依次运行各组件的uninstall脚本**。
- Volcano组件清理说明：
  ```bash
  # 节点级清理（默认）：仅影响当前执行节点
  cd Volcano ; bash volcano_uninstall.sh
  
  # 集群级清理（彻底）：影响整个集群，会完全删除Volcano所有资源
  cd Volcano ; DO_CLUSTER_CLEAN=1 bash volcano_uninstall.sh
  ```
- Device_Plugin组件清理说明：
  ```bash
  # 节点级清理（默认）：仅清理当前节点（适用于调试/修复containerd/重新安装DP）
  cd Device_Plugin ; bash dp_uninstall.sh
  
  # 集群级清理（彻底）：从整个集群中卸载Device Plugin
  cd Device_Plugin ; DO_CLUSTER_CLEAN=1 bash dp_uninstall.sh
  ```

### 2. 添加工作节点
若后续要添加工作节点到主节点集群，需先在工作节点上完成上述所有部署步骤（其中Volcano和Device_Plugin组件仅需执行基础安装脚本：`bash volcano_install.sh` 和 `bash dp_install.sh`），再参考文档`<control_plane_to_worker_node.md>`中的步骤依次操作。

## 五、蓝区部署版本包获取
### 1. 镜像版本获取
镜像版本参考：https://support.huaweicloud.com/bestpractice-modelarts/modelarts_ds_inferpd_s_908c11.html#section3
```bash
ctr -n k8s.io images pull swr.cn-east-4.myhuaweicloud.com/modelarts-rse/modelarts-device-plugin:7.3.0-20251010094010
ctr -n k8s.io images pull swr.cn-east-4.myhuaweicloud.com/atelier/modelarts-infers-operator:1.5.1.20250922143736
```

### 2. 版本包下载与处理
版本参考：https://support.huaweicloud.com/bestpractice-modelarts/modelarts_ds_inferpd_s_908c11.html#section0
下载来源：https://support.huawei.com/enterprise/zh/cloud-computing/modelarts-pid-23404305/software/

1. 找到版本包6.5.911-Compello，申请下载软件包（需要主动联系审批人加速审批）。
2. 下载后，找到目录：`AscendCloud-Solution-6.5.911-20251216203616/llm_infer/deploy/plugin/`。
3. 根据实际获取的镜像版本，打开`modelarts-device-plugin`、`modelarts-infers-operator`中的`values.yaml`，修改 `tag: "${image_tag}"` 为实际镜像版本。
4. 修改后，将两个文件夹分别打包成压缩包：
   - modelarts-device-plugin-{version}.tgz
   - modelarts-infers-operator-{version}.sh
   （待部署device_plugin、kubeinfer时使用）。

**示例**：如按照官方文档推荐版本，两个压缩包名称应为：
- modelarts-device-plugin-7.3.0-20251010094010.tgz
- modelarts-infers-operator-1.5.1.20250922143736.sh
