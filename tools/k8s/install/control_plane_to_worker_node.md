# 工作节点加入Kubernetes集群步骤
## 1. 清理工作节点上的旧配置
首先，在工作节点上执行以下命令，清理之前的集群配置，确保工作节点能顺利加入集群：

```bash
# 清理 Kubernetes 配置
kubeadm reset --force

# 删除 Kubernetes 配置文件
rm -rf /etc/kubernetes
rm -rf /var/lib/kubelet/*
rm -rf /var/lib/etcd

# 如果安装了 CNI 插件（如 Calico、Weave 等），需要清理网络配置
rm -rf /etc/cni/net.d/*

# 删除 kubelet 配置文件
rm -f /etc/kubernetes/kubelet.conf

# 重启 kubelet 服务
systemctl restart kubelet
```

这些步骤会清理工作节点上的旧配置文件，并确保节点可以正常加入集群。

## 2. 生成加入令牌
在控制平面节点（主节点）上生成 `kubeadm join` 命令，用于将工作节点加入集群：

```bash
kubeadm token create --print-join-command
```

这条命令会输出一个 `kubeadm join` 命令，形如：

```bash
kubeadm join x.x.x.x:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>
```

请注意保存输出的令牌和哈希值。

## 3. 在工作节点上执行 kubeadm join 命令
在工作节点上执行从控制平面节点获得的 `kubeadm join` 命令：

```bash
kubeadm join x.x.x.x:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>
```

这将把工作节点加入到控制平面节点的 Kubernetes 集群中。

## 4. 验证工作节点是否成功加入集群
在控制平面节点上，检查集群状态，确认工作节点已经成功加入：

```bash
kubectl get nodes
kubectl get pods -n kube-system
```

你应该看到工作节点显示为 `Ready` 状态。

## 5. 完成并验证集群
在控制平面节点上再次确认集群状态：

```bash
kubectl get nodes
kubectl get pods -n kube-system
```

你应该看到所有节点都处于 `Ready` 状态，且集群中没有异常，继续进行后续操作。

## 配置工作节点Device Plugin（DP）
### 步骤1：为工作节点添加标签
在控制平面节点上为工作节点添加标签（注：`hostname` 填写你插入的相应工作节点的节点名字）：

```bash
kubectl label node <hostname> accelerator/huawei-npu=true
```

### 步骤2：验证插件部署
在控制平面节点上执行以下命令，查看主节点和工作节点的 Pod 状态，验证插件的部署：

```bash
kubectl get pods -n kube-system
```

应该看到对应的 `modelarts-device-plugin` Pod 数量从1个变为2个，示例如下：
```
kube-system      modelarts-device-plugin-fxxxx                                     1/1     Running     0          xxxxx
kube-system      modelarts-device-plugin-fxxxx                                     1/1     Running     0          xxxxx
```