# FedTA → FedSMR-v2 技术架构文档

---

## 一、原方法 FedTA (CVPR 2025)

### 1.1 问题定义

联邦类增量持续学习 (Federated Class-Incremental Continual Learning, FCIL):
- K=5 个客户端，每个经历 T=5 个类增量任务
- **空间异质性**: 不同客户端类别分布不同 (私有类 + 公共类)
- **时间异质性**: 每个客户端按时间顺序经历任务，旧数据不可回放

### 1.2 实验配置

| 参数 | CIFAR-100 | ImageNet-R |
|------|-----------|------------|
| `client_num` | 5 | 5 |
| `task_num` | 5 | 5 |
| `private_class_num` | 15 | 40 |
| 公共类数 |Cp| | 25 | 0 |
| `global_epoch` | 5 | 5 |
| `local_epoch` | 30 | 20 |
| `batch_size` | 16 | 16 |
| `lr` | 0.001 | 0.001 |
| `surrogate_num` | 20 | 5 |
| `threshold` | 0.15 | 0.1 |
| ViT 骨干 | vit_base_patch16_224 (冻结) |

### 1.3 模型架构

```
预训练 ViT-B/16 (完全冻结)
  ├── Prompt Pool: 100 prompts × 10 tokens × 768 dim
  │     └── Key-query 相似度选择 top-k prompts
  ├── Tail Anchor (可训练, 约 253K 参数):
  │     ├── Key Pool:   (nb_classes, 768)
  │     ├── Anchor Pool: (nb_classes, 768)
  │     └── Classification Head: (1536, nb_classes)
  └── 输出: feat_mixed = concat(ViT_feat, anchor_feat) → Classifier
```

### 1.4 训练流程

```
for round 0..24:  # 5 tasks × 5 global_epochs
  task_id = round // global_epoch
  1. 客户端更新当前 task 数据
  2. 所有客户端本地训练 (Phase 1: Prompts + Phase 2: Anchor+Head)
  3. 服务器贪婪原型选择 (cosine similarity matrix → lowest avg sim)
  4. 知识蒸馏 Prompt 融合
  5. 分类头 FedAvg
  6. 分发全局模型 (原型 + 分类头 + Prompt) 给客户端
```

### 1.5 原始 Tail_Anchor (Hard Selection)

```python
similarity = x_norm @ key_norm.T         # (B, nb_class)
_, index = torch.topk(similarity, k=1)   # 选最相似的 1 个 anchor
anchor_feat = anchor_pool[index]          # (B, 768)
feat_mixed = concat(x, anchor_feat)       # (B, 1536)
logits = classification_head(feat_mixed)
```

### 1.6 损失函数

```
L = L_CE + λ_spatial·L_pull_off + λ_sikf·L_infonce
```

---

## 二、FedSMR-v2 改进方案

### 2.1 改进总览

| 改进 | 位置 | 解决的问题 |
|------|------|-----------|
| Residual Soft-Anchor | Tail_Anchor.forward | 保持类别判别性 + 平滑梯度 |
| L_route 路由损失 | Client_DF | Key 路由正确性 |
| Class-Aware Head Agg | Server_DF | 空间遗忘（每类只从见过客户端聚合） |
| Head Grad Mask | Client_DF.train | 未见过类不被污染 |
| K+A Temporal | Client_DF | 时间遗忘（Key+Anchor 双重保护） |
| Proto Head Replay | Client_DF.train | 旧类分类边界保护 |
| Seen-Only Diversity | Tail_Anchor | 避免未训练 anchor 被无意义推开 |

### 2.2 Residual Soft-Anchor

```
a_final = (1-γ)·a_hard + γ·a_soft

其中:
  hard_idx = argmax(similarity)
  a_hard = anchor_pool[hard_idx]
  a_soft = softmax(similarity/τ) @ anchor_pool
  γ = soft_anchor_ratio (CIFAR: 0.25, ImageNet-R: 0.15)
```

**实现**: `Tail_Anchor.py:forward()`

**效果**: 保留 hard anchor 的强类别指向性，同时通过 soft branch 获得平滑梯度。

### 2.3 监督路由损失 L_route

```
L_route = CE(softmax(similarity[:, seen]/τ_route), target)

只在 seen_classes 上计算，强制 x → key_y 对齐。
```

**实现**: `Client_DF._compute_route_loss()`

**参数**: `route_temperature=0.1`, `lambda_route=0.05`

### 2.4 Class-Aware Head Aggregation

```
对于 category c:
  valid_clients = [k | client_k 见过 class c]
  W_c^G = mean(W_c^k for k in valid_clients)
```

**实现**: `Server_DF.fed_avg_head()`

**效果**: ImageNet-R 上每个类只属于 1 个客户端，不再被另 4 个没见过的客户端平均稀释。

### 2.5 Head Gradient Mask

```python
# 从未见过 class c 的客户端，其 head 对 c 的梯度置零
model.head.weight.grad[unseen_mask] = 0.0
model.head.bias.grad[unseen_mask] = 0.0
```

**实现**: `Client_DF.train()`, Phase 2 backward 之后

### 2.6 Key+Anchor Temporal Stability

```
L_temporal = L_anchor_tmp + η·L_key_tmp

L_anchor_tmp = mean(conf_c · (1-cos(A_c^t, A_c^{t-1})))  for c in old_seen_classes
L_key_tmp    = mean(conf_c · (1-cos(K_c^t, K_c^{t-1})))   for c in old_seen_classes

conf_c = hard_usage_c / max(hard_usage)  # 基于 argmax 的使用频率
η = key_temporal_ratio (0.5)
```

**与 v1 的区别**:
- MSE → cosine loss (与推理时 cosine similarity 一致)
- Anchor only → Key + Anchor 双重保护
- Soft usage → Hard usage (argmax-based, 不受 soft attention 污染)
- All anchors → Only old seen classes

**实现**: `Client_DF._compute_msp_losses()` + `Tail_Anchor.anchor_hard_usage`

### 2.7 Prototype Head Replay

```
proto_feat = global_protos[old_classes]  # 1536 维 feat_mixed
proto_logits = classification_head(proto_feat)
L_proto = CE(proto_logits, old_classes)
```

**实现**: `Client_DF._compute_proto_replay_loss()`

**参数**: `lambda_proto=0.2` (CIFAR-100), `0.3` (ImageNet-R)

### 2.8 Seen-Only Diversity with Margin

```
seen_anchors = anchor_pool[seen_class_mask]
sim = normalize(seen_anchors) @ normalize(seen_anchors).T
L_div = max(0, sim[i≠j] - margin).mean()
```

**与 v1 的区别**:
- All anchors → Only seen classes
- No margin → margin=0.2 (允许一定相关性，只防止高度坍缩)

**实现**: `Tail_Anchor.anchor_diversity_loss()` + `seen_class_mask`

---

## 三、完整损失函数 (FedSMR-v2)

```
L_total = L_CE
        + 0.2·task_per_epoch·L_infonce     # SIKF 对比学习
        - 0.1·L_pull_off                    # 拉约束
        + λ_route·L_route                   # E2+: 路由监督
        + α_div·L_div(seen, margin)         # E4+: Seen-Only Diversity
        + α_tmp·(L_anchor_tmp + η·L_key_tmp) # E4+: Key+Anchor Temporal
        + λ_proto·L_proto_replay            # E5: Proto Replay
```

---

## 四、消融实验设计

| 实验 | Anchor | Route | Head | Temporal | Proto |
|------|:---:|:---:|:---:|:---:|:---:|
| E0 | Hard | | FedAvg | | |
| E1 | Residual Soft | | FedAvg | | |
| E2 | Residual Soft | ✓ | FedAvg | | |
| E3 | Residual Soft | ✓ | Class-Aware + Mask | | |
| E4 | Residual Soft | ✓ | Class-Aware + Mask | K+A cos | |
| E5 | Residual Soft | ✓ | Class-Aware + Mask | K+A cos | ✓ |

**预期关键差值**:
- E2−E1: 路由损失消除 Soft-Anchor routing ambiguity
- E3−E2: Class-Aware Head 修复 ImageNet-R spatial forgetting
- E4−E3: K+A Temporal 提升时间保留率
- E5−E4: Proto Replay 保护旧类决策边界

---

## 五、代码结构

```
Models/
  Tail_Anchor.py    — Residual Soft-Anchor, seen-only diversity, hard usage
  Client_DF.py      — Route loss, K+A temporal, proto replay, head grad mask
  Server_DF.py      — Class-aware head aggregation, v2 日志命名

config/
  cifar100_delay.py   — 12 个 v2 超参数 (γ, route, head, temporal, proto)
  imagenet_r_delay.py — 同上，数据集特定默认值

RUN_COMMANDS.md    — E0~E5 完整实验命令
```