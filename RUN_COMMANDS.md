# FedSMR 实验运行命令

本文档提供 **FedSMR (Federated Soft Memory Retrieval)** 论文的完整实验命令。

---

## 实验方案总览

```
基线: 原始 FedTA（Hard Selection）
  │
  ├── A: Soft-Anchor（软锚选择）
  │
  └── A+B: Soft-Anchor + MSP 三重约束（FedSMR 核心）
            │
            ├── 适应性多样性约束（anchor_diversity）
            ├── 自适应一致性约束（coherence，按公共类占比缩放）
            └── 置信度加权时序稳定性（temporal，按 anchor 使用频率加权）
```

| 编号 | 配置 | 说明 |
|------|------|------|
| 0 | 基线 | 原始 FedTA（Hard Anchor + Hard Prompt） |
| A | Soft-Anchor | 余弦相似度 + softmax 加权所有 anchor |
| A+B | + aMSP | 自适应三层正则化（多样性 + 一致性 + 时序稳定性） |

---

## 通用参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | vit_base_patch16_224 | ViT-B/16 预训练模型 |
| `--batch-size` | 16 | 批次大小（所有实验统一） |
| `--data-path` | local_datasets/ | 数据集路径 |
| `--client_num` | 5 | 客户端数量 |
| `--task_num` | 5 | 每个客户端任务数 |
| `--private_class_num` | 15 | 每客户端私有类别数 |
| `--global_epoch` | 5 | 全局通信轮数 |
| `--local_epoch` | 30 | 本地训练轮数 |
| `--surrogate_num` | 20 | 服务器代理数据每类样本数 |
| `--seed` | 42 | 随机种子（建议跑 3 个 seed：42, 123, 2024） |

---

## 一、CIFAR-100 数据集

### 0. 基线 — 原始 FedTA（Hard Selection）

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_baseline \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=False \
  --use_soft_prompt=False \
  --use_msp=False
```

### A. Soft-Anchor（FedA）

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_soft_anchor \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=False
```

### A+B. Soft-Anchor + MSP 三重约束（FedSMR 核心）

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1
```

---

## 二、ImageNet-R 数据集

### 0. 基线 — 原始 FedTA（Hard Selection）

```bash
python main.py imagenet_r_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/imagenet_r_baseline \
  --data_name ImageNet-R \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=False \
  --use_soft_prompt=False \
  --use_msp=False
```

### A. Soft-Anchor（FedA）

```bash
python main.py imagenet_r_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/imagenet_r_soft_anchor \
  --data_name ImageNet-R \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=False
```

### A+B. Soft-Anchor + MSP 三重约束（FedSMR 核心）

```bash
python main.py imagenet_r_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/imagenet_r_fedsmr \
  --data_name ImageNet-R \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1
```

---

## 三、消融实验（MSP 三组件拆分）

> 仅需在 CIFAR-100 上跑，验证 MSP 三个正则化项各自的贡献。

### 去掉 Diversity Loss

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_ablation_no_div \
  --data_name cifar100 \
  --client_num 5 --task_num 5 --global_epoch 5 --local_epoch 30 \
  --use_soft_anchor=True --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=True \
  --msp_diversity_coeff=0.0 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1
```

### 去掉 Coherence Loss

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_ablation_no_coh \
  --data_name cifar100 \
  --client_num 5 --task_num 5 --global_epoch 5 --local_epoch 30 \
  --use_soft_anchor=True --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.0 \
  --msp_temporal_coeff=0.1
```

### 去掉 Temporal Stability Loss

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_ablation_no_tmp \
  --data_name cifar100 \
  --client_num 5 --task_num 5 --global_epoch 5 --local_epoch 30 \
  --use_soft_anchor=True --soft_temperature=0.17 \
  --use_soft_prompt=False \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.0
```

---

## 四、参数速查

### 核心模块开关

| 模块 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Soft-Anchor | `--use_soft_anchor` | False | 余弦相似度 + softmax 加权所有 anchor |
| Soft-Anchor 温度 | `--soft_temperature` | 0.1 | 越小越接近 hard，CIFAR-100 用 0.17 |
| MSP 总开关 | `--use_msp` | False | 三层自适应正则化 |
| MSP 多样性 | `--msp_diversity_coeff` | 0.1 | Intra-Pool Diversity 权重 |
| MSP 一致性 | `--msp_coherence_coeff` | 0.1 | Cross-Pool Coherence 权重（自适应缩放） |
| MSP 时序 | `--msp_temporal_coeff` | 0.1 | Temporal Stability 权重（置信度加权） |

### 自适应行为

| 机制 | CIFAR-100 (有公共类) | ImageNet-R (无公共类) |
|------|---------------------|---------------------|
| Coherence 开关 | 启用（|Cp\|=25 > 0） | 禁用（|Cp\|=0） |
| Temporal 加权 | 高频 anchor 强约束 | 全域低使用频率，弱约束 |

### 暂不使用的参数

| 参数 | 说明 |
|------|------|
| `--use_soft_prompt` | Soft Prompt Retrieval，效果不佳，不启用 |
| `--use_sparse_softmax` | Top-K 稀疏 Softmax，可选增强 |
| `--temperature_anneal` | 温度退火，可选增强 |
| `--use_fed_smr_aggregate` | 使用频率联邦加权聚合，可选增强 |

---

## 五、损失函数结构

```
L_total = L_CE
        + λ_spatial · (-0.1 · L_pull_off)              # 空间分支（Tail Anchor）
        + λ_sikf · (0.2 · t · L_infonce)               # SIKF 对比学习分支
        + α_div · L_anchor_div                          # MSP: Anchor 多样性
        + α_coh · r_pub · L_coherence                   # MSP: 自适应一致性
        + α_tmp · Σ conf_i · MSE(anchor_i^t, anchor_i^{t-1})  # MSP: 置信度加权时序
```

其中：
- **L_anchor_div** = `max(0, cos(anchor_i, anchor_j))` 的均值，约束 anchor 间保持多样性
- **L_coherence** = `max(0, 0.3-cos) + max(0, cos-0.7)`，区间约束，仅在有关公共类时生效
- **r_pub** = `|Cp| / |C|`，公共类占比（CIFAR-100: 0.25, ImageNet-R: 0）自动缩放 coherence
- **L_temporal** = 置信度加权 MSE，`conf_i = usage_i / max(usage)`，使用频率高→强约束

---

## 六、建议执行顺序

```
第一优先级（核心对比）:
  ├── 0. 基线 (CIFAR-100 + ImageNet-R)
  ├── A. Soft-Anchor (CIFAR-100 + ImageNet-R)
  ├── A+B. FedSMR + aMSP (CIFAR-100 + ImageNet-R)
  └── 每个跑 3 个 seed (42, 123, 2024)

第二优先级（消融）:
  ├── MSP 三组件拆分 (CIFAR-100)
  └── 温度 τ 敏感性分析 (CIFAR-100: τ ∈ {0.05, 0.1, 0.17, 0.3, 0.5})

第三优先级（超参调优）:
  ├── msp_diversity_coeff ∈ {0.05, 0.1, 0.2}
  └── msp_temporal_coeff ∈ {0.05, 0.1, 0.2}
```

---

## 七、技术原理

### 7.1 Soft-Anchor（A）

```
# Hard Selection（原始 FedTA）
anchor = anchor_pool[argmax(similarity)]

# Soft-Anchor
weights = softmax(cosine_similarity(query, anchors) / temperature)
anchor = Σ(weights[i] · anchor_pool[i])
```

### 7.2 自适应 MSP 三重约束（B）

```
L_ms = L_diversity + r_pub · L_coherence + Σ conf_i · L_temporal_i

L_diversity  = max(0, cos(anchor_i, anchor_j))      # 只惩罚正相似度
L_coherence  = max(0, l-cos) + max(0, cos-u)         # 区间 [l, u] 内不惩罚
r_pub         = |Cp| / |C|                             # 自适应缩放（无公共类→禁能）
conf_i        = usage_i / max(usage)                   # 置信度加权（高频→强约束）
L_temporal_i = (anchor_i^t - anchor_i^{t-1})²         # 按 anchor 置信度加权
```

- **CIFAR-100**（有公共类、数据充足）：三项全效，A+B > A
- **ImageNet-R**（无公共类、数据稀疏）：coherence 自动禁用，temporal 自动减弱