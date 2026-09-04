# FedSMR 运行命令

本文档提供 **FedSMR (Federated Soft Memory Retrieval)** 论文方案的运行命令。

## 核心改进（FedSMR）

| 改进 | 说明 | 默认值 |
|------|------|--------|
| **Unified Soft Memory Retrieval** | Prompt 和 Anchor 均使用 softmax 加权选择 | 启用 |
| **Memory Structure Preservation (MSP)** | 三层正则化：多样性 + 一致性 + 时序稳定性 | 需显式启用 |
| **Temperature Annealing** | 训练中温度从高到低余弦退火 | 可选 |
| **Top-K Sparse Softmax** | 仅 top-k anchor 参与 softmax，减少噪声 | 可选 |
| **Usage-Weighted Aggregation** | 联邦聚合基于记忆使用频率加权 | 可选 |

## 通用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | vit_base_patch16_224 | ViT 模型架构 |
| `--batch-size` | 16 | 训练批次大小 |
| `--data-path` | local_datasets/ | 数据集路径 |
| `--output_dir` | ./output | 输出目录 |
| `--client_num` | 5 | 客户端数量 |
| `--task_num` | 5 | 任务数量 |
| `--global_epoch` | 5 | 全局轮次数 |
| `--local_epoch` | 30 | 本地轮次数 |

## 一、CIFAR-100 数据集

### 1.1 Baseline（原 FedTA）

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

### 1.2 FedA（Soft Anchor）

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
  --soft_temperature=0.17
```

### 1.3 FedAP（Soft Anchor + Soft Prompt）效果不好不开启

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
  --soft_temperature=0.1 \
  --use_soft_prompt=False \
  --prompt_temperature=0.1
```

### 1.4 FedSMR（默认配置：Soft Anchor + MSP）

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

### 1.5 FedSMR + Top-K Sparse Softmax

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr_sparse \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.1 \
  --use_soft_prompt=False \
  --use_sparse_softmax=True \
  --top_k_anchor=5 \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1
```

### 1.5.1 FedSMR + Top-K Sparse Softmax + Temperature Annealing

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr_sparse \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --soft_temperature=0.1 \
  --use_soft_prompt=False \
  --use_sparse_softmax=True \
  --use_msp=True \
  --msp_diversity_coeff=0.1 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1 \
  --temperature_anneal=True
```

### 1.6 FedSMR + Temperature Annealing

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr_anneal \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --temperature_anneal=True \
  --use_soft_prompt=False \
  --use_msp=True
```

### 1.7 FedSMR + Usage-Weighted Aggregation

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr_agg \
  --data_name cifar100 \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --use_soft_prompt=False \
  --use_msp=True \
  --use_fed_smr_aggregate=True
```

### 1.8 FedSMR + EMA Prototype + Quality Weight + Fisher Temporal（方案1+2 组合）

```bash
python main.py cifar100_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/cifar100_fedsmr_v2 \
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
  --msp_temporal_coeff=0.1 \
  --use_ema_proto=True \
  --ema_momentum=0.9 \
  --use_quality_weight=True \
  --quality_lambda=0.1 \
  --use_fisher_temporal=True \
  --fisher_ema_decay=0.9
```

## 二、ImageNet-R 数据集

### 2.1 Baseline（原 FedTA）

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

### 2.2 FedA（Soft Anchor）

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
  --soft_temperature=0.17
```

### 2.3 FedAP（Soft Anchor + Soft Prompt）

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
  --soft_temperature=0.1 \
  --use_soft_prompt=False \
  --prompt_temperature=0.1
```

### 2.4 FedSMR（默认配置）

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
  --prompt_temperature=0.1 \
  --use_msp=True \
  --msp_diversity_coeff=0.11 \
  --msp_coherence_coeff=0.1 \
  --msp_temporal_coeff=0.1
```

### 2.5 FedSMR + Top-K Sparse Softmax

```bash
python main.py imagenet_r_delay \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path local_datasets/ \
  --output_dir ./output/imagenet_r_fedsmr_sparse \
  --data_name ImageNet-R \
  --client_num 5 \
  --task_num 5 \
  --global_epoch 5 \
  --local_epoch 30 \
  --use_soft_anchor=True \
  --use_soft_prompt=False \
  --use_sparse_softmax=True \
  --top_k_anchor=10 \
  --use_msp=True
```

## 三、参数说明

### 3.1 FedSMR 核心参数

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--use_soft_anchor` | True | **Soft Anchor Mixture**：cosine sim + softmax 加权所有 anchor |
| `--soft_temperature` | 0.1 | Anchor softmax 温度，越大越平滑，建议 [0.05, 0.5] |
| `--use_soft_prompt` | True | **Soft Prompt Retrieval**：prompt 侧 softmax 加权选择 |
| `--prompt_temperature` | 0.1 | Prompt softmax 温度，建议 [0.05, 0.5] |
| `--use_msp` | False | **MSP 三层正则化**：多样性 + 一致性 + 时序稳定性 |
| `--msp_diversity_coeff` | 0.1 | Intra-Pool Diversity 权重 |
| `--msp_coherence_coeff` | 0.1 | Cross-Pool Coherence 权重 |
| `--msp_temporal_coeff` | 0.1 | Temporal Stability 权重 |

### 3.2 FedSMR 可选增强

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--use_sparse_softmax` | False | 仅 top-k anchor 参与 softmax，减少噪声 |
| `--top_k_anchor` | None | Top-K 的 K 值，None = 使用全部 |
| `--temperature_anneal` | False | 温度余弦退火，训练中从高到低 |
| `--use_fed_smr_aggregate` | False | 基于使用频率的联邦加权聚合 |

### 3.3 EMA Prototype Update（方案1：原型平滑更新）

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--use_ema_proto` | False | **EMA 原型更新**：指数移动平均平滑全局原型，防止剧烈波动 |
| `--ema_momentum` | 0.9 | EMA 动量系数，越大越平滑，建议 [0.8, 0.99] |

### 3.4 Quality-Weighted Prototype Selection（方案1：质量加权选择）

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--use_quality_weight` | False | **原型质量加权**：样本量越大的原型权重越高，优先选择可靠原型 |
| `--quality_lambda` | 0.1 | 样本量对选择的影响强度，建议 [0.01, 0.5] |

### 3.5 Fisher-Weighted Temporal Stability（方案2：重要记忆保护）

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--use_fisher_temporal` | False | **Fisher 加权时序稳定性**：重要 anchor 变化时惩罚更重，保护关键记忆 |
| `--fisher_ema_decay` | 0.9 | Fisher 信息累积的 EMA 衰减系数，建议 [0.8, 0.99] |

### 3.6 原始 FedTA 参数

| 开关 | 默认值 | 效果 |
|------|--------|------|
| `--lambda_spatial` | 1.0 | Tail Anchor pull-off 约束强度 |
| `--lambda_sikf` | 1.0 | InfoNCE 对比学习强度 |

## 四、FedSMR 损失函数结构

```
loss = CE_loss
     + lambda_spatial * (-0.1 * pull_off)                     # 空间分支
     + lambda_sikf * (0.2 * task_per_global_epoch * loss_infonce)  # SIKF 分支
     + msp_diversity_coeff * L_anchor_div                     # MSP: Anchor 多样性
     + msp_coherence_coeff * L_coherence                      # MSP: Prompt-Anchor 一致性
     + msp_temporal_coeff * L_temporal                        # MSP: 时序稳定性
```

其中：
- **L_anchor_div** = abs(cosine(anchor_i, anchor_j)) 的均值，约束 anchor 间保持多样性
- **L_coherence** = clamp(cos(F_prompt, F_anchor))，约束 prompt 与 anchor 特征保持合理范围
- **L_temporal** = MSE(anchor_t, anchor_{t-1})，约束 anchor 跨轮次平滑更新

## 五、消融实验建议

### 5.1 推荐消融顺序

1. **Step 1**：Baseline（`--use_soft_anchor=False --use_soft_prompt=False --use_msp=False`）
2. **Step 2**：+ Soft Anchor（`--use_soft_anchor=True`）
3. **Step 3**：+ Soft Prompt（`--use_soft_prompt=True`）
4. **Step 4**：+ MSP（`--use_msp=True`）
5. **Step 5**：+ Sparse Softmax（`--use_sparse_softmax=True --top_k_anchor=10`）
6. **Step 6**：+ Temperature Annealing（`--temperature_anneal=True`）
7. **Step 7**：+ Usage-Weighted Aggregation（`--use_fed_smr_aggregate=True`）

### 5.2 超参调整建议

| 参数 | 建议范围 | 备注                                      |
|------|----------|-----------------------------------------|
| `--soft_temperature` | [0.05, 0.5] | 越小越接近 hard selection，越大越平滑，目前测试0.17效果最好 |
| `--prompt_temperature` | [0.05, 0.5] | 同上，效果不好不启用                              |
| `--msp_diversity_coeff` | [0.05, 0.5] | 越大越强调 anchor 多样性                        |
| `--msp_coherence_coeff` | [0.05, 0.5] | 越大越强调 prompt-anchor 一致性                 |
| `--msp_temporal_coeff` | [0.05, 0.5] | 越大越强调跨轮次稳定性                             |
| `--top_k_anchor` | [5, 20] | 取决于类别数量，建议 nb_class/2                   |

### 5.3 预期效果

- **Accuracy ↑**：Unified Soft Memory Retrieval 提供更稳定、更丰富的梯度信号
- **Catastrophic Forgetting ↓**：MSP 三层正则化约束记忆结构保持
- **Communication Efficiency ↑**：Usage-Weighted Aggregation 高效传递共识信息

## 六、FedSMR 技术原理

### 6.1 Unified Soft Memory Retrieval

```
# Hard Selection（原始 FedTA）
anchor = anchor_pool[topk(similarity, k=1)]  # 只选一个
prompt = prompt_pool[topk(similarity, k=top_k)]  # 只选 top_k 个

# Soft Memory Retrieval（FedSMR）
weights_anchor = softmax(similarity / temperature_anchor)
anchor = sum(weights_anchor[i] * anchor_pool[i])  # 加权所有 anchor

weights_prompt = softmax(similarity / temperature_prompt)
prompt = sum(weights_prompt[i] * prompt_pool[i])  # 加权所有 prompt
```

### 6.2 Memory Structure Preservation (MSP)

```
L_ms = L_diversity + L_coherence + L_temporal

L_diversity = mean(abs(cos(anchor_i, anchor_j)))  # Intra-Pool
L_coherence = clamp(cos(F_prompt, F_anchor), 0.3, 0.7)  # Cross-Pool
L_temporal = MSE(anchor_t, anchor_{t-1})  # Temporal
```

### 6.3 Usage-Weighted Federation Aggregation

```
w_i = usage_i / sum(usage)  # 使用频率归一化权重
anchor_global = sum(w_i * anchor_i)  # 高频向量获得更高权重
```