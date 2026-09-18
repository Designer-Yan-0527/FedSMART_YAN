# FedSMR-v2 实验运行命令

---

## 实验方案总览

```
基线: 原始 FedTA（Hard Anchor + FedAvg Head）
  │
  ├── E1: Residual Soft-Anchor  (γ=0.25)
  │     a = (1-γ)·a_hard + γ·a_soft
  │
  ├── E2: E1 + Anchor Routing Loss  (L_route)
  │     监督 Key 路由到正确类别
  │
  ├── E3: E2 + Class-Aware Head Aggregation + Head Grad Mask
  │     每类只从见过的客户端聚合，未见过类梯度置零
  │
  ├── E4: E3 + Key&Anchor Temporal Stability (cosine, seen-only, hard usage)
  │     同时保护 Key 和 Anchor 的时序稳定性
  │
  └── E5: E4 + Prototype Head Replay  (完整 FedSMR-v2)
        全局原型重放保护旧类分类边界
```

| 实验 | Residual Soft | L_route | Class-Aware Head | K+A Temporal | Proto Replay |
|------|:---:|:---:|:---:|:---:|:---:|
| E0 (Baseline) | | | | | |
| E1 | ✓ | | | | |
| E2 | ✓ | ✓ | | | |
| E3 | ✓ | ✓ | ✓ | | |
| E4 | ✓ | ✓ | ✓ | ✓ | |
| E5 | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 通用参数

| 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|-----------|------------|------|
| `--model` | vit_base_patch16_224 | 同 | ViT-B/16 |
| `--batch-size` | 16 | 16 | 统一 |
| `--client_num` | 5 | 5 | |
| `--task_num` | 5 | 5 | |
| `--private_class_num` | 15 | 40 | |
| `--global_epoch` | 5 | 5 | |
| `--local_epoch` | 30 | 30 | 同 CIFAR |
| `--surrogate_num` | 20 | 5 | 服务器代理数据每类 |
| `--seed` | 42 | 42 | 建议 3 seeds: 42, 123, 2024 |

---

## 一、CIFAR-100

### E0. 基线 — 原始 FedTA

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e0 \
  --use_soft_anchor=False --use_msp=False
```

### E1. Residual Soft-Anchor

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_msp=False
```

### E2. E1 + Anchor Routing Loss

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False
```

### E3. E2 + Class-Aware Head + Head Grad Mask

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=False
```

### E4. E3 + Key&Anchor Temporal Stability

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0
```

### E5. E4 + Prototype Head Replay（完整版）

```bash
python main.py cifar100_delay --batch-size 16 --data-path local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.2
```

---

## 二、ImageNet-R

### E0. 基线

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e0 \
  --use_soft_anchor=False --use_msp=False
```

### E1. Residual Soft-Anchor

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_msp=False
```

### E2. E1 + Anchor Routing Loss

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False
```

### E3. E2 + Class-Aware Head + Head Grad Mask

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=False
```

### E4. E3 + Key&Anchor Temporal Stability

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0
```

### E5. E4 + Prototype Head Replay（完整版）

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_class_aware_head_agg=True --use_head_grad_mask=True \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.3
```

---

## 三、参数速查

### 核心模块开关

| 模块 | 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|------|-----------|------------|------|
| Residual Soft-Anchor | `--use_soft_anchor` | True | True | 残差软锚 |
| 温度 | `--soft_temperature` | 0.17 | 0.17 | Softmax τ |
| 残差比 | `--soft_anchor_ratio` | 0.25 | 0.15 | γ, 0=纯hard |
| 路由损失 | `--use_route_loss` | True | True | L_route |
| 路由温度 | `--route_temperature` | 0.1 | 0.1 | |
| 路由权重 | `--lambda_route` | 0.05 | 0.05 | |
| Class-Aware Head | `--use_class_aware_head_agg` | True | True | 每类别感知 |
| Head Grad Mask | `--use_head_grad_mask` | True | True | 未见过类梯度置零 |
| MSP 总开关 | `--use_msp` | True | True | |
| Diversity 系数 | `--msp_diversity_coeff` | 0.03 | 0.03 | Seen-only |
| Diversity Margin | `--diversity_margin` | 0.2 | 0.2 | cos>margin 才惩罚 |
| Temporal 系数 | `--msp_temporal_coeff` | 0.1 | 0.15 | Key+Anchor |
| Key 权重 | `--key_temporal_ratio` | 0.5 | 0.5 | η |
| Coherence | `--msp_coherence_coeff` | 0.0 | 0.0 | 已禁用 |
| Proto Replay | `--use_proto_replay` | True | True | |
| Proto 权重 | `--lambda_proto` | 0.2 | 0.3 | |

---

## 四、损失函数 (FedSMR-v2)

```
L = L_CE
  + λ_sikf · L_infonce
  + λ_route · L_route              # E2+: 监督 Key 路由
  + α_div · L_div(seen, margin)    # E4+: Seen-Only Diversity
  + α_tmp · (L_anchor_tmp + η·L_key_tmp)  # E4+: Key+Anchor Temporal
  + λ_proto · L_proto_replay       # E5: Prototype Head Replay
```

---

## 五、建议执行顺序

```bash
# 第一优先级: E0→E1→E2 (CIFAR-100 优先)
# 验证 Residual Soft-Anchor 和 Route Loss 的效果

# 第二优先级: E3 (ImageNet-R 优先)
# 验证 Class-Aware Head 对 spatial forgetting 的效果

# 第三优先级: E4→E5 (两张数据集)
# 验证完整 FedSMR-v2
```