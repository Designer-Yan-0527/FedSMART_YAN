# FedSTAR 实验运行命令

> 架构：ViT-B/16 (frozen) + L2P Prompt + Tail_Anchor (anchor_pool + key) + per-task Head
> 
> per-task head 天然隔离跨任务遗忘，无需 head_grad_mask / class_aware_head_agg

---

## 实验方案总览

```
基线: FedTA（Hard Anchor + per-task Head）
  │
  ├── E1: Residual Soft-Anchor (γ=0.25)
  │     a = (1-γ)·a_hard + γ·a_soft
  │
  ├── E2: E1 + Anchor Routing Loss (L_route)
  │     监督 Key 路由到正确类别
  │
  ├── E3: E2 + MSP (Diversity + Key&Anchor Temporal)
  │     Seen-Only Diversity + 跨轮稳定性
  │
  ├── E4: E3 + Prototype Head Replay
  │     全局原型重放保护旧类分类边界
  │
  └── E5: E4 + Seen Routing (完整 FedSTAR)
        Anchor 路由限制在 seen classes
```

| 实验 | Soft Anchor | Route Loss | MSP | Proto Replay | Seen Routing |
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
| `--batch-size` | 16 | 16 | |
| `--client_num` | 5 | 5 | |
| `--task_num` | 5 | 5 | |
| `--private_class_num` | 15 | 40 | |
| `--global_epoch` | 5 | 5 | |
| `--local_epoch` | 30 | 30 | |
| `--surrogate_num` | 20 | 5 | 服务器代理数据每类 |
| `--seed` | 42 | 42 | 建议 3 seeds: 42, 123, 2024 |

---

## 一、CIFAR-100

### E0. 基线 — FedTA (Hard Anchor + per-task Head)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e0 \
  --use_soft_anchor=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E1. Residual Soft-Anchor

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_msp=False --use_proto_replay=False --use_seen_routing=False
```

### E2. E1 + Anchor Routing Loss

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False --use_proto_replay=False --use_seen_routing=False
```

### E3. E2 + MSP (Diversity + Key&Anchor Temporal)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=False --use_seen_routing=False
```

### E4. E3 + Prototype Head Replay

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.2 \
  --use_seen_routing=False
```

### E5. E4 + Seen Routing（完整 FedSTAR）

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.2 \
  --use_seen_routing=True
```

---

## 二、ImageNet-R

### E0. 基线

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e0 \
  --use_soft_anchor=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E1. Residual Soft-Anchor

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_msp=False --use_proto_replay=False --use_seen_routing=False
```

### E2. E1 + Anchor Routing Loss

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False --use_proto_replay=False --use_seen_routing=False
```

### E3. E2 + MSP (Diversity + Key&Anchor Temporal)

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=False --use_seen_routing=False
```

### E4. E3 + Prototype Head Replay

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.3 \
  --use_seen_routing=False
```

### E5. E4 + Seen Routing（完整 FedSTAR）

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.3 \
  --use_seen_routing=True
```

---

## 三、参数速查

| 模块 | 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|------|-----------|------------|------|
| Residual Soft-Anchor | `--use_soft_anchor` | True | True | |
| 温度 | `--soft_temperature` | 0.17 | 0.17 | Softmax τ |
| 残差比 | `--soft_anchor_ratio` | 0.25 | 0.15 | γ |
| 路由损失 | `--use_route_loss` | True | True | L_route |
| 路由温度 | `--route_temperature` | 0.1 | 0.1 | |
| 路由权重 | `--lambda_route` | 0.05 | 0.05 | |
| MSP | `--use_msp` | True | True | |
| Diversity | `--msp_diversity_coeff` | 0.03 | 0.03 | |
| Diversity Margin | `--diversity_margin` | 0.2 | 0.2 | |
| Temporal | `--msp_temporal_coeff` | 0.1 | 0.15 | |
| Key 权重 | `--key_temporal_ratio` | 0.5 | 0.5 | η |
| Coherence | `--msp_coherence_coeff` | 0.0 | 0.0 | 已禁用 |
| Proto Replay | `--use_proto_replay` | True | True | |
| Proto 权重 | `--lambda_proto` | 0.2 | 0.3 | |
| Seen Routing | `--use_seen_routing` | True | True | E5 only |

---

## 四、损失函数

```
L = L_CE
  + λ_infonce · L_infonce
  + λ_route · L_route              # E2+: 监督 Key 路由
  + α_div · L_div(seen, margin)    # E3+: Seen-Only Diversity
  + α_tmp · (L_anchor_tmp + η·L_key_tmp)  # E3+: Key+Anchor Temporal
  + λ_proto · L_proto_replay       # E4+: Prototype Head Replay
```

---

## 五、多 seed 运行

```bash
for seed in 42 123 2024; do
  for exp in e0 e1 e2 e3 e4 e5; do
    python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
      --data_name cifar100 --output_dir ./output/cifar100_${exp}_seed${seed} \
      --seed ${seed} \
      ...  # 按上面的 E0-E5 参数填
  done
done
```

---

## 六、已移除的参数

| 参数 | 原因 |
|------|------|
| `--use_head_grad_mask` | per-task head 物理隔离，不需要梯度掩码 |
| `--use_class_aware_head_agg` | per-task head 各自独立聚合，不需要类别感知 |
| `--use_soft_prompt / --temperature_anneal / --use_sparse_softmax` | 已废弃 |