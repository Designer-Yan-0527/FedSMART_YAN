"""
FedSMR: Tail_Anchor 模型模块 (Soft Memory Retrieval)

该模块实现了 FedSMR（Federated Soft Memory Retrieval）框架中的 Anchor Memory，
在尾部添加可学习的锚点参数来增强特征表示。

核心改进（FedSMR）：
- Soft Anchor Mixture: 使用余弦相似度 + softmax 进行软加权组合
- Top-K Sparse Softmax: 仅对 top-k 最相似 anchor 计算 softmax，减少噪声
- Temperature Annealing: 训练过程中温度从高到低退火
- Anchor Diversity Regularization: 保持 anchor 之间的多样性

核心功能：
- 动态软锚点选择机制
- 基于余弦相似度的特征匹配
- 拉约束损失计算
- Anchor 多样性正则化
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from Models.classification_head import Chead


class Tail_Anchor(nn.Module):
    """
    FedSMR Anchor Memory 模型

    在预训练模型的尾部添加可学习的锚点参数，通过 soft attention 权重
    组合锚点来增强特征表示，适用于联邦持续学习场景。

    FedSMR 新增：
    - soft_anchor=True: 使用 softmax 加权组合 anchor_pool
    - top_k_anchor: 仅对 top-k 最相似的 anchor 做 softmax（稀疏softmax）
    - temperature_anneal: 温度从高到低退火
    - anchor_diversity_loss(): 保持 anchor 多样性
    """

    def __init__(self, anchor_size, key_size, nb_class,
                 soft_anchor=True, soft_temperature=0.1,
                 top_k_anchor=None, temperature_anneal=False,
                 use_sparse_softmax=False, fisher_ema_decay=0.9):
        """
        初始化 Tail_Anchor 模型

        Args:
            anchor_size: 锚点大小
            key_size: 特征维度大小
            nb_class: 类别数量
            soft_anchor: 是否启用 Soft Anchor Mixture
            soft_temperature: softmax 的温度系数（越大越平滑）
            top_k_anchor: Top-K 稀疏 softmax 的 K 值，None 表示使用全部 anchor
            temperature_anneal: 是否启用温度退火
            use_sparse_softmax: 是否启用 Top-K 稀疏 softmax
            fisher_ema_decay: Fisher 信息累积的 EMA 衰减系数
        """
        super(Tail_Anchor, self).__init__()
        self.size = anchor_size
        self.key_size = key_size
        self.nb_class = nb_class

        # FedSMR 超参数
        self.soft_anchor = soft_anchor
        self.soft_temperature = soft_temperature
        self.top_k_anchor = top_k_anchor
        self.temperature_anneal = temperature_anneal
        self.use_sparse_softmax = use_sparse_softmax
        self.fisher_ema_decay = fisher_ema_decay

        # 温度退火范围
        self.temp_min = 0.03
        self.temp_max = max(soft_temperature, 0.5)

        # 使用频率追踪（用于联邦加权聚合）
        self.register_buffer('anchor_usage', torch.zeros(nb_class))

        key_pool_size = (nb_class, key_size)
        self.key = nn.Parameter(torch.randn(key_pool_size))
        nn.init.uniform_(self.key, -1, 1)

        self.head = Chead(200)
        anchor_pool_size = (nb_class, key_size)
        self.anchor_pool = nn.Parameter(torch.randn(anchor_pool_size))

        # Fisher 信息累积（用于 Fisher-Weighted Temporal Stability）
        self.register_buffer('anchor_fisher', torch.zeros(anchor_pool_size))

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """
        对张量进行 L2 归一化

        Args:
            x: 输入张量
            dim: 归一化维度
            epsilon: 数值稳定性参数

        Returns:
            L2 归一化后的张量
        """
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm

    def compute_temperature(self, global_step=None, total_steps=None):
        """
        计算当前温度（支持余弦退火）

        Args:
            global_step: 当前全局步数
            total_steps: 总步数

        Returns:
            当前温度值
        """
        if self.temperature_anneal and global_step is not None and total_steps is not None:
            progress = min(global_step / total_steps, 1.0)
            # 余弦退火: 从 temp_max 逐渐降到 temp_min
            temp = self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (
                1.0 + np.cos(np.pi * progress)
            )
            return temp
        return self.soft_temperature

    def forward(self, x, class_mask, global_step=None, total_steps=None):
        """
        前向传播（FedSMR 版本）

        Args:
            x: 预训练模型输出的特征向量 (batch_size x key_size)
            class_mask: 类别掩码（兼容旧调用）
            global_step: 全局步数（用于温度退火）
            total_steps: 总步数（用于温度退火）

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits: 分类 logits
                - output_mixed: 拼接后的特征向量 (B, 2*key_size)
                - reduce_sim: 拉约束损失值
                - anchor_feat: 组合后的 anchor 特征 (B, key_size)
                - attn_weights: attention 权重 (B, Pool_size)，用于使用频率统计
        """
        # 计算特征归一化
        x_embed_norm = self.l2_normalize(x, dim=1)  # B, C

        tem_key = self.key.reshape(-1, self.key_size)
        key_norm = self.l2_normalize(tem_key, dim=1)  # Pool_size, C

        # 计算余弦相似度
        key_norm_dev = key_norm.to(x.device)
        similarity = torch.matmul(x_embed_norm, key_norm_dev.t())  # B, Pool_size

        anchor_norm = self.l2_normalize(
            self.anchor_pool.reshape(-1, self.key_size), dim=1
        ).to(x.device)  # Pool_size, C

        if self.soft_anchor:
            # ==========================================================
            # FedSMR : Soft Anchor Mixture
            # ==========================================================
            temp = self.compute_temperature(global_step, total_steps)

            attn_logits = similarity / temp  # B, Pool_size

            if self.use_sparse_softmax and self.top_k_anchor is not None:
                # ---- Top-K Sparse Softmax ----
                # 仅对 top-k 最相似的 anchor 计算 softmax，其余置零
                top_k = min(self.top_k_anchor, attn_logits.shape[1])
                topk_logits, topk_idx = torch.topk(attn_logits, k=top_k, dim=1)
                attn_weights = torch.zeros_like(attn_logits)
                attn_weights.scatter_(1, topk_idx, torch.softmax(topk_logits, dim=1))
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            else:
                # ---- Full Softmax ----
                attn_weights = torch.softmax(attn_logits, dim=1)  # B, Pool_size

            # 加权组合 anchor 特征 (B,Pool) @ (Pool,C) -> (B, C)
            anchor_feat = torch.matmul(attn_weights, anchor_norm)

            # 拉约束损失：加权后的 feature 与加权后的 key-feature 对齐
            reduce_sim = torch.sum(
                x_embed_norm * torch.matmul(attn_weights, key_norm_dev)
            ) / self.key_size
        else:
            # ==========================================================
            # 原始 FedTA : hard top-1 anchor 选择（消融对比）
            # ==========================================================
            q, index = torch.topk(similarity, k=1)
            anchor_feat = self.anchor_pool[index].reshape(-1, self.key_size)

            # 构建 one-hot 形式的 attention weights（用于使用频率统计）
            attn_weights = torch.zeros(
                similarity.shape[0], similarity.shape[1], device=x.device
            )
            attn_weights.scatter_(1, index, 1.0)

            batched_key_norm = key_norm_dev[index]
            x_embed_norm_u = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm_u   # B, top_k, C
            reduce_sim = torch.sum(sim) / self.key_size  # Scalar

        # 拼接特征和锚点（用于分类头 + 原型计算）
        output_mixed = torch.stack((x, anchor_feat), dim=1).view(-1, self.key_size * 2)
        logits = self.head(output_mixed)

        return logits, output_mixed, reduce_sim, anchor_feat, attn_weights

    def anchor_diversity_loss(self):
        """
        FedSMR: Memory Structure Preservation - Intra-Pool Diversity

        计算 anchor 多样性损失，防止 anchor pool 发生坍缩。
        通过惩罚 anchor 之间过高的余弦相似度来保持多样性。

        Returns:
            diversity_loss: 标量损失值
        """
        anchor_norm = self.l2_normalize(
            self.anchor_pool.reshape(-1, self.key_size), dim=1
        )
        # 计算 anchor 之间的余弦相似度矩阵
        sim_matrix = anchor_norm @ anchor_norm.T  # (Pool_size, Pool_size)
        # 去掉对角线（自己和自己的相似度）
        mask = 1.0 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
        # 惩罚过高的 anchor 间相似度（仅惩罚正相似度，避免梯度抵消）
        loss_div = (sim_matrix * mask).clamp(min=0).mean()
        return loss_div

    def get_anchor_usage(self):
        """
        获取 anchor 使用频率统计

        Returns:
            anchor_usage: 每个 anchor 的累计使用权重
        """
        return self.anchor_usage.clone()

    def reset_anchor_usage(self):
        """重置 anchor 使用频率统计"""
        self.anchor_usage.zero_()

    def accumulate_fisher(self):
        """
        Fisher-Weighted Temporal Stability:
        累积 anchor_pool 梯度的平方作为 Fisher 信息的近似估计。

        F(θ) ≈ EMA( (∂L/∂θ)² )
        """
        if self.anchor_pool.grad is not None:
            grad_sq = self.anchor_pool.grad.detach() ** 2
            self.anchor_fisher = (
                self.fisher_ema_decay * self.anchor_fisher
                + (1 - self.fisher_ema_decay) * grad_sq
            )

    def get_anchor_fisher(self):
        """
        获取归一化后的 Fisher 权重。

        Returns:
            fisher: 归一化到 [0, 1] 的 Fisher 信息矩阵
        """
        fisher = self.anchor_fisher.clone()
        max_val = fisher.max()
        if max_val > 0:
            fisher = fisher / max_val
        return fisher

    def reset_anchor_fisher(self):
        """重置 Fisher 信息累积"""
        self.anchor_fisher.zero_()

    def load_head(self, head):
        """
        加载分类头权重

        Args:
            head: 分类头状态字典或模块
        """
        self.head = deepcopy(head)

    def get_head(self):
        """
        获取当前分类头

        Returns:
            Chead: 分类头模块
        """
        return self.head