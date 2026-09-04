"""
FedSMR: L2P (Learn to Prompt) 全局提示模块 (Soft Memory Retrieval)

该模块实现了 Global_Prompt 类，用于管理可学习的提示池，并根据输入特征
动态选择最相关的提示。

FedSMR 核心改进：
- Soft Prompt Retrieval: 使用 softmax 加权组合 prompt 替代 hard top-k 选择
  保持梯度可导，所有 prompt 都能接收梯度信号
- Prompt Usage Tracking: 追踪每个 prompt 的使用频率，用于联邦加权聚合
- Prompt Diversity: 保持 prompt 池的多样性

核心功能：
- 提示池管理
- 基于余弦相似度的动态提示选择（Hard / Soft 可选）
- 支持多种嵌入键计算方式（mean, max, mean_max, cls）
- 批次级提示选择选项
- 拉约束损失计算
"""

import torch
import torch.nn as nn


class Global_Prompt(nn.Module):
    """
    FedSMR 全局提示模块

    维护一个可学习的提示token池，并基于输入特征通过余弦相似度动态选择提示。
    FedSMR 新增 Soft Prompt Retrieval 模式。
    """

    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                 prompt_pool=False, prompt_key=False, pool_size=None, top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 use_soft_prompt=False, prompt_temperature=0.1):
        """
        初始化全局提示模块

        Args:
            length: 每个提示序列的长度
            embed_dim: 嵌入维度
            embedding_key: 计算嵌入键的方法 ('mean', 'max', 'mean_max', 'cls')
            prompt_init: 提示初始化方法 ('zero', 'uniform')
            prompt_pool: 是否使用提示池
            prompt_key: 是否使用可学习的提示键
            pool_size: 提示池中的提示数量
            top_k: 每个输入选择的提示数量
            batchwise_prompt: 是否对整个批次使用相同的提示
            prompt_key_init: 提示键初始化方法 ('zero', 'uniform')
            use_soft_prompt: 是否启用 Soft Prompt Retrieval（FedSMR）
            prompt_temperature: Soft Prompt Retrieval 的温度系数
        """
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        # FedSMR 超参数
        self.use_soft_prompt = use_soft_prompt
        self.prompt_temperature = prompt_temperature

        # 使用频率追踪（用于联邦加权聚合）
        if pool_size is not None:
            self.register_buffer('prompt_usage', torch.zeros(pool_size))

        prompt_pool_shape = (pool_size, length, embed_dim)
        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)

        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

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
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        """
        前向传播，选择并应用提示（FedSMR 版本）

        Args:
            x_embed: 输入嵌入 (batch_size x num_patches x embed_dim)
            prompt_mask: 可选的任务特定提示选择掩码
            cls_features: 可选的 CLS token 特征，用于嵌入键计算

        Returns:
            Dictionary 包含:
                - prompted_embedding: 拼接提示后的输入
                - total_prompt_len: 选择的提示总长度
                - prompt_idx: 选择的提示索引
                - similarity: 输入与提示键之间的余弦相似度
                - reduce_sim: 拉约束损失值
                - prompt_weights: Soft Prompt 的 attention 权重（FedSMR 新增）
        """
        out = dict()

        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())

            if prompt_mask is None:
                if self.use_soft_prompt:
                    # ==========================================================
                    # FedSMR : Soft Prompt Retrieval
                    # 使用 softmax 加权 top-k prompt，保持梯度可导
                    # 当 top_k=1 时退化为 hard 选择（合理行为）
                    # 当 top_k>1 时多个 prompt 按权重加权组合
                    # ==========================================================
                    attn_weights = torch.softmax(
                        similarity / self.prompt_temperature, dim=1
                    )  # (B, pool_size)

                    # 追踪使用频率
                    if hasattr(self, 'prompt_usage'):
                        self.prompt_usage += attn_weights.sum(dim=0).detach().to(self.prompt_usage.device)

                    # Top-k 加权选择：取 top-k 权重最大的 prompt，按权重缩放
                    batch_size = similarity.shape[0]
                    topk_weights, topk_idx = torch.topk(attn_weights, k=self.top_k, dim=1)
                    # 在 top-k 内重新归一化
                    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)

                    # 获取选中的 prompt tokens
                    batched_prompt_raw = self.prompt[topk_idx]  # (B, top_k, length, embed_dim)
                    # 按权重缩放
                    batched_prompt_raw = batched_prompt_raw * topk_weights.unsqueeze(-1).unsqueeze(-1)
                    
                    # ==========================================================
                    # 幅度归一化：使 soft prompt 的总幅度与 hard prompt 一致
                    # hard 模式下每个 prompt 权重为 1，总幅度 = top_k
                    # soft 模式下权重和为 1（归一化后），总幅度 = 1
                    # 需要乘以 top_k 使其幅度与 hard 模式匹配
                    # ==========================================================
                    batched_prompt_raw = batched_prompt_raw * self.top_k
                    
                    batched_prompt = batched_prompt_raw.reshape(
                        batch_size, self.top_k * self.length, self.embed_dim
                    )

                    out['prompt_idx'] = topk_idx
                    out['prompt_weights'] = attn_weights  # FedSMR: 返回完整权重

                    # 软拉约束损失：加权 key 与 x_embed 对齐
                    selected_key_weighted = (
                        prompt_norm[topk_idx] * topk_weights.unsqueeze(-1)
                    ).sum(dim=1)  # (B, C)
                    sim = selected_key_weighted * x_embed_norm
                    reduce_sim = torch.sum(sim) / x_embed.shape[0]
                else:
                    # ==========================================================
                    # 原始 FedTA : hard top-k 选择（消融对比）
                    # ==========================================================
                    _, idx = torch.topk(similarity, k=self.top_k, dim=1)

                    if self.batchwise_prompt:
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        if prompt_id.shape[0] < self.pool_size:
                            prompt_id = torch.cat([
                                prompt_id,
                                torch.full((self.pool_size - prompt_id.shape[0],),
                                           torch.min(idx.flatten()),
                                           device=prompt_id.device)
                            ])
                            id_counts = torch.cat([
                                id_counts,
                                torch.full((self.pool_size - id_counts.shape[0],),
                                           0,
                                           device=id_counts.device)
                            ])
                        _, major_idx = torch.topk(id_counts, k=self.top_k)
                        major_prompt_id = prompt_id[major_idx]
                        idx = major_prompt_id.expand(x_embed.shape[0], -1)

                    batched_prompt_raw = self.prompt[idx]
                    batch_size, top_k, length, c = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

                    out['prompt_idx'] = idx
                    out['prompt_weights'] = None  # hard 模式无权重

                    batched_key_norm = prompt_norm[idx]
                    out['selected_key'] = batched_key_norm
                    x_embed_norm_expanded = x_embed_norm.unsqueeze(1)
                    sim = batched_key_norm * x_embed_norm_expanded
                    reduce_sim = torch.sum(sim) / x_embed.shape[0]
            else:
                idx = prompt_mask
                batched_prompt_raw = self.prompt[idx]
                batch_size, top_k, length, c = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)
                out['prompt_idx'] = idx
                out['prompt_weights'] = None

                batched_key_norm = prompt_norm[idx]
                out['selected_key'] = batched_key_norm
                x_embed_norm_expanded = x_embed_norm.unsqueeze(1)
                sim = batched_key_norm * x_embed_norm_expanded
                reduce_sim = torch.sum(sim) / x_embed.shape[0]

            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity
            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
            out['prompt_weights'] = None
            out['reduce_sim'] = 0

        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out

    def prompt_diversity_loss(self):
        """
        FedSMR: Memory Structure Preservation - Prompt Pool Diversity

        计算 prompt 池的多样性损失，防止 prompt 向量趋同。

        Returns:
            diversity_loss: 标量损失值
        """
        if not self.prompt_pool or self.pool_size is None:
            return torch.tensor(0.0)

        # 将 prompt 展开为 (pool_size, length * embed_dim)
        prompt_flat = self.prompt.reshape(self.pool_size, -1)
        prompt_norm = self.l2_normalize(prompt_flat, dim=1)
        sim_matrix = prompt_norm @ prompt_norm.T
        # 去掉对角线
        mask = 1.0 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
        loss_div = (sim_matrix * mask).clamp(min=0).mean()
        return loss_div

    def get_prompt_usage(self):
        """
        获取 prompt 使用频率统计

        Returns:
            prompt_usage: 每个 prompt 的累计使用权重
        """
        if hasattr(self, 'prompt_usage'):
            return self.prompt_usage.clone()
        return None

    def reset_prompt_usage(self):
        """重置 prompt 使用频率统计"""
        if hasattr(self, 'prompt_usage'):
            self.prompt_usage.zero_()