"""
FedSMR 框架客户端模块

该模块实现了 FedSMR（Federated Soft Memory Retrieval）框架的客户端类。
每个客户端维护自己的模型、训练数据，并执行本地训练和评估。

核心改进（FedSMR）：
- Unified Soft Memory Retrieval: Prompt 和 Anchor 均使用 softmax 加权检索
- Memory Structure Preservation (MSP): 三层正则化防止记忆坍缩
  - Intra-Pool Diversity: 保持 pool 内向量多样性
  - Cross-Pool Coherence: 保持 prompt 和 anchor 记忆空间结构一致
  - Temporal Stability: 防止记忆在联邦轮次间剧烈变化
- Temperature Annealing: 温度从高到低退火
- Usage Tracking: 追踪记忆使用频率供联邦聚合

核心功能：
- 基于提示学习的本地训练
- 多种评估方法（标准评估、余弦相似度、仅提示、仅分类头）
- InfoNCE 对比学习支持
- 准确率指标日志系统
"""

import csv
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from Models.Tail_Anchor import Tail_Anchor
from Models.classification_head import Chead
from utils import CosineSimilarityClassifier


class Client_DF:
    """
    FedSMR 联邦客户端类

    每个客户端处理：
    - 本地数据管理和分割
    - 基于 Soft Memory Retrieval 的训练
    - 多种评估策略
    - 训练指标日志
    """

    def __init__(self, client_id, original_model, model_name, task_per_global_epoch,
                 subset, local_epoch, batch_size, lr, device, method, class_mask, args, vit, log_file=None):
        """
        初始化客户端实例（FedSMR 版本）

        Args:
            client_id: 客户端唯一标识符
            original_model: 预训练基础模型（如 ViT）
            model_name: 本地模型架构名称
            task_per_global_epoch: 每个全局训练轮次的任务数量
            subset: 该客户端的训练数据子集
            local_epoch: 本地训练轮次数量
            batch_size: 训练批次大小
            lr: 学习率
            device: 训练设备（CPU/GPU）
            method: 训练方法标识符
            class_mask: 每个任务的类别索引
            args: 命令行参数
            vit: 支持提示的 Vision Transformer 模型
        """
        self.id = client_id
        self.original_model = original_model
        self.vit = vit

        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader = []
        self.train_data = subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.method = method
        self.nb_classes = args.nb_classes

        # ---- FedSMR 超参数 -------------------------------------------------
        self.use_soft_anchor = getattr(args, 'use_soft_anchor', True)
        self.soft_temperature = getattr(args, 'soft_temperature', 0.1)
        self.use_soft_prompt = getattr(args, 'use_soft_prompt', False)
        self.prompt_temperature = getattr(args, 'prompt_temperature', 0.1)
        self.use_sparse_softmax = getattr(args, 'use_sparse_softmax', False)
        self.top_k_anchor = getattr(args, 'top_k_anchor', None)
        self.temperature_anneal = getattr(args, 'temperature_anneal', False)
        self.use_msp = getattr(args, 'use_msp', True)
        self.msp_diversity_coeff = getattr(args, 'msp_diversity_coeff', 0.05)
        self.msp_coherence_coeff = getattr(args, 'msp_coherence_coeff', 0.01)
        self.msp_temporal_coeff = getattr(args, 'msp_temporal_coeff', 0.01)
        # --------------------------------------------------------------------

        self.model = self._init_local_model(model_name)

        # ---- FedSMR: 将 soft prompt 参数传播到 Global_Prompt 模块 ----
        if hasattr(self.vit, 'prompt'):
            self.vit.prompt.use_soft_prompt = self.use_soft_prompt
            self.vit.prompt.prompt_temperature = self.prompt_temperature
        # ------------------------------------------------------------

        # 存储上一轮的记忆状态（用于 temporal stability loss）
        self.prev_anchor_pool = None
        self.prev_prompt = None
        self.prev_prompt_key = None

        # 自适应 MSP: 是否有公共类（用于 coherence 开关）
        # |Cp| = nb_classes - client_num * private_class_num
        total_private = args.client_num * args.private_class_num
        public_count = max(0, args.nb_classes - total_private)
        self._has_public_classes = (public_count > 0)  # CIFAR-100: True, ImageNet-R: False

        # Initialize class mask based on dataset type
        if args.data_name in ['cifar100', '5datasets', 'ImageNet-R', 'svhn-mnist']:
            self.class_mask = class_mask
        else:
            self.class_mask = []

        # Initialize model components
        self.local_protos = None
        self.global_protos = None
        self.prompts = None
        self.heads = [None] * 10  # Classifier heads for up to 10 tasks
        self.vit_heads = [None] * 10
        self.head = Chead(args.nb_classes)

        # Initialize logging system
        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = self._init_log_file()
        self.round = 0

    def _init_local_model(self, model_name):
        """
        根据指定的架构初始化本地模型（传入 FedSMR 超参数）

        Args:
            model_name: 模型架构名称

        Returns:
            初始化的模型实例
        """
        if model_name == 'Tail_Anchor':
            return Tail_Anchor(
                anchor_size=10,
                key_size=768,
                nb_class=200,
                soft_anchor=self.use_soft_anchor,
                soft_temperature=self.soft_temperature,
                top_k_anchor=self.top_k_anchor,
                temperature_anneal=self.temperature_anneal,
                use_sparse_softmax=self.use_sparse_softmax,
            )

    def _init_log_file(self):
        """
        初始化用于记录准确率指标的 CSV 日志文件

        Returns:
            创建的日志文件路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"log_{timestamp}.csv"
        filepath = os.path.join("logs", filename)

        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Round', 'Task_id', 'Client_id', 'Accuracy', 'Notes', 'Phase'])

        return filepath

    def _log_accuracy(self, accuracy, notes, phase, task_id=None):
        """
        将准确率指标记录到日志文件
        """
        if task_id is None:
            task_id = self.task_id

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.round, task_id, self.id, accuracy, notes, phase])

    def set_round(self, round_num):
        """更新当前全局轮次编号"""
        self.round = round_num

    def get_data_office_home(self, task_id, data, mask):
        """准备 Office-Home 数据集的训练数据"""
        self.train_dataset = data
        self.current_class = mask
        self.class_mask.append(mask)
        print(f'Client {self.id}, Task {task_id} has {len(self.current_class)} classes: {self.current_class}')

        trainset = self.train_dataset
        traindata, testdata = random_split(trainset, [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        self.test_loader.append(testdata)
        self.traindata = traindata

    def get_data(self, task_id):
        """准备标准数据集的训练数据"""
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'Client {self.id}, Task {task_id} has {len(self.current_class)} classes: {self.current_class}')

        trainset = self.train_dataset
        traindata, testdata = random_split(trainset, [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        self.test_loader.append(testdata)
        self.traindata = traindata
        print(len(traindata))

    def update_data(self, round, args):
        """切换到新任务时更新训练数据"""
        task = round // self.task_per_global_epoch
        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R', 'svhn-mnist']:
                self.get_data(task)
            self.task_id = task

    def _compute_msp_losses(self, feat_prompt, anchor_feat, round_num):
        """
        FedSMR: 计算 Memory Structure Preservation 损失（自适应版本）

        三层正则化：
        1. Intra-Pool Diversity:  保持 anchor 池内多样性（所有场景适用）
        2. Cross-Pool Coherence:  保持 prompt-anchor 一致性（仅在有关公共类时生效）
        3. Temporal Stability:    置信度加权时序约束（使用频率高→强约束，低→弱约束）

        Args:
            feat_prompt: prompt 增强后的特征 (B, C)
            anchor_feat: anchor 组合后的特征 (B, C)
            round_num: 当前全局轮次

        Returns:
            total_msp_loss: MSP 总损失
        """
        total_msp_loss = torch.tensor(0.0, device=self.device)

        # ---- 1. Intra-Pool Diversity ----
        # 所有场景统一使用，不做自适应
        if hasattr(self.model, 'anchor_diversity_loss'):
            loss_anchor_div = self.model.anchor_diversity_loss()
            total_msp_loss = total_msp_loss + self.msp_diversity_coeff * loss_anchor_div

        # Prompt diversity (通过 vit 的 prompt 模块)
        if self.use_soft_prompt and hasattr(self.vit, 'prompt'):
            for attr_name in ['prompt', 'e_prompt', 'g_prompt']:
                if hasattr(self.vit, attr_name):
                    prompt_module = getattr(self.vit, attr_name)
                    if hasattr(prompt_module, 'prompt_diversity_loss'):
                        loss_prompt_div = prompt_module.prompt_diversity_loss()
                        total_msp_loss = total_msp_loss + self.msp_diversity_coeff * loss_prompt_div
                        break

        # ---- 2. Cross-Pool Coherence（自适应：无公共类时自动禁用） ----
        if feat_prompt is not None and anchor_feat is not None and self.msp_coherence_coeff > 0:
            # |Cp|=0 时禁用 coherence，有公共类时保持原始强度
            if self._has_public_classes:
                f_p_norm = torch.nn.functional.normalize(feat_prompt, p=2, dim=1)
                f_a_norm = torch.nn.functional.normalize(anchor_feat, p=2, dim=1)
                cos_sim = torch.sum(f_p_norm * f_a_norm, dim=1)
                # 区间约束: cos 在 [0.3, 0.7] 内不惩罚
                lower_violation = torch.clamp(0.3 - cos_sim, min=0)
                upper_violation = torch.clamp(cos_sim - 0.7, min=0)
                loss_coherence = (lower_violation + upper_violation).mean()
                total_msp_loss = total_msp_loss + (
                    self.msp_coherence_coeff * loss_coherence
                )

        # ---- 3. Temporal Stability（自适应：置信度加权） ----
        if self.prev_anchor_pool is not None and self.msp_temporal_coeff > 0:
            current_anchor = self.model.anchor_pool.data
            prev_anchor = self.prev_anchor_pool.to(self.device)

            # 置信度加权: 使用频率高的 anchor → 更可靠的估计 → 更强的时序约束
            # 使用频率低的 anchor → 噪声大 → 弱约束（允许自适应调整）
            if hasattr(self.model, 'get_anchor_usage'):
                usage = self.model.get_anchor_usage().to(self.device)  # (nb_class,)
                # 归一化到 [0, 1]，避免绝对尺度影响
                usage_max = usage.max()
                if usage_max > 0:
                    confidence = (usage / usage_max).unsqueeze(1)  # (nb_class, 1)
                else:
                    confidence = torch.ones_like(current_anchor)
            else:
                confidence = torch.ones_like(current_anchor)

            # 加权 MSE: 高置信度 anchor → 强约束；低置信度 → 弱约束
            loss_temporal = (
                confidence * (current_anchor - prev_anchor) ** 2
            ).mean()
            total_msp_loss = total_msp_loss + self.msp_temporal_coeff * loss_temporal

        return total_msp_loss

    def train(self, round, args):
        """
        FedSMR: 使用 Soft Memory Retrieval 执行本地训练

        Phase 1: 训练 prompt 参数（Soft Prompt Retrieval + MSP）
        Phase 2: 训练 anchor pool + classification head（Soft Anchor + MSP + InfoNCE）

        Args:
            round: 当前全局轮次
            args: 命令行参数
        """
        self.original_model.eval()

        self.set_round(round)

        # 保存上一轮记忆状态（用于 temporal stability）
        if hasattr(self.model, 'anchor_pool'):
            self.prev_anchor_pool = self.model.anchor_pool.data.clone().cpu()
        if self.prompts is not None:
            self.prev_prompt = deepcopy(self.prompts)

        # Load or initialize prompts
        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(
            self.traindata, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )
        print(f'Client {self.id} on Task {self.task_id} is training (FedSMR)')

        # 计算总步数（用于温度退火）
        total_steps = self.local_epoch * len(train_loader)
        global_step = 0

        # =============================================================
        # Phase 1: Train prompts with Soft Prompt Retrieval + MSP
        # =============================================================
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(
                    self.device, non_blocking=True
                )
                target = target.long().to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']

                output = self.vit(input, task_id=self.task_id,
                                  cls_features=cls_features, train=True)
                logits = output['logits']
                pull_off = output['reduce_sim']
                feat_prompt = output['feat']

                # Apply class mask to logits
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask,
                                           value=float('-inf'))

                loss = criterion(logits, target) - 0.1 * pull_off

                # ---- FedSMR: MSP Loss (Phase 1) ----
                if self.use_msp and feat_prompt is not None:
                    with torch.no_grad():
                        # 用冻结的 anchor 获取 anchor 特征用于 coherence loss
                        _, _, _, anchor_feat, _ = self.model(
                            feat_prompt.detach(), target.to(self.device)
                        )
                    msp_loss = self._compute_msp_losses(feat_prompt, anchor_feat, round)
                    loss = loss + msp_loss
                # ------------------------------------

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1

        # =============================================================
        # Phase 2: Train classification head + anchor_pool (Soft Anchor + MSP + InfoNCE)
        # =============================================================
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(
                    self.device, non_blocking=True
                )
                target = target.long().to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                    output = self.vit(input, task_id=self.task_id,
                                      cls_features=cls_features, train=True)
                    feat_prompt = output['feat']

                # FedSMR: forward 返回 5 个值 (logits, output_mixed, reduce_sim, anchor_feat, attn_weights)
                pre, output_mixed, pull_off2, anchor_feat, attn_weights = self.model(
                    feat_prompt.to(self.device), target.to(self.device),
                    global_step=global_step, total_steps=total_steps * 2
                )
                logits = pre

                # Calculate InfoNCE loss if global prototypes exist
                if self.global_protos is None:
                    loss_infonce = 0
                else:
                    count = 0
                    loss_infonce = None
                    for i, label in enumerate(target):
                        if label.item() in self.global_protos:
                            count += 1
                            feature = output_mixed[i].unsqueeze(0)
                            loss_instance = self._calculate_infonce(
                                feature, label.item(),
                                (round + 1) % self.task_per_global_epoch == 0,
                            )
                            loss_infonce = (
                                loss_instance
                                if loss_infonce is None
                                else loss_infonce + loss_instance
                            )

                    loss_infonce = loss_infonce / count if count != 0 else 0

                # Apply class mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask,
                                           value=float('-inf'))

                loss = (
                    criterion(logits, target)
                    + 0.2 * self.task_per_global_epoch * loss_infonce
                    - 0.1 * pull_off2
                )

                # ---- FedSMR: MSP Loss (Phase 2) ----
                if self.use_msp:
                    msp_loss = self._compute_msp_losses(feat_prompt, anchor_feat, round)
                    loss = loss + msp_loss
                # ------------------------------------

                if round == 16 and self.id == 0:
                    print(loss_infonce)
                    print(criterion(logits, target))

                optimizer.zero_grad()
                loss.backward()

                
                optimizer.step()

                global_step += 1

        # Extract local prototypes from training data
        target_list = []
        feature_list = []

        for input, target in train_loader:
            input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input.to(self.device))
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output,
                                  train=True)
                _, output_mixed, _, _, _ = self.model(
                    output['feat'].to(self.device), target.to(self.device)
                )

            if not np.isnan(output_mixed.cpu().detach().numpy()).any():
                target_list.append(target)
                feature_list.append(output_mixed)

        local_protos = {}
        if target_list:
            target_list = torch.cat(target_list, dim=0)
            feature_list = torch.cat(feature_list, dim=0)

            for class_index in self.current_class:
                data_index = (target_list == class_index).nonzero().squeeze(-1)
                if data_index.shape[0] != 0:
                    all_features = feature_list[data_index]
                    proto = all_features.mean(0).cpu().detach().numpy()
                    local_protos[class_index] = proto

        self.local_protos = local_protos
        self.heads[self.task_id] = deepcopy(self.model.get_head())

        # Evaluate after training
        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes)
        else:
            self.evaluate(0, args.nb_classes)
            self.evaluate(self.task_id, args.nb_classes)

        self.prompts = deepcopy(self.vit.get_prompts())

    def load_global_weights(self, weights):
        """加载全局模型权重并评估"""
        self.model.load_state_dict(weights)
        self.evaluate(self.task_id, self.nb_classes)

    def get_global_proto_and_head(self, proto, head, prompt, round_num):
        """更新全局原型、分类头和提示，然后进行评估"""
        self.global_protos = deepcopy(proto)
        self.global_head = head
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        if self.task_id == 0:
            self.evaluate_with_global_head(self.task_id, self.nb_classes)
        else:
            self.evaluate_with_global_head(0, self.nb_classes)
            self.evaluate_with_global_head(self.task_id, self.nb_classes)

    def get_global_proto_and_head_no_test(self, proto, head, prompt, round_num):
        """更新全局原型、分类头和提示，不进行评估"""
        self.global_protos = deepcopy(proto)
        self.global_head = head
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

    def get_head(self, head):
        """更新本地分类头并评估"""
        self.heads[self.task_id] = deepcopy(head)
        if self.task_id == 0:
            self.evaluate(self.task_id, self.nb_classes)
        else:
            self.evaluate(0, self.nb_classes)
            self.evaluate(self.task_id, self.nb_classes)

    def train_only_heads(self, round_num, args):
        """仅训练分类头（不训练提示）"""
        task = round_num // self.task_per_global_epoch

        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R']:
                self.get_data(task)
            self.task_id = task

        if self.heads[self.task_id] is not None:
            self.head.load_head(self.heads[self.task_id])

        self.head.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.local_epoch):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)

                output = self.head(output['feat'])
                logits = output

                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.head.get_head())

        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes)
        else:
            self.evaluate(0, args.nb_classes)
            self.evaluate(self.task_id, args.nb_classes)

    def evaluate(self, task=0, nb_classes=None):
        """使用提示和分类头进行标准评估"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        correct = 0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, _, _, _, _ = self.model(output['feat'].to(self.device), target.to(self.device))

            logits = pre

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), f"Local evaluation on task {task}", "Local Training", task_id=task)

    def evaluate_with_global_head(self, task=0, nb_classes=None):
        """使用全局分类头进行评估（服务器聚合后）"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        correct = 0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, _, _, _, _ = self.model(output['feat'].to(self.device), target.to(self.device))

            logits = pre

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'Global Head Evaluation: {acc}')

        self._log_accuracy(acc.item(), f"Global head evaluation on task {task}", "Server Aggregation", task_id=task)

    def evaluate_cosin_similarity(self, task=0, nb_classes=None):
        """使用全局原型的余弦相似度进行评估"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                _, output_mix, _, _, _ = self.model(output, None)

            for i, label in enumerate(target):
                predicts = CosineSimilarityClassifier(output_mix[i].squeeze(0), self.global_protos, self.current_class)
                if predicts == label:
                    correct += 1
            total += len(target)

        acc = 100 * correct / total
        print(f'Client {self.id} on Task {task} acc is {acc}')

        self._log_accuracy(acc, f"Cosine similarity evaluation on task {task}", "Cosine Similarity", task_id=task)

    def evaluate_only_prompts(self, task=0, nb_classes=None):
        """仅使用提示增强的特征进行评估（不使用分类头）"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)

            logits = output['logits']

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), f"Prompts only evaluation on task {task}", "Prompts Only", task_id=task)

    def evaluate_only_heads(self, task=0, nb_classes=None):
        """仅使用分类头进行评估（不使用提示）"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        self.vit.load_head(self.heads[task])
        self.vit.to(self.device)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                output = self.head(output['feat'])

            logits = output

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), f"Heads only evaluation on task {task}", "Heads Only", task_id=task)

    def _calculate_infonce(self, feature, label, is_last):
        """计算 InfoNCE 对比损失"""
        all_global_protos_keys = np.array(list(self.global_protos.keys()))
        all_protos = [self.global_protos[key] for key in all_global_protos_keys]
        all_protos = np.vstack(all_protos)

        pos_index = np.where(all_global_protos_keys == label)[0]
        neg_index = np.where(all_global_protos_keys != label)[0]

        f_pos = torch.from_numpy(all_protos[pos_index]).to(self.device)
        f_neg = torch.from_numpy(all_protos[neg_index]).to(self.device)
        f_proto = torch.cat((f_pos, f_neg), dim=0)

        l = torch.cosine_similarity(feature, f_proto, dim=1)
        l = l / 0.2

        exp_l = torch.exp(l).view(1, -1)
        pos_mask = torch.tensor([1] * f_pos.shape[0] + [0] * f_neg.shape[0],
                                 dtype=torch.float).to(self.device).view(1, -1)

        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)

        if is_last:
            infonce_loss = 1 - torch.log(sum_pos_l)
        else:
            infonce_loss = -torch.log(sum_pos_l / sum_exp_l)

        return infonce_loss

    def train_only_prompts(self, round_num, args):
        """仅训练提示参数（不训练分类头）"""
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        if self.heads[self.task_id] is not None:
            self.vit.load_head(self.heads[self.task_id])

        task = round_num // self.task_per_global_epoch

        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R']:
                self.get_data(task)
            self.task_id = task

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']

                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                logits = output['logits']
                pull_off = output['reduce_sim']

                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target) - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.vit.head)
        self.prompts = deepcopy(self.vit.get_prompts())

        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes)
        else:
            self.evaluate(0, args.nb_classes)
            self.evaluate(self.task_id, args.nb_classes)

    def get_global_prompt_head(self, head, prompt):
        """更新全局提示和分类头参数"""
        self.heads[self.task_id] = deepcopy(head)
        self.model.head.to(self.device)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        if self.task_id == 0:
            self.evaluate(self.task_id, self.nb_classes)
        else:
            self.evaluate(0, self.nb_classes)
            self.evaluate(self.task_id, self.nb_classes)

    def evaluate_on_global_testset(self, testdata):
        """在全局测试数据集上进行评估"""
        test_loader = DataLoader(testdata, batch_size=16, shuffle=True, num_workers=2)

        self.vit.load_head(self.global_head)
        self.vit.to(self.device)

        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    _, _, _, _, _ = self.model(output['feat'].to(self.device), target.to(self.device))

                output = self.head(output['feat'])

            logits = output
            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), "Global test set evaluation", "Global Test", task_id=self.task_id)

    def get_anchor_usage(self):
        """获取 anchor 使用频率统计（FedSMR: 供联邦聚合使用）"""
        if hasattr(self.model, 'get_anchor_usage'):
            return self.model.get_anchor_usage()
        return None

    def get_prompt_usage(self):
        """获取 prompt 使用频率统计（FedSMR: 供联邦聚合使用）"""
        if hasattr(self.vit, 'prompt') and hasattr(self.vit.prompt, 'get_prompt_usage'):
            return self.vit.prompt.get_prompt_usage()
        return None