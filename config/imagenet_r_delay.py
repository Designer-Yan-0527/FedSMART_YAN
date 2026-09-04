"""
ImageNet-R Delay 配置模块

该模块定义了 ImageNet-R 延迟实验的命令行参数。

参数分类：
- 训练参数：batch-size
- 模型参数：model, input-size, pretrained, drop, drop-path
- 优化器参数：opt, opt-eps, opt-betas, clip-grad, momentum, weight-decay
- 学习率调度：sched, lr, warmup-lr, min-lr, decay-epochs
- 数据增强：color-jitter, aa, smoothing, train-interpolation
- 随机擦除：reprob, remode, recount
- 数据参数：data-path, dataset, shuffle, output_dir, device, seed
- 持续学习：train_mask, task_inc
- Prompt 参数：size, length, top_k, initializer, prompt_key
- ViT 参数：global_pool, head_type, freeze
- FedTA 参数：method, client_num, task_num, private_class_num
"""

import argparse


def get_args_parser(subparsers):
    """
    添加 ImageNet-R Delay 实验的命令行参数
    
    Args:
        subparsers: argparse 子解析器
    """
    # 训练参数
    subparsers.add_argument('--batch-size', default=16, type=int, help='每个设备的批次大小')

    # 模型参数
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', 
                           help='训练模型名称')
    subparsers.add_argument('--input-size', default=224, type=int, help='图像输入尺寸')
    subparsers.add_argument('--pretrained', default=True, help='是否加载预训练模型')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout 率')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path 率')

    # 优化器参数
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='优化器类型')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='优化器 epsilon')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', 
                           help='优化器 beta 参数')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM', help='梯度裁剪范数')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD 动量')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='是否重新初始化优化器')

    # 学习率调度参数
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='学习率调度器')
    subparsers.add_argument('--lr', type=float, default=0.001, metavar='LR', help='学习率')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', 
                           help='学习率噪声开关百分比')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='学习率噪声限制')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='学习率噪声标准差')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='热身学习率')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='最小学习率')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='学习率衰减间隔')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='热身轮数')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='冷却轮数')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='Plateau 调度器耐心轮数')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='学习率衰减率')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='是否按批次大小缩放学习率')

    # 数据增强参数
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='颜色抖动因子')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='使用 AutoAugment 策略')
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='标签平滑')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='训练插值方法')

    # 随机擦除参数
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='随机擦除概率')
    subparsers.add_argument('--remode', type=str, default='pixel', help='随机擦除模式')
    subparsers.add_argument('--recount', type=int, default=1, help='随机擦除次数')

    # 数据参数
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='数据集路径')
    subparsers.add_argument('--dataset', default='ImageNet-R', type=str, help='数据集名称')
    subparsers.add_argument('--shuffle', default=False, help='是否打乱数据顺序')
    subparsers.add_argument('--output_dir', default='output/', help='输出路径')
    subparsers.add_argument('--device', default='cuda', help='训练设备')
    subparsers.add_argument('--seed', default=42, type=int, help='随机种子')
    subparsers.add_argument('--eval', action='store_true', help='仅执行评估')
    subparsers.add_argument('--num_workers', default=2, type=int, help='数据加载线程数')
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='固定 CPU 内存以提高传输效率')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='不固定内存')
    subparsers.set_defaults(pin_mem=True)

    # 持续学习参数
    subparsers.add_argument('--train_mask', default=True, type=bool, help='训练时是否使用类别掩码')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='是否进行任务增量学习')

    # Prompt Pool 参数
    subparsers.add_argument('--size', default=20, type=int, help='Prompt 池大小')
    subparsers.add_argument('--length', default=15, type=int, help='Prompt 长度')
    subparsers.add_argument('--top_k', default=1, type=int, help='选择的 Prompt 数量')
    subparsers.add_argument('--initializer', default='uniform', type=str, help='初始化方法')
    subparsers.add_argument('--prompt_key', default=True, type=bool, help='是否使用 Prompt 键')
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str, help='Prompt 键初始化')
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool, help='是否使用 Prompt 掩码')
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool, help='是否共享 Prompt 池')
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool, help='是否共享 Prompt 键')
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool, help='是否批次级 Prompt')
    subparsers.add_argument('--embedding_key', default='cls', type=str, help='嵌入键类型')
    subparsers.add_argument('--predefined_key', default='', type=str, help='预定义键')
    subparsers.add_argument('--pull_constraint', default=True, help='是否使用拉约束')
    subparsers.add_argument('--pull_constraint_coeff', default=0.1, type=float, help='拉约束系数')

    # ViT 参数
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, 
                           help='全局池化类型')
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], 
                           type=str, help='分类头输入类型')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], 
                           nargs='*', type=list, help='冻结的模型部分')

    # 杂项参数
    subparsers.add_argument('--print_freq', type=int, default=10, help='打印频率')

    # FedTA 参数
    subparsers.add_argument('--method', type=str, default='delay', help='Prompt 方法')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='E-Prompt 层索引')
    subparsers.add_argument('--client_num', default=10, type=int, help='客户端数量')
    subparsers.add_argument('--task_num', default=10, type=int, help='任务数量')
    subparsers.add_argument('--private_class_num', default=40, type=int, help='每个客户端私有类别数')
    subparsers.add_argument('--surrogate_num', default=20, type=int, help='代理数据数量')
    subparsers.add_argument('--global_epoch', default=5, type=int, help='全局训练轮数')
    subparsers.add_argument('--local_epoch', default=20, type=int, help='本地训练轮数')
    subparsers.add_argument('--threshold', default=0.1, type=float, help='相似度阈值')
    subparsers.add_argument('--data_name', default='ImageNet-R', type=str, help='数据集名称')
    subparsers.add_argument('--model_name', default='Tail_Anchor', choices=['AlexNet', 'VGG16', 'ResNet18', 'Tail_Anchor'],
                           type=str, help='模型名称')

    # ===== FedSMR 超参数（论文改进方案） =====
    # -- Unified Soft Memory Retrieval --
    subparsers.add_argument('--use_soft_anchor', default=True, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Soft Anchor Mixture（余弦相似度 + softmax 加权）')
    subparsers.add_argument('--soft_temperature', default=0.1, type=float,
                            help='Soft Anchor Mixture 的 softmax 温度系数，越大越平滑')
    subparsers.add_argument('--use_soft_prompt', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Soft Prompt Retrieval（prompt 侧 softmax 加权）')
    subparsers.add_argument('--prompt_temperature', default=0.1, type=float,
                            help='Soft Prompt Retrieval 的温度系数')

    # -- Top-K Sparse Softmax --
    subparsers.add_argument('--use_sparse_softmax', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Top-K 稀疏 Softmax（仅 top-k anchor 参与 softmax）')
    subparsers.add_argument('--top_k_anchor', default=None, type=int,
                            help='Top-K 稀疏 Softmax 的 K 值，None 表示使用全部 anchor')

    # -- Temperature Annealing --
    subparsers.add_argument('--temperature_anneal', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用温度退火（训练中温度从高到低）')

    # -- Memory Structure Preservation (MSP) --
    subparsers.add_argument('--use_msp', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 MSP 三层正则化（多样性 + 一致性 + 时序稳定性）')
    subparsers.add_argument('--msp_diversity_coeff', default=0.1, type=float,
                            help='Intra-Pool Diversity 正则化系数')
    subparsers.add_argument('--msp_coherence_coeff', default=0.1, type=float,
                            help='Cross-Pool Coherence 正则化系数')
    subparsers.add_argument('--msp_temporal_coeff', default=0.1, type=float,
                            help='Temporal Stability 正则化系数')

    # -- Federated Usage-Weighted Aggregation --
    subparsers.add_argument('--use_fed_smr_aggregate', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用基于使用频率的联邦加权聚合')

    # ===== EMA Prototype Update（方案1） =====
    subparsers.add_argument('--use_ema_proto', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 EMA 原型更新（指数移动平均平滑全局原型）')
    subparsers.add_argument('--ema_momentum', default=0.9, type=float,
                            help='EMA 动量系数，越大越平滑，建议 [0.8, 0.99]')

    # ===== Quality-Weighted Prototype Selection（方案1） =====
    subparsers.add_argument('--use_quality_weight', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用原型质量加权（样本量越大的原型权重越高）')
    subparsers.add_argument('--quality_lambda', default=0.1, type=float,
                            help='样本量对原型选择的影响强度，建议 [0.01, 0.5]')

    # ===== Fisher-Weighted Temporal Stability（方案2） =====
    subparsers.add_argument('--use_fisher_temporal', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Fisher 加权的时序稳定性（重要 anchor 更严格约束）')
    subparsers.add_argument('--fisher_ema_decay', default=0.9, type=float,
                            help='Fisher 信息累积的 EMA 衰减系数，建议 [0.8, 0.99]')
