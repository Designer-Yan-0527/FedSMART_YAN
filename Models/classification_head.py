"""
分类头模块

该模块实现了 FedTA 框架的分类头，由两层全连接神经网络和 dropout 组成，
用于提高模型的泛化能力。
"""

from copy import deepcopy

import torch
from torch import nn


class Chead(torch.nn.Module):
    """
    图像分类任务的分类头
    
    实现一个两层全连接网络：
    - 输入维度: 768 * 2 (拼接特征)
    - 隐藏层: 512 个单元，带 ReLU 激活
    - Dropout 层用于正则化
    - 输出层: 类别数量
    """

    def __init__(self, label_num):
        """
        初始化分类头
        
        Args:
            label_num: 输出类别数量
        """
        super(Chead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, label_num)
        )

    def forward(self, x):
        """
        前向传播通过分类头
        
        Args:
            x: 输入特征 (batch_size x 1536)
            
        Returns:
            分类 logits (batch_size x label_num)
        """
        x = self.head(x)
        return x

    def load_head(self, head):
        """
        加载预训练的分类头权重
        
        Args:
            head: 要加载的状态字典或 Sequential 模块
        """
        self.head = deepcopy(head)

    def get_head(self):
        """
        获取当前的分类头模块
        
        Returns:
            Sequential 分类头模块
        """
        return self.head