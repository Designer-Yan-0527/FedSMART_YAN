"""
CIFAR100 数据集包装模块

该模块提供了 CIFAR100 数据集的自定义包装类，支持按类别选取训练和测试数据，
适用于联邦学习场景中的数据分割。
"""

from typing import TypeVar

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class iCIFAR100c(object):
    """
    CIFAR100 数据集包装类
    
    支持按类别选取训练和测试数据，提供自定义的数据加载接口。
    """
    
    def __init__(self, subset):
        """
        初始化 iCIFAR100c 实例
        
        Args:
            subset: 数据集子集对象，可以是 Subset 或原始 Dataset
        """
        super(iCIFAR100c, self).__init__()
        if hasattr(subset, 'dataset'):
            subset = subset.dataset

        self.data = np.array(subset.data)
        self.targets = subset.targets
        self.trans = build_transform(True, 224)

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        """
        拼接多个数据和标签数组
        
        Args:
            datas: 数据数组列表
            labels: 标签数组列表
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 拼接后的数据和标签
        """
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        """
        获取指定类别范围的测试数据
        
        Args:
            classes: 类别范围，格式为 [start, end]
        """
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes):
        """
        获取指定类别的训练数据
        
        Args:
            classes: 类别列表
        """
        datas, labels = [], []
        for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        """
        获取训练数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[int, torch.Tensor, int]: 索引、变换后的图像、标签
        """
        img, target = self.TrainData[index], self.TrainLabels[index]
        img = self.trans(Image.fromarray(img).convert('RGB'))
        return index, img, target

    def getTestItem(self, index):
        """
        获取测试数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[int, np.ndarray, int]: 索引、原始图像、标签
        """
        img, target = self.TestData[index], self.TestLabels[index]
        return index, img, target

    def __getitem__(self, index):
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[int, torch.Tensor, int] 或 Tuple[int, np.ndarray, int]
        """
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        """
        获取数据集长度
        
        Returns:
            int: 数据集大小
        """
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        """
        获取指定类别的所有图像
        
        Args:
            label: 类别标签
        
        Returns:
            np.ndarray: 指定类别的图像数据
        """
        return self.data[np.array(self.targets) == label]
