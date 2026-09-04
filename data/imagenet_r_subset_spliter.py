"""
ImageNet-R 数据集分割模块

提供 ImageNet-R 数据集的联邦学习分割功能，支持私有类别和公共类别的划分，
以及基于 Dirichlet 分布的数据分配。
"""

import random
from typing import TypeVar, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

from data.continual_datasets import Imagenet_R, ILSVRC
from utils import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class ImagenetR_spliter:
    """
    ImageNet-R 数据集分割器
    
    将 ImageNet-R 数据集分割为多个客户端的子集，支持私有类别和公共类别的混合。
    """
    
    def __init__(self, client_num, task_num, private_class_num, input_size):
        """
        初始化数据分割器
        
        Args:
            client_num: 客户端数量
            task_num: 任务数量
            private_class_num: 每个客户端的私有类别数量
            input_size: 图像输入尺寸
        """
        self.client_num = client_num
        self.task_num = task_num
        self.private_class_num = private_class_num
        self.input_size = input_size

    def random_split(self):
        """
        随机分割数据集为多个客户端子集
        
        将类别划分为私有类别和公共类别，私有类别仅分配给特定客户端，
        公共类别通过 Dirichlet 分布分配给所有客户端。
        
        Returns:
            Tuple[List[List[CustomedSubset]], List[List[List[int]]]]:
                - 客户端数据集子集列表
                - 客户端类别掩码列表
        """
        trans = build_transform(True, self.input_size)
        self.Imagenet_R = Imagenet_R(root='./local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

        class_counts = torch.zeros(200)
        class_label = []
        for i in range(200):
            class_label.append([])
        j = 0
        xs = []
        js = []
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
            if label == 72:
                js.append(j)
                xs.append(x)

        shown_sample = xs[len(xs) - 1] if xs else None

        total_private_class_num = self.client_num * self.private_class_num
        public_class_num = 200 - total_private_class_num
        class_public = [i for i in range(200)]
        class_public = set(class_public)
        class_public.remove(72)
        class_public = list(class_public)

        class_p = random.sample(class_public, total_private_class_num - 1)
        class_public = list(set(class_public) - set(class_p))

        class_private = [
            class_p[self.private_class_num * i : self.private_class_num * i + self.private_class_num] 
            for i in range(self.client_num)
        ]
        for i in range(self.client_num):
            if i == 0:
                tem = class_private[i][0]
                class_private[i][0] = 72
                class_public.append(tem)
            class_private[i].extend(class_public)

        client_subset = [[] for _ in range(self.client_num)]
        client_mask = [[] for _ in range(self.client_num)]

        class_every_task = int((public_class_num + self.private_class_num) / self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        for i in range(self.client_num):
            for j in range(self.task_num):
                index = []
                class_this_task = class_private[i][j * class_every_task: j * class_every_task + class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_this_task:
                    if k in class_public:
                        length = int(int(class_counts[k]) * dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < length:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                        class_label[k] = unused_indice
                    else:
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, trans, shown_sample))

        return client_subset, client_mask

    def process_testdata(self, surrogate_num):
        """
        处理测试数据，生成代理数据和测试数据
        
        Args:
            surrogate_num: 每个类别选取的代理数据数量
        
        Returns:
            Tuple[CustomedSubset, CustomedSubset]: 代理数据集和测试数据集
        """
        trans = build_transform(False, self.input_size)
        self.Imagenet_R_test = Imagenet_R(root='./local_datasets', train=False, download=True)
        testset = self.Imagenet_R_test

        class_counts = torch.zeros(200)
        class_label = []
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        surro_index = []
        test_index = []
        for i in tqdm(range(200)):
            q = 0
            unused_indice = set(class_label[i])
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))

        surrodata = CustomedSubset(testset, surro_index, trans, None)
        testdata = CustomedSubset(testset, test_index, trans, None)
        return surrodata, testdata

    def random_split_synchron(self):
        """
        同步分割数据集（所有客户端共享相同的任务类别划分）
        
        Returns:
            Tuple[List[List[CustomedSubset]], List[List[List[int]]]]:
                - 客户端数据集子集列表
                - 客户端类别掩码列表
        """
        trans = build_transform(True, self.input_size)
        self.Imagenet_R = Imagenet_R(root='./local_datasets', train=True, download=True)
        trainset = self.Imagenet_R

        class_counts = torch.zeros(200)
        class_label = []
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        class_public = [i for i in range(200)]

        client_subset = [[] for _ in range(self.client_num)]
        client_mask = [[] for _ in range(self.client_num)]

        class_every_task = 10
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        for i in range(self.client_num):
            for j in range(self.task_num):
                index = []
                class_this_task = class_public[j * class_every_task: j * class_every_task + class_every_task]
                client_mask[i].append(class_this_task)

                for k in class_this_task:
                    length = int(int(class_counts[k]) * 0.8)
                    unused_indice = set(class_label[k])
                    q = 0
                    while q < length:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        unused_indice.remove(random_index)
                        q += 1
                    random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset, index, trans, None))

        return client_subset, client_mask

    def process_ILSVRC(self, surrogate_num):
        """
        处理 ILSVRC 数据集作为测试数据
        
        Args:
            surrogate_num: 每个类别选取的代理数据数量
        
        Returns:
            Tuple[CustomedSubset, CustomedSubset]: 代理数据集和测试数据集
        """
        trans = build_transform(False, self.input_size)
        self.ILSVRC = ILSVRC(root='D:/datasets', train=False, download=True)
        testset = self.ILSVRC

        class_counts = torch.zeros(200)
        class_label = []
        for i in range(200):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        surro_index = []
        test_index = []
        for i in tqdm(range(200)):
            q = 0
            unused_indice = set(class_label[i])
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))

        surrodata = CustomedSubset(testset, surro_index, trans, None)
        testdata = CustomedSubset(testset, test_index, trans, None)
        return surrodata, testdata


class CustomedSubset(Dataset[T_co]):
    """
    自定义数据集子集类
    
    支持数据预处理和变换，适用于联邦学习场景。
    """
    
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans, show_sample) -> None:
        """
        初始化自定义子集
        
        Args:
            dataset: 原始数据集
            indices: 子集索引列表
            trans: 数据变换函数
            show_sample: 展示用的样本图像
        """
        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform_pretrain = trans
        self.show_sample = show_sample

        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        self.targets = np.array(self.targets)
        self.transform = trans
        self.target_transform = None

    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
        
        Returns:
            Tuple[torch.Tensor, int]: 变换后的图像和标签
        """
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_pre, target

    def get_test_sample(self):
        """
        获取展示用的样本图像
        
        Returns:
            PIL.Image: 样本图像
        """
        img = self.show_sample
        if self.transform_pretrain is not None:
            img_pre = self.transform_pretrain(img)
        return img

    def __len__(self):
        """
        获取子集长度
        
        Returns:
            int: 子集大小
        """
        return len(self.indices)