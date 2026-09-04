"""
FedTA 框架工具函数模块

提供联邦学习框架中常用的工具函数，包括：
- 准确率计算
- 知识蒸馏损失
- 数据变换
- 特征可视化
- 度量计算
- 对比学习损失
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def accuracy(output, target, topk=(1,)):
    """
    计算分类任务的 Top-k 准确率
    
    Args:
        output: 模型输出的 logits (batch_size x num_classes)
        target: 真实标签 (batch_size)
        topk: 要计算的 k 值元组，如 (1, 5)
    
    Returns:
        Tuple[List[torch.Tensor], int]: 
            第一个元素是每个 k 值对应的正确预测数列表
            第二个元素是 batch 大小
    """
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) for k in topk], batch_size


def global_distillation_loss(output, outputs):
    """
    计算融合输出与多个源输出之间的知识蒸馏损失
    
    Args:
        output: 融合后的模型输出
        outputs: 用于蒸馏的多个源模型输出列表
    
    Returns:
        torch.Tensor: 总蒸馏损失（MSE损失之和）
    """
    mse = torch.nn.MSELoss(reduction='sum')
    total_loss = 0
    for i in outputs:
        loss = mse(output, i)
        total_loss += loss
    return total_loss


def build_transform(is_train, input_size):
    """
    构建训练或评估阶段的数据变换流水线
    
    Args:
        is_train: 是否为训练阶段
        input_size: 目标图像尺寸
    
    Returns:
        torchvision.transforms.Compose: 组合变换对象
    """
    resize_im = input_size > 32
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def visualize_feature_map(img_batch, name):
    """
    可视化并保存特征图
    
    Args:
        img_batch: 输入的特征图批次
        name: 输出文件名（含路径）
    """
    feature_map = img_batch[0]
    feature_map = feature_map.reshape(feature_map.shape[1], feature_map.shape[2], -1)
    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[0]

    for i in range(num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)

    plt.savefig(name)
    plt.show()

    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


def get_row_col(num_pic):
    """
    计算显示图像的最佳网格尺寸
    
    Args:
        num_pic: 图像总数
    
    Returns:
        Tuple[int, int]: (行数, 列数)
    """
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def euclidean_metric_cal(a, b):
    """
    计算两个张量之间的欧氏距离矩阵
    
    Args:
        a: 第一个张量
        b: 第二个张量
    
    Returns:
        torch.Tensor: 欧氏距离矩阵
    """
    euc_metric = torch.empty(0, dtype=torch.long).cuda()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            euc = torch.sqrt(((a[i][j].repeat(b.shape[1], 1) - b[i]) ** 2).sum(dim=1))
            euc_metric = torch.cat([euc_metric, euc.unsqueeze(0)])
    return euc_metric.view(-1, a.shape[1], b.shape[1])


def function_cal(x, y, temperature, kccl_euc=0, ks=1, km=0):
    """
    使用欧氏距离或余弦相似度计算相似度分数
    
    Args:
        x: 输入张量
        y: 输入张量
        temperature: 温度缩放因子
        kccl_euc: 是否使用欧氏距离（1=是，0=否）
        ks: 相似度缩放因子
        km: 边缘参数
    
    Returns:
        torch.Tensor: 相似度分数
    """
    if kccl_euc == 1:
        euc_dis = euclidean_metric_cal(x, y)
        return torch.exp(torch.div(1, euc_dis * temperature))
    else:
        return torch.exp((torch.div(torch.bmm(x, y.permute(0, 2, 1)), temperature) - km) * ks)


def kccl_loss(pooled_output, labels, k, temperature, neg_num=1, weight=None, 
              loss_metric=0, neg_method=0, centroids=None, kccl_euc=0, ks=1, km=0):
    """
    知识对比持续学习（KCCL）损失函数
    
    Args:
        pooled_output: 池化后的特征表示
        labels: 真实标签
        k: 正样本数量
        temperature: 温度缩放因子
        neg_num: 负样本数量
        weight: 损失权重
        loss_metric: 损失度量类型（0=对称，1=非对称）
        neg_method: 负样本采样方法
        centroids: 类别中心
        kccl_euc: 是否使用欧氏距离
        ks: 缩放因子
        km: 边缘参数
    
    Returns:
        torch.Tensor: KCCL损失值
    """
    features = torch.cat((pooled_output, labels.unsqueeze(1)), 1)
    B, H = pooled_output.shape
    
    if neg_method in [3, 6]:
        pooled_output = pooled_output.view(-1, 1 + k + neg_num, H)
        labels = labels.view(-1, 1 + k + neg_num)
        neg_examples = pooled_output[:, (1 + k):, :]
        pooled_output = pooled_output[:, :(1 + k), :]

    pos = function_cal(pooled_output, pooled_output, temperature, kccl_euc, ks, km)
    neg1 = function_cal(pooled_output, neg_examples, temperature, kccl_euc, ks, 0)
    
    if neg_num > 1 or neg_examples.shape[1] > 1:
        neg1 = torch.sum(neg1, dim=2).unsqueeze(2)
    
    neg2 = neg1.permute(0, 2, 1).repeat(1, k + 1, 1)
    pos_neg_mask = 1 - torch.eye(pos.shape[-1], device=neg2.device).unsqueeze(0).repeat(pos.shape[0], 1, 1)
    pos_neg_mask = pos_neg_mask.float()
    
    pos_neg = torch.sum(torch.sum(pos * pos_neg_mask, dim=-1), dim=-1).unsqueeze(1).unsqueeze(1).repeat(1, *pos.shape[1:])
    pos_neg = pos_neg - pos * 2
    neg = pos + neg2 + neg2.permute(0, 2, 1) + pos_neg
    loss_a = -torch.log(torch.div(pos, neg))
    
    if loss_metric == 0:
        for i in range(loss_a.shape[1]):
            loss_a[:, i, i] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = loss_b / (pooled_output.shape[0] * k * (k + 1))
    elif loss_metric == 1:
        for i in range(loss_a.shape[1]):
            for j in range(loss_a.shape[2]):
                if i >= j:
                    loss_a[:, i, j] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = 2 * loss_b / (pooled_output.shape[0] * k * (k + 1))
    
    return loss


def CosineSimilarityClassifier(test_feature, global_protos, current_class):
    """
    使用余弦相似度与全局原型进行分类
    
    Args:
        test_feature: 待分类的特征向量
        global_protos: 类别原型字典
        current_class: 当前类别索引列表
    
    Returns:
        numpy.ndarray: 预测的类别标签
    """
    all_global_protos_keys = np.array(list(current_class))
    all_protos = [global_protos[key] for key in all_global_protos_keys]
    all_protos = torch.tensor(np.vstack(all_protos)).to('cuda')

    l = torch.cosine_similarity(test_feature, all_protos, dim=1)
    
    _, index = torch.topk(l, dim=0, k=1)
    predicted_label = all_global_protos_keys[index]

    return predicted_label
