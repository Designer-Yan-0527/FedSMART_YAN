"""
持续学习数据集模块

该模块提供多种数据集的自定义实现，支持持续学习和联邦学习场景。

支持的数据集：
- MNIST_RGB: MNIST 数据集的 RGB 版本
- FashionMNIST: Fashion-MNIST 数据集
- NotMNIST: NotMNIST 数据集
- SVHN: SVHN 数据集
- CUB200: CUB-200 鸟类数据集
- TinyImagenet: Tiny ImageNet 数据集
- Scene67: 室内场景数据集
- Imagenet_R: ImageNet-R 数据集
- Office_Home: Office-Home 跨域数据集
- ILSVRC: ImageNet 数据集
"""

import os
import os.path
import pathlib
from pathlib import Path
from typing import Any, Tuple
import glob
from shutil import move, rmtree

import numpy as np
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image
from tqdm import tqdm

from .dataset_utils import read_image_file, read_label_file


class MNIST_RGB(datasets.MNIST):
    """
    MNIST 数据集的 RGB 版本
    
    将原始 MNIST 的灰度图像转换为 RGB 三通道图像，便于与其他彩色数据集一起使用。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化 MNIST_RGB 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        super(MNIST_RGB, self).__init__(root, transform=transform, 
                                        target_transform=target_transform, download=download)
        self.train = train

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        """检查旧版本数据是否存在"""
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) 
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        """加载旧版本数据（向后兼容）"""
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        """加载原始数据文件"""
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionMNIST(MNIST_RGB):
    """
    Fashion-MNIST 数据集
    
    包含 10 类时尚物品的图像数据集。
    
    来源: https://github.com/zalandoresearch/fashion-mnist
    """
    
    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class NotMNIST(MNIST_RGB):
    """
    NotMNIST 数据集
    
    包含 A-J 共 10 个字母的手写字符图像。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化 NotMNIST 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')
        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')

        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN(datasets.SVHN):
    """
    SVHN 数据集
    
    Street View House Numbers 数据集，包含真实场景中的门牌号数字。
    """
    
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        """
        初始化 SVHN 数据集
        
        Args:
            root: 数据集根目录
            split: 数据集分割 ('train', 'test', 'extra')
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        super(SVHN, self).__init__(root, split=split, transform=transform, 
                                   target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """获取数据集长度"""
        return len(self.data)

    def _check_integrity(self) -> bool:
        """检查数据完整性"""
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        """下载数据集"""
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        """返回额外信息"""
        return "Split: {split}".format(**self.__dict__)


class CUB200(torch.utils.data.Dataset):
    """
    CUB-200 鸟类数据集
    
    包含 200 种鸟类的图像数据集。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        """
        初始化 CUB200 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(root)
            zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(root)
            tar_ref.close()

            self.split()
        
        if self.train:
            fpath = os.path.join(root, 'CUB_200_2011', 'train')
        else:
            fpath = os.path.join(root, 'CUB_200_2011', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        """分割数据集为训练集和测试集"""
        train_folder = self.root + 'CUB_200_2011/train'
        test_folder = self.root + 'CUB_200_2011/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + 'CUB_200_2011/images.txt'
        train_test_split = self.root + 'CUB_200_2011/train_test_split.txt'

        with open(images, 'r') as image:
            with open(train_test_split, 'r') as f:
                for line in f:
                    image_path = image.readline().split(' ')[-1]
                    image_path = image_path.replace('\n', '')
                    class_name = image_path.split('/')[0].split(' ')[-1]
                    src = self.root + 'CUB_200_2011/images/' + image_path

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)


class TinyImagenet(torch.utils.data.Dataset):
    """
    Tiny ImageNet 数据集
    
    ImageNet 的简化版本，包含 200 个类别，每个类别有 500 张训练图像和 50 张验证图像。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        """
        初始化 TinyImagenet 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)
        
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()
            self.split()

        if self.train:
            fpath = root + 'tiny-imagenet-200/train'
        else:
            fpath = root + 'tiny-imagenet-200/test'
        
        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        """将验证集转换为测试集"""
        test_folder = self.root + 'tiny-imagenet-200/test'

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(self.root + 'tiny-imagenet-200/val/images/*')
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + '/' + folder):
                os.mkdir(test_folder + '/' + folder)
                os.mkdir(test_folder + '/' + folder + '/images')
            
        
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + '/' + folder + '/images/' + file
            move(src, dst)
        
        rmtree(self.root + 'tiny-imagenet-200/val')


class Scene67(torch.utils.data.Dataset):
    """
    Scene67 室内场景数据集
    
    包含 67 种室内场景的图像数据集。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        """
        初始化 Scene67 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        train_annos_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
        test_annos_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = 'indoorCVPR_09.tar'
        self.train_annos_fname = 'TrainImage.txt'
        self.test_annos_fname = 'TestImage.txt'
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + url)
                    download_url(url, root, filename=fname)
        
        if not os.path.exists(os.path.join(root, 'Scene67')):
            import tarfile
            with tarfile.open(os.path.join(root, image_fname)) as tar:
                tar.extractall(os.path.join(root, 'Scene67'))
            self.split()

        if self.train:
            fpath = os.path.join(root, 'Scene67', 'train')
        else:
            fpath = os.path.join(root, 'Scene67', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        """分割数据集为训练集和测试集"""
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'train')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'train'))
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'test')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'test'))
        
        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'train/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0]))
                move(src, dst)
        
        with open(test_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'test/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0]))
                move(src, dst)


class Imagenet_R(torch.utils.data.Dataset):
    """
    ImageNet-R 数据集
    
    ImageNet 的渲染版本，包含 200 个类别的合成图像。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        """
        初始化 Imagenet_R 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')
        if not os.path.exists(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'imagenet-r')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
            tar_ref.extractall(root)
            tar_ref.close()
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'
        else:
            fpath = self.fpath + '/test'

        X, Y = [], []
        folders = os.listdir(fpath)

        for idx, folder in tqdm(enumerate(folders)):
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                img_path = os.path.join(folder_path, ims)
                a = np.array(Image.open(img_path).convert('RGB'))
                X.append(a)
                Y.append(idx)

        self.data = X
        self.targets = Y

    def split(self):
        """分割数据集为训练集和测试集"""
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img)
        except:
            pass
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Office_Home(torch.utils.data.Dataset):
    """
    Office-Home 跨域数据集
    
    包含 4 个域的图像数据集，用于域适应研究。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, domain=1):
        """
        初始化 Office_Home 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
            domain: 域索引 (0-3)
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw'
        self.filename = 'OfficeHomeDataset_10072016.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        fpath = os.path.join(root, 'OfficeHomeDataset_10072016', str(domain))

        X, Y = [], []
        folders = os.listdir(fpath)

        for idx, folder in enumerate(folders):
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                img_path = os.path.join(folder_path, ims)
                X.append(np.array(Image.open(img_path).convert('RGB')))
                Y.append(idx)

        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img)
        except:
            pass
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ILSVRC(torch.utils.data.Dataset):
    """
    ILSVRC (ImageNet) 数据集
    
    ImageNet Large Scale Visual Recognition Challenge 数据集。
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化 ILSVRC 数据集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 图像变换
            target_transform: 标签变换
            download: 是否下载数据集
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'ILSVRC2012')
        fpath = self.fpath.replace('\\', '/')

        X, Y = [], []
        folders = os.listdir(fpath)

        if not os.path.exists(os.path.join(root, 'ILSVRC2012')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
            tar_ref.extractall(root)
            tar_ref.close()

        for idx, folder in tqdm(enumerate(folders)):
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                img_path = os.path.join(folder_path, ims)
                a = np.array(Image.open(img_path).convert('RGB'))
                X.append(a)
                Y.append(idx)

        self.data = X
        self.targets = Y

    def split(self):
        """分割数据集为训练集和测试集"""
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))

        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取数据项
        
        Args:
            index: 数据索引
        
        Returns:
            Tuple[PIL.Image, int]: 图像和标签
        """
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img)
        except:
            pass
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
