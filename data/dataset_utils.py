"""
数据集工具函数模块

该模块提供数据集下载、完整性检查、文件解压等工具函数，
主要用于支持自定义数据集的加载和管理。

代码来源: https://github.com/pytorch/vision/blob/8635be94d1216f10fb8302da89233bd86445e449/torchvision/datasets/utils.py
"""

import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile
import numpy as np
import torch
import codecs

from torch.utils.model_zoo import tqdm


def gen_bar_updater():
    """
    生成进度条更新器
    
    Returns:
        function: 进度条更新函数
    """
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    """
    计算文件的 MD5 哈希值
    
    Args:
        fpath: 文件路径
        chunk_size: 读取块大小，默认 1MB
    
    Returns:
        str: MD5 哈希值
    """
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    """
    检查文件 MD5 是否匹配
    
    Args:
        fpath: 文件路径
        md5: 期望的 MD5 值
        **kwargs: 传递给 calculate_md5 的参数
    
    Returns:
        bool: MD5 是否匹配
    """
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    """
    检查文件完整性
    
    Args:
        fpath: 文件路径
        md5: 期望的 MD5 值，为 None 时仅检查文件是否存在
    
    Returns:
        bool: 文件是否完整
    """
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    创建目录，若已存在则忽略错误
    
    Args:
        dirpath: 目录路径
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """
    从 URL 下载文件到指定目录
    
    Args:
        url: 下载链接
        root: 目标目录
        filename: 保存文件名，默认为 URL 中的文件名
        md5: 期望的 MD5 值，用于验证文件完整性
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead. '
                      'Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e


def list_dir(root, prefix=False):
    """
    列出目录下所有子目录
    
    Args:
        root: 根目录路径
        prefix: 是否添加路径前缀
    
    Returns:
        List[str]: 子目录列表
    """
    root = os.path.expanduser(root)
    directories = list(filter(
        lambda p: os.path.isdir(os.path.join(root, p)),
        os.listdir(root)
    ))

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """
    列出目录下所有指定后缀的文件
    
    Args:
        root: 根目录路径
        suffix: 文件后缀，如 '.png' 或 ('.jpg', '.png')
        prefix: 是否添加路径前缀
    
    Returns:
        List[str]: 文件列表
    """
    root = os.path.expanduser(root)
    files = list(filter(
        lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
        os.listdir(root)
    ))

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """
    从 Google Drive 下载文件
    
    Args:
        file_id: Google Drive 文件 ID
        root: 目标目录
        filename: 保存文件名，默认为文件 ID
        md5: 期望的 MD5 值
    """
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()
        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    """
    从 Google Drive 响应中提取确认 token
    
    Args:
        response: HTTP 响应对象
    
    Returns:
        str: 确认 token，不存在则返回 None
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination, chunk_size=32768):
    """
    保存 HTTP 响应内容到文件
    
    Args:
        response: HTTP 响应对象
        destination: 目标文件路径
        chunk_size: 读取块大小
    """
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _is_tar(filename):
    """判断是否为 tar 文件"""
    return filename.endswith(".tar")


def _is_targz(filename):
    """判断是否为 tar.gz 文件"""
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    """判断是否为 gzip 文件（不包括 tar.gz）"""
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    """判断是否为 zip 文件"""
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    """
    解压归档文件
    
    Args:
        from_path: 归档文件路径
        to_path: 解压目标目录，默认为归档文件所在目录
        remove_finished: 是否在解压后删除原文件
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    """
    下载并解压归档文件
    
    Args:
        url: 下载链接
        download_root: 下载目录
        extract_root: 解压目标目录，默认为下载目录
        filename: 保存文件名
        md5: 期望的 MD5 值
        remove_finished: 是否在解压后删除原文件
    """
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable):
    """
    将可迭代对象转换为字符串表示
    
    Args:
        iterable: 可迭代对象
    
    Returns:
        str: 格式化后的字符串
    """
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    """
    验证字符串参数
    
    Args:
        value: 参数值
        arg: 参数名称
        valid_values: 有效值列表
        custom_msg: 自定义错误消息
    
    Returns:
        str: 验证后的参数值
    
    Raises:
        ValueError: 参数无效时抛出
    """
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value


def get_int(b):
    """
    将字节转换为整数
    
    Args:
        b: 字节数据
    
    Returns:
        int: 转换后的整数
    """
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """
    打开可能压缩的文件
    
    Args:
        path: 文件路径
    
    Returns:
        file object: 文件对象
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """
    读取 SN3 格式的张量文件
    
    Args:
        path: 文件路径或文件对象
        strict: 是否严格验证
    
    Returns:
        torch.Tensor: 读取的张量
    """
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }

    with open_maybe_compressed_file(path) as f:
        data = f.read()

    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    """
    读取标签文件
    
    Args:
        path: 文件路径
    
    Returns:
        torch.Tensor: 标签张量
    """
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    """
    读取图像文件
    
    Args:
        path: 文件路径
    
    Returns:
        torch.Tensor: 图像张量
    """
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x
