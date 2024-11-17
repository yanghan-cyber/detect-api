import os
from datetime import datetime


def prepare_output_path(subdir: str) -> str:
    """
    准备输出路径

    :param subdir: 子目录名称
    :return: 输出目录路径
    """
    output_dir = os.path.join('output', subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_timestamp_filename(filename: str) -> str:
    """
    生成带时间戳的文件名

    :param filename: 当前文件名
    :return: 带时间戳的文件名
    """
    base_name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'{base_name}_{timestamp}{ext}'