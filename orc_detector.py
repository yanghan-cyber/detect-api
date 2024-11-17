from datetime import datetime
from typing import List, Optional, Union

import easyocr


def initialize_ocr_reader(
    lang_list: List[str] = ["ch_sim", "en"], gpu: bool = True
) -> easyocr.Reader:
    """
    初始化 OCR 阅读器

    :param lang_list: 识别的语言列表，默认中文和英文
    :param gpu: 是否使用 GPU 加速
    :return: 初始化的 OCR 阅读器
    """
    try:
        return easyocr.Reader(lang_list, gpu=gpu)
    except:
        return easyocr.Reader(lang_list, gpu=False)


def detect_image(image_path: str, conf: float = 0.5) -> str:
    """
    OCR 文字检测

    :param image_path: 图片路径
    :param reader: 初始化的 OCR 阅读器
    :param conf: 置信度阈值
    :return: 识别的文本
    """
    reader = initialize_ocr_reader()
    result = reader.readtext(image_path)
    text = ""
    for bbox, ocr_text, prob in result:
        if prob >= conf:
            text += ocr_text + "\n"
    return text


if __name__ == "__main__":

    # 检测图像并输出文本
    ocr_results = detect_image("images/20241117215009.png", conf=0.5)
    print(ocr_results)
