from datetime import datetime
import easyocr
import cv2
import os
from typing import Union, List, Optional

class OCRDetector:
    def __init__(self, lang_list=['ch_sim', 'en'], gpu=True):
        """
        初始化 OCR 检测器
        
        :param lang_list: 识别的语言列表，默认中文和英文
        :param gpu: 是否使用 GPU 加速
        """
        try:
            if gpu:
                self.reader = easyocr.Reader(lang_list, gpu=True)
            else:
                self.reader = easyocr.Reader(lang_list, gpu=False)
        except:
            self.reader = easyocr.Reader(lang_list, gpu=False)
    
    def detect_image(self, 
                    image_path: str, 
                    conf: float = 0.5,) -> Union[str, List[dict]]:
        """
        OCR 文字检测 (目前好像不支持中文路径，存在路径乱码错误)
        
        :param image_path: 图片路径
        :param conf: 置信度阈值
        :return: 输出文件路径或检测结果JSON
        """
        # 读取图像
        result = self.reader.readtext(image_path)
        text = ""
        temp_text = ""  # 初始化一个临时字符串用于存储本次识别的文本
        for bbox, ocr_text, prob in result:  # 遍历识别结果
            if prob >= conf:  # 只考虑概率大于等于75%的结果
                temp_text += ocr_text + "\n"  # 将识别到的文本追加到临时字符串，并加上空格分隔
        text += temp_text + " "  # 将本次识别的文本追加到全局 text 变量中，并加上换行符分隔
        return text
    
    def _prepare_output_path(self, subdir):
        """
        准备输出路径
        """
        output_dir = os.path.join('output', subdir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _generate_timestamp_filename(self, ext):
        """
        生成带时间戳的文件名
        """
        return f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.{ext}'
    


if __name__ == '__main__': 
    # 初始化 OCR 检测器
    ocr_detector = OCRDetector()
    from pathlib import Path

    # 检测图像并输出 JSON
    ocr_results = ocr_detector.detect_image('images/20241117215009.png', conf=0.5)
    print(ocr_results)