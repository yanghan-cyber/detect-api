import concurrent.futures
import datetime
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import torch
from ultralytics import YOLOv10, YOLOWorld

from utils import generate_timestamp_filename, prepare_output_path


class YOLOModel:
    """
    YOLO模型管理器，支持模型的缓存、自动加载和超时释放
    """
    _instance = None
    _model_cache = {}
    DEFAULT_TIMEOUT = 600  # 默认超时时间：10分钟

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._last_access_time = {}

    def _get_device(self):
        """
        获取可用的设备（CUDA或CPU）
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(
        self, 
        model_type: str = "v10", 
        model_path: Optional[str] = None, 
        device: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        加载或获取缓存的模型
        
        :param model_type: 模型类型 ('v10' 或 'world')
        :param model_path: 自定义模型路径
        :param device: 设备选择（cuda/cpu）
        :param timeout: 模型缓存超时时间
        :return: 加载的模型
        """
        # 如果未指定设备，使用自动检测
        device = device or self._get_device()

        # 生成唯一的模型键
        model_key = f"{model_type}_{model_path or 'default'}"

        # 检查是否有缓存的模型
        current_time = time.time()
        if model_key in self._model_cache:
            # 更新最后访问时间
            self._last_access_time[model_key] = current_time
            return self._model_cache[model_key]

        try:
            # 加载模型
            if model_type == "v10":
                model_path = model_path or "./models/yolov10x.pt"
                model = YOLOv10(model_path).to(device)
            elif model_type == "world":
                model_path = model_path or "./models/yolov8x-worldv2.pt"
                model = YOLOWorld(model_path).to(device)
            else:
                raise ValueError(f"Invalid model type. Choose 'v10' or 'world', you give {model_type}")

            # 缓存模型并记录最后访问时间
            self._model_cache[model_key] = model
            self._last_access_time[model_key] = current_time

            return model

        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def cleanup_models(self, timeout: int = DEFAULT_TIMEOUT):
        """
        清理超时的模型
        
        :param timeout: 模型超时时间（秒）
        """
        current_time = time.time()
        expired_keys = [
            key for key, last_access in self._last_access_time.items()
            if current_time - last_access > timeout
        ]

        for key in expired_keys:
            del self._model_cache[key]
            del self._last_access_time[key]
            print(f"Model {key} has been unloaded due to timeout")

    def __del__(self):
        # 释放所有模型
        self._model_cache.clear()
        self._last_access_time.clear()
        
        

_yolo_manager = YOLOModel()


def detect_image(
    image_path: str,
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "image",
) -> Union[str, dict]:
    """
    检测单张图片

    :param image_path: 图片路径
    :param model: 加载的模型
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式 ('image', 'json')
    :return: 输出文件路径或检测结果JSON
    """
    model = _yolo_manager.load_model(model_type)

    if isinstance(model, YOLOWorld) and classes:
        model.set_classes(classes)

    results = model(image_path, conf=conf)

    if output_format == "image":
        output_path = prepare_output_path("image")
        output_filename = generate_timestamp_filename(Path(image_path).name)
        output_file = os.path.join(output_path, output_filename)

        annotated_img = results[0].plot()
        cv2.imwrite(output_file, annotated_img)
        return output_file

    elif output_format == "json":
        detections = results[0].boxes.data.tolist()
        json_data = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            json_data.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": model.names[int(cls)],
                }
            )
        return json_data
    
def detect_images(
    image_paths: List[str],
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "image",
    max_workers: int = None,
) -> List[Union[str, dict]]:
    """
    并行检测多张图片

    :param image_paths: 图片路径列表
    :param model: 加载的模型
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式
    :param max_workers: 并行工作线程数
    :return: 输出结果列表
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda path: detect_image(path, model_type, conf, classes, output_format),
                image_paths,
            )
        )
    return results


def detect_video(
    video_path: str,
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "video",
) -> Union[str, List[dict]]:
    """
    检测视频

    :param video_path: 视频路径
    :param model: 加载的模型
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式 ('video', 'json')
    :return: 输出文件路径或检测结果JSON
    """

    model = _yolo_manager.load_model(model_type)
    
    if isinstance(model, YOLOWorld) and classes:
        model.set_classes(classes)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_format == "video":
        output_path = prepare_output_path("video")
        output_filename = generate_timestamp_filename(Path(video_path).name)

        output_file = os.path.join(output_path, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 使用 detect_image 函数检测当前帧
            result = detect_image(frame, model_type, conf, classes, output_format="image")
            annotated_frame = cv2.imread(result)
            out_video.write(annotated_frame)
            frame_count += 1

        cap.release()
        out_video.release()
        return output_file

    elif output_format == "json":
        json_data = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_timestamp = frame_count / fps
            timestamp_str = str(datetime.timedelta(seconds=int(current_timestamp)))

            # 使用 detect_image 函数检测当前帧
            result = detect_image(frame, model_type, conf, classes, output_format="json")
            frame_detections = {
                "frame_number": frame_count,
                "timestamp": timestamp_str,
                "time_in_seconds": current_timestamp,
                "detections": result,
            }

            json_data.append(frame_detections)
            frame_count += 1

        cap.release()
        return json_data


# 使用示例
def main():
    model = 'v10'

    result = detect_image("images/image.jpg", model, conf=0.5, output_format="json")
    print(result)

    images = ["images/image1.jpg", "images/image2.jpg", "images/image3.jpg"]
    results = detect_images(images, model, conf=0.5, output_format="image")
    
    video_result = detect_video("images/video.mp4", model, output_format="json")
    with open(f"video_detections.json", "w") as f:
        json.dump(video_result, f, indent=4)


if __name__ == "__main__":
    main()
