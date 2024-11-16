import os
import json
import datetime
import cv2
import concurrent.futures
import torch
from ultralytics import YOLOv10, YOLOWorld
from typing import List, Union, Optional

class YOLODetector:
    def __init__(self, 
                model_type: str = 'v10',
                device=None):
        """
        初始化目标检测器
        :param model_type: 模型类型 ('v10' 或 'world')
        :param device: 设备选择（cuda/cpu）
        """
        
        self.load_model(model_type, device)
        
    
    def load_model(self, model_type: str = 'v10', device=None):
        """
        加载模型
        
        Args:
        :param model_type: 模型类型 ('v10' 或 'world')
        :param device: 设备选择（cuda/cpu）
        """
        # 设备选择逻辑
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # 尝试使用指定设备
            device = torch.device(device)
        except:
            # 如果指定设备失败，默认使用CPU
            device = torch.device('cpu')
        
        self.model_type = model_type
        
        try:
            if model_type == 'v10':
                self.model = YOLOv10('./models/yolov10x.pt').to(device)
            elif model_type == 'world':
                self.model = YOLOWorld('./models/yolov8x-worldv2.pt').to(device)
            else:
                raise ValueError("Invalid model type. Choose 'v10' or 'world'.")
            
            self.names = self.model.names
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def _prepare_output_path(self, output_type: str) -> str:
        """
        准备输出路径
        
        :param output_type: 输出类型 ('image' 或 'video')
        :return: 输出路径
        """
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        output_path = os.path.join('output', output_type, today)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def _generate_timestamp_filename(self, extension: str) -> str:
        """
        生成带时间戳的文件名
        
        :param extension: 文件扩展名
        :return: 文件名
        """
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f'{current_time}.{extension}'

    def detect_image(self, 
                     image_path: str, 
                     conf: float = 0.5, 
                     classes: Optional[List[str]] = None, 
                     output_format: str = 'image') -> Union[str, dict]:
        """
        检测单张图片
        
        :param image_path: 图片路径
        :param conf: 置信度阈值
        :param classes: 要检测的类别
        :param output_format: 输出格式 ('image', 'json')
        :return: 输出文件路径或检测结果JSON
        """
        # 如果是世界模型且指定了类别，需要设置
        if self.model_type == 'world' and classes:
            self.model.set_classes(classes)
        
        if self.model_type == "world" and classes is None:
            self.model.set_classes(self.names)

        # 检测
        results = self.model(image_path, conf=conf)
        
        # 根据输出格式处理结果
        if output_format == 'image':
            output_path = self._prepare_output_path('image')
            output_filename = self._generate_timestamp_filename('jpg')
            output_file = os.path.join(output_path, output_filename)
            
            # 绘制并保存图像
            annotated_img = results[0].plot()
            cv2.imwrite(output_file, annotated_img)
            return output_file
        
        elif output_format == 'json':
            # 提取检测结果
            detections = results[0].boxes.data.tolist()
            json_data = []
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                json_data.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
            return json_data

    def detect_images(self, 
                      image_paths: List[str], 
                      conf: float = 0.5, 
                      classes: Optional[List[str]] = None, 
                      output_format: str = 'image',
                      max_workers: int = None) -> List[Union[str, dict]]:
        """
        并行检测多张图片
        
        :param image_paths: 图片路径列表
        :param conf: 置信度阈值
        :param classes: 要检测的类别
        :param output_format: 输出格式
        :param max_workers: 并行工作线程数
        :return: 输出结果列表
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda path: self.detect_image(path, conf, classes, output_format), 
                image_paths
            ))
        return results

    def detect_video(self, 
                     video_path: str, 
                     conf: float = 0.5, 
                     classes: Optional[List[str]] = None, 
                     output_format: str = 'video') -> Union[str, List[dict]]:
        """
        检测视频
        
        :param video_path: 视频路径
        :param conf: 置信度阈值
        :param classes: 要检测的类别, 只有world模型起作用
        :param output_format: 输出格式 ('video', 'json')
        :return: 输出文件路径或检测结果JSON
        """
        # 如果是世界模型且指定了类别，需要设置
        if self.model_type == 'world' and classes:
            self.model.set_classes(classes)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 准备输出
        if output_format == 'video':
            output_path = self._prepare_output_path('video')
            output_filename = self._generate_timestamp_filename('mp4')
            output_file = os.path.join(output_path, output_filename)
            
            # 视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            # 处理视频
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=conf, classes=classes)
                annotated_frame = results[0].plot()
                out_video.write(annotated_frame)
                frame_count += 1

            cap.release()
            out_video.release()
            return output_file

        elif output_format == 'json':
            json_data = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 计算当前帧的时间戳
                current_timestamp = frame_count / fps  # 秒为单位
                
                # 将秒转换为 HH:MM:SS 格式
                timestamp_str = str(datetime.timedelta(seconds=int(current_timestamp)))

                results = self.model(frame, conf=conf, classes=classes)
                detections = results[0].boxes.data.tolist()
                
                frame_detections = {
                    'frame_number': frame_count,
                    'timestamp': timestamp_str,  # 视频的实际时间戳
                    'time_in_seconds': current_timestamp,  # 可选：秒数表示
                    'detections': []
                }

                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    frame_detections['detections'].append({
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.model.names[int(cls)]
                    })
                
                json_data.append(frame_detections)
                frame_count += 1

            cap.release()
            return json_data

# 使用示例
def main():
    # 创建检测器
    detector = YOLODetector(model_type='standard')

    # 检测单张图片
    result = detector.detect_image('images/image.jpg', conf=0.5, output_format='json')
    print(result)

    # 检测多张图片
    images = ['images/image1.jpg', 'images/image2.jpg', 'images/image3.jpg']
    results = detector.detect_images(images, conf=0.5)

    # 检测视频
    video_result = detector.detect_video('video.mp4', output_format='json')
    print(video_result)

if __name__ == '__main__':
    main()