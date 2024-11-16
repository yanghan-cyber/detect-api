import datetime
import json
from ultralytics import YOLOv10
from tqdm import tqdm
import cv2
import os
from ultralytics import YOLOWorld

model = YOLOv10.from_pretrained('jameslahm/yolov10x')
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'

model_word = YOLOWorld("./model/yolov8x-worldv2.pt")


def load_yolov10_model(model_path='jameslahm/yolov10x'):
    """
    加载 YOLOv10 模型
    
    Args:
        model_path (str): 模型的路径,默认为 'jameslahm/yolov10x'
    
    Returns:
        YOLO: 已加载的 YOLOv10 模型
    """
    return YOLOv10.from_pretrained(model_path)


def detect_image(model, image_path, output_dir='output', confidence_threshold=0.5):
    """
    对单张图像进行目标检测并保存结果
    
    Args:
        model (YOLO): 已加载的 YOLOv10 模型
        image_path (str): 图像文件路径
        output_dir (str): 输出文件夹,默认为 'output'
    
    Returns:
        str: 保存的图像文件路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 进行目标检测
    results = model(img, conf=confidence_threshold)
    annotated_img = results[0].plot()

    # 创建当日日期文件夹
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, 'image', today)
    os.makedirs(output_path, exist_ok=True)

    # 生成带有时间戳的文件名
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_image_path = os.path.join(output_path, f'{current_time}.jpg')

    # 保存带有检测结果的图像
    cv2.imwrite(output_image_path, annotated_img)

    return output_image_path

def detect_video(model, video_path, output_dir='output', confidence_threshold=0.5):
    """
    对视频进行目标检测并保存结果
    
    Args:
        model (YOLO): 已加载的 YOLOv10 模型
        video_path (str): 视频文件路径
        output_dir (str): 输出文件夹,默认为 'output'
        confidence_threshold (float): 置信度阈值，默认为 0.5
    """
    # 创建当日日期文件夹
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, 'video', today)
    os.makedirs(output_path, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video_path = os.path.join(output_path, f'{current_time}.mp4')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 使用目标跟踪处理整个视频
    results = model.track('video.mp4', conf=confidence_threshold, stream=True)

    # 将检测结果写入视频
    for result in results:
        annotated_frame = result.plot()  # 绘制检测结果在帧上
        out_video.write(annotated_frame)

    # 释放资源
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print(f"处理完成，视频已保存至 {output_video_path}")

def detect_image_and_get_json(model, image_path):
    """
    对单张图像进行目标检测,并返回检测结果的 JSON 格式
    
    Args:
        model (YOLO): 已加载的 YOLOv10 模型
        image_path (str): 图像文件路径
    
    Returns:
        str: 检测结果的 JSON 字符串
    """
    img = cv2.imread(image_path)
    results = model(img)
    
    # 提取检测结果
    detections = results[0].boxes.data.tolist()
    
    # 将检测结果转换为 JSON 格式
    json_data = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        json_data.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': conf,
            'class': int(cls),
            'class_name': model.names[int(cls)]
        })
    
    return json.dumps(json_data)



# 加载模型
model = load_yolov10_model().to('cuda')

# 检测单张图像
# annotated_img = detect_image(model, 'image.jpg')

# cv2.imshow('img show', annotated_img)
# cv2.waitKey(0)

detect_video(model, 'video.mp4')