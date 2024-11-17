# detect_api.py
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from orc_detector import OCRDetector
from orc_detector import detect_image as ocr_detect_image
from yolo_detector import YOLODetector, detect_image, detect_video

app = FastAPI(title="Detection API")


@app.post("/ocr/detect")
async def ocr_detect(image: UploadFile = File(...)):
    """
    OCR文字检测API
    :param image: 上传的图片文件
    :return: 识别的文本
    """
    image_path = f"temp/{image.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    text = ocr_detect_image(image_path)
    os.remove(image_path)
    return JSONResponse(content={"text": text})


@app.post("/yolo/detect_image")
async def yolo_detect_image(
    image: UploadFile = File(...),
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "json",
):
    """
    YOLO目标检测API
    :param image: 上传的图片文件
    :param model_type: 模型类型 ('v10' 或 'world')
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式 ('image', 'json')
    :return: 检测结果
    """
    image_path = f"temp/{image.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    result = detect_image(image_path, model_type, conf, classes, output_format)
    os.remove(image_path)

    if output_format == "image":
        return FileResponse(result)

    return JSONResponse(content=result)


@app.post("/yolo/detect_video")
async def yolo_detect_video(
    video: UploadFile = File(...),
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "json",
):
    """
    YOLO视频检测API
    :param video: 上传的视频文件
    :param model_type: 模型类型 ('v10' 或 'world')
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式 ('video', 'json')
    :return: 检测结果
    """
    video_path = f"temp/{video.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())

    result = detect_video(video_path, model_type, conf, classes, output_format)
    os.remove(video_path)

    if output_format == "video":
        return FileResponse(result)
    return JSONResponse(content=result)


def is_image_file(filename: str) -> bool:
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def is_video_file(filename: str) -> bool:
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    return any(filename.lower().endswith(ext) for ext in video_extensions)


@app.post("/yolo/detect")
async def yolo_detect(
    file: UploadFile = File(...),
    model_type: str = "v10",
    conf: float = 0.5,
    classes: Optional[List[str]] = None,
    output_format: str = "json",
):
    """
    通用检测API，根据文件类型自动选择图片或视频检测
    :param file: 上传的文件（图片或视频）
    :param model_type: 模型类型 ('v10' 或 'world citizens')
    :param conf: 置信度阈值
    :param classes: 要检测的类别
    :param output_format: 输出格式 ('image', 'video', 'json')
    """
    if is_image_file(file.filename):
        return await yolo_detect_image(file, model_type, conf, classes, output_format)
    elif is_video_file(file.filename):
        return await yolo_detect_video(file, model_type, conf, classes, output_format)
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
