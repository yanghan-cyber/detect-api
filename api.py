# api.py
import os
import io
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

import uvicorn

from orc_detector import OCRDetector
from yolo_detector import YOLODetector

app = FastAPI(title="YOLO Detection API")

# 全局模型管理
current_model = None
current_model_type = None

def load_model(model_type: str):
    global current_model, current_model_type
    
    # 如果已经是当前模型类型，直接返回
    if current_model and current_model_type == model_type:
        return current_model
    
    # 如果已有模型，先卸载
    if current_model:
        del current_model
    
    
    current_model = YOLODetector(
        model_type=model_type
    )
    current_model_type = model_type
    return current_model

def is_image_file(filename: str) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def is_video_file(filename: str) -> bool:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

            
# 其他方法保持不变，只需要将 content_type 替换为文件后缀名判断
@app.post("/yolo/detect")
async def detect_yoloworld(
    file: UploadFile = File(...), 
    conf: float = Form(default=0.3),
    output_format: str = Form(default="json"),
    classes: Optional[str] = Form(default=None)  # 改为可选的字符串输入
):
    try:
        # 解析 classes 字符串为列表
        classes_list = None
        if classes:
            classes_list = [c.strip() for c in classes.split(",")]
                        
            # 加载 YOLO World 模型
            model = load_model('world')
        else:
            model = load_model('v10')

        
        # 保存上传的文件
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 根据文件后缀判断类型
        if is_image_file(file.filename):
            if output_format not in ['json', 'image']:
                raise HTTPException(status_code=400, detail="output_format: json | image") 

            result = model.detect_image(
                file_path, 
                conf=conf, 
                classes=classes_list, 
                output_format=output_format
            )
        elif is_video_file(file.filename):
            if output_format not in ['json', 'video']:
                raise HTTPException(status_code=400, detail="output_format: json | video") 
            result = model.detect_video(
                file_path, 
                conf=conf, 
                classes=classes_list, 
                output_format=output_format
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # 根据输出格式返回
        if output_format == 'json':
            return JSONResponse(content=result)
        else:
            return FileResponse(result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 删除临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/yolo/names")
def get_yolov10_names():
    model = load_model('v10')
    return JSONResponse(content=model.model.names)


# 初始化 OCR 检测器
ocr_detector = OCRDetector()

@app.post("/ocr/detect")
async def detect_ocr(
                    file: UploadFile = File(...), 
                    conf: float = Form(default=0.3)):
    """
    上传图像进行OCR检测
    :param file: 上传的图像文件
    :param request: 包含置信度的请求体
    :return: 检测到的文本
    """
    try:
        if not is_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # 保存文件到临时路径
        temp_file_path = f"temp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 使用 OCR 检测图像
        ocr_results = ocr_detector.detect_image(temp_file_path, conf=conf)

        # 清理临时文件
        os.remove(temp_file_path)

        return {"text": ocr_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)