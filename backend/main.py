from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
import asyncio
import base64
from typing import Dict, List, Optional
import os
from camera_detection.configurable_detector import ConfigurableObjectDetector
import tempfile
import uuid
from pydantic import BaseModel

app = FastAPI(title="Object Detection API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],  # NextJSのデフォルトポート
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket接続管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 接続が切れている場合は削除
                self.active_connections.remove(connection)

manager = ConnectionManager()

# 設定モデル
class DetectionConfig(BaseModel):
    object_classes: Dict[str, str]
    confidence_threshold: float = 0.6
    line_ratio: float = 0.5
    line_angle: float = 0.0
    zone_ratio: float = 0.9
    direction: str = "both"

class StartDetectionRequest(BaseModel):
    config: DetectionConfig
    file_path: str

# グローバル変数
current_detector: Optional[ConfigurableObjectDetector] = None
processing_task: Optional[asyncio.Task] = None

@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """動画ファイルをアップロード"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="動画ファイルをアップロードしてください")

    # 一時ファイルに保存
    file_id = str(uuid.uuid4())
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"file_id": file_id, "file_path": file_path, "filename": file.filename}

@app.post("/start-detection")
async def start_detection(request: StartDetectionRequest):
    """検知処理を開始"""
    global current_detector, processing_task

    if processing_task and not processing_task.done():
        processing_task.cancel()

    # 設定ファイルを更新
    config_dict = {
        "input": {
            "video_path": request.file_path,
            "output_directory": "outputs"
        },
        "model": {
            "path": "yolov8n.pt",
            "confidence_threshold": request.config.confidence_threshold
        },
        "detection": {
            "object_classes": request.config.object_classes,
            "tracking_history_frames": 10
        },
        "tracking": {
            "max_disappeared_frames": 30,
            "max_distance": 100,
            "min_confidence": 0.5
        },
        "counting": {
            "line_angle": request.config.line_angle,
            "line_ratio": request.config.line_ratio,
            "zone_ratio": request.config.zone_ratio,
            "direction": request.config.direction
        },
        "display": {
            "show_video": False,
            "save_screenshots": False,
            "progress_interval": 30,
            "colors": {
                "counted_object": [0, 255, 0],
                "uncounted_object": [0, 0, 255],
                "counting_line": [255, 0, 0],
                "counting_zone": [255, 255, 0],
                "object_count_text": [128, 0, 0]
            }
        },
        "output": {
            "save_video": False,
            "video_codec": "mp4v",
            "screenshot_format": "jpg"
        }
    }

    # 設定ファイルを保存
    with open("temp_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    try:
        # 検知器を作成
        current_detector = ConfigurableObjectDetector("temp_config.json")

        # 非同期で動画処理を開始
        processing_task = asyncio.create_task(process_video_async(request.file_path))

        return {"message": "検知処理を開始しました", "file_path": request.file_path}
    except Exception as e:
        print(f"検知器の作成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"検知器の作成に失敗しました: {str(e)}")

@app.post("/stop-detection")
async def stop_detection():
    """検知処理を停止"""
    global processing_task

    if processing_task and not processing_task.done():
        processing_task.cancel()
        return {"message": "検知処理を停止しました"}

    return {"message": "検知処理は実行されていません"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続"""
    await manager.connect(websocket)
    try:
        while True:
            # クライアントからのメッセージを待機
            data = await websocket.receive_text()
            # 必要に応じてクライアントからのメッセージを処理
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_video_async(video_path: str):
    """非同期で動画処理を実行"""
    global current_detector

    if not current_detector:
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await manager.broadcast(json.dumps({
            "type": "error",
            "message": f"動画ファイル '{video_path}' を開けませんでした"
        }))
        return

    # 動画の情報を取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # カウントラインとゾーンを設定
    current_detector.set_counting_line(frame_height, frame_width)
    current_detector.set_counting_zone(frame_width, frame_height)

    frame_count = 0
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを処理
            processed_frame, object_count = current_detector.process_frame(frame)

            # フレームをJPEGにエンコード
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # WebSocketでフレームとカウント情報を送信
            message = {
                "type": "frame",
                "frame": frame_base64,
                "object_count": object_count,
                "frame_count": frame_count,
                "fps": fps
            }

            await manager.broadcast(json.dumps(message))

            frame_count += 1

            # 進捗情報を送信
            if frame_count % 30 == 0:
                elapsed_time = asyncio.get_event_loop().time() - start_time
                fps_processed = frame_count / elapsed_time
                progress_message = {
                    "type": "progress",
                    "frame_count": frame_count,
                    "object_count": object_count,
                    "fps": fps_processed
                }
                await manager.broadcast(json.dumps(progress_message))

            # フレームレート制御
            await asyncio.sleep(1.0 / fps)

    except asyncio.CancelledError:
        # 処理がキャンセルされた場合
        pass
    finally:
        cap.release()

        # 完了メッセージを送信
        final_message = {
            "type": "complete",
            "total_frames": frame_count,
            "final_count": current_detector.object_count,
            "processing_time": asyncio.get_event_loop().time() - start_time
        }
        await manager.broadcast(json.dumps(final_message))

@app.get("/get-classes")
async def get_classes():
    """利用可能なクラス一覧を取得"""
    classes = {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        "3": "motorcycle",
        "4": "airplane",
        "5": "bus",
        "6": "train",
        "7": "truck",
        "8": "boat",
        "9": "traffic light",
        "10": "fire hydrant",
        "11": "stop sign",
        "12": "parking meter",
        "13": "bench",
        "14": "bird",
        "15": "cat",
        "16": "dog",
        "17": "horse",
        "18": "sheep",
        "19": "cow",
        "20": "elephant",
        "21": "bear",
        "22": "zebra",
        "23": "giraffe"
    }
    return classes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
