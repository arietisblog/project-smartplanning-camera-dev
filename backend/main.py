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

class UpdateConfigRequest(BaseModel):
    config: DetectionConfig

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

@app.post("/update-config")
async def update_config(request: UpdateConfigRequest):
    """検知設定をリアルタイム更新"""
    global current_detector

    if current_detector is None:
        raise HTTPException(status_code=400, detail="検知が開始されていません")

    try:
        # 設定を更新
        current_detector.config['model']['confidence_threshold'] = request.config.confidence_threshold
        current_detector.config['detection']['object_classes'] = request.config.object_classes
        current_detector.config['counting']['line_ratio'] = request.config.line_ratio
        current_detector.config['counting']['line_angle'] = request.config.line_angle
        current_detector.config['counting']['zone_ratio'] = request.config.zone_ratio
        current_detector.config['counting']['direction'] = request.config.direction

        # カウントラインとエリアを再設定（現在の動画の解像度を使用）
        # 動画の実際の解像度を取得
        if hasattr(current_detector, 'cap') and current_detector.cap is not None:
            frame_width = int(current_detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(current_detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_detector.set_counting_line(frame_height, frame_width)
            current_detector.set_counting_zone(frame_width, frame_height)
            pass
        else:
            # 動画が読み込まれていない場合は、現在の設定を保持
            pass

        return {"message": "設定が更新されました", "config": request.config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定更新エラー: {str(e)}")

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

    # カウントをリセット
    current_detector.reset_counting()

    # カウントラインとゾーンを設定（元の解像度で設定）
    current_detector.set_counting_line(frame_height, frame_width)
    current_detector.set_counting_zone(frame_width, frame_height)

    frame_count = 0
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームレート調整: 10FPSに調整（3フレームに1回処理）
            if frame_count % 3 != 0:
                frame_count += 1
                continue

            # 元の解像度でフレームを処理（カウント表示も元解像度で実行）
            processed_frame, object_count = current_detector.process_frame(frame)

            # アスペクト比を保ちながら最大640pxにリサイズ（表示用）
            original_height, original_width = processed_frame.shape[:2]
            max_width = 640

            if original_width > max_width:
                # 幅が640pxを超える場合、アスペクト比を保ってリサイズ
                scale = max_width / original_width
                new_width = max_width
                new_height = int(original_height * scale)
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            else:
                # 幅が640px以下の場合、そのまま使用
                new_width = original_width
                new_height = original_height

            # フレームをJPEGにエンコード（品質60%）
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
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

            # 進捗情報を送信（10FPSに合わせて調整）
            if frame_count % 10 == 0:
                elapsed_time = asyncio.get_event_loop().time() - start_time
                fps_processed = frame_count / elapsed_time
                progress_message = {
                    "type": "progress",
                    "frame_count": frame_count,
                    "object_count": object_count,
                    "fps": fps_processed
                }
                await manager.broadcast(json.dumps(progress_message))

            # フレームレート制御（10FPSに固定）
            await asyncio.sleep(0.1)  # 10FPS = 0.1秒間隔

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
