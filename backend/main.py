from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import cv2
import numpy as np
import json
import asyncio
import base64
import subprocess
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
            "save_video": True,
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

    print(f"process_video_async開始: {video_path}")

    if not current_detector:
        print("エラー: current_detectorがNoneです")
        return

    print("動画ファイルを開いています...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした")
        await manager.broadcast(json.dumps({
            "type": "error",
            "message": f"動画ファイル '{video_path}' を開けませんでした"
        }))
        return

    # 動画の情報を取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"動画情報: {frame_width}x{frame_height}, FPS: {fps}")

    # カウントをリセット
    print("カウントをリセット中...")
    current_detector.reset_counting()

    # カウントラインとゾーンを設定（元の解像度で設定）
    print("カウントラインとゾーンを設定中...")
    current_detector.set_counting_line(frame_height, frame_width)
    current_detector.set_counting_zone(frame_width, frame_height)

    # 動画ライターをセットアップ
    # video_pathからファイルIDを抽出
    video_filename = os.path.basename(video_path)
    file_id = video_filename.split('_')[0]  # ファイルID部分を取得
    output_path = f"outputs/{file_id}_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    current_detector.setup_video_writer(frame_width, frame_height, fps, output_path)
    print(f"出力動画パス: {output_path}")

    frame_count = 0
    start_time = asyncio.get_event_loop().time()
    print("フレーム処理を開始します...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("動画の終了に到達しました")
                break

            # フレームレート調整: 10FPSに調整（3フレームに1回処理）
            if frame_count % 3 != 0:
                frame_count += 1
                continue

            # 元の解像度でフレームを処理（カウント表示も元解像度で実行）
            if frame_count % 30 == 0:  # 30フレームごとにログ出力
                print(f"フレーム {frame_count} を処理中...")
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

            if frame_count % 30 == 0:  # 30フレームごとにログ出力
                print(f"WebSocket送信: フレーム {frame_count}, オブジェクト数: {object_count}")
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
        print("動画処理がキャンセルされました")
        pass
    except Exception as e:
        print(f"動画処理エラー: {e}")
        import traceback
        traceback.print_exc()
        await manager.broadcast(json.dumps({
            "type": "error",
            "message": f"動画処理中にエラーが発生しました: {str(e)}"
        }))
    finally:
        print("動画処理を終了します...")
        cap.release()

        # 動画ライターをリリース
        if current_detector and current_detector.video_writer is not None:
            print(f"動画ライターをリリース中: {current_detector.output_path}")
            current_detector.video_writer.release()
            current_detector.video_writer = None  # 明示的にNoneに設定

            # ファイルの書き込み完了を待機
            import time
            time.sleep(1.0)  # 1秒待機（少し長めに）

            # ファイルの存在を確認
            if os.path.exists(current_detector.output_path):
                file_size = os.path.getsize(current_detector.output_path)
                print(f"動画ファイルサイズ: {file_size} bytes")
                if file_size == 0:
                    print("警告: 動画ファイルのサイズが0です")
                else:
                    # ストリーミング用の動画を変換
                    streaming_path = current_detector.output_path.replace('.mp4', '_streaming.mp4')
                    print(f"ストリーミング用動画を変換中: {streaming_path}")
                    if convert_video_for_streaming(current_detector.output_path, streaming_path):
                        print(f"ストリーミング用動画変換完了: {streaming_path}")
                    else:
                        print("ストリーミング用動画変換に失敗しました")
            else:
                print("警告: 動画ファイルが存在しません")
                # ディレクトリの内容を確認
                outputs_dir = "outputs"
                if os.path.exists(outputs_dir):
                    files = os.listdir(outputs_dir)
                    print(f"outputsディレクトリの内容: {files}")

        # 検知結果を保存
        if current_detector:
            current_detector.save_detection_results()

        # 完了メッセージを送信
        if current_detector:
            print(f"最終カウント: {current_detector.object_count}")
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

def convert_video_for_streaming(input_path: str, output_path: str) -> bool:
    """動画をブラウザ互換性の高い形式に変換"""
    try:
        # FFmpegコマンドで動画を変換
        # -c:v libx264: H.264コーデック
        # -profile:v baseline: ブラウザ互換性の高いベースラインプロファイル
        # -pix_fmt yuv420p: ブラウザ互換性の高いピクセル形式
        # -movflags +faststart: moov atomを先頭に配置（Fast Start）
        # -crf 23: 品質設定（18-28の範囲、23が標準）
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # 上書き許可
            output_path
        ]

        print(f"動画変換開始: {input_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"動画変換完了: {output_path}")
            return True
        else:
            print(f"動画変換エラー: {result.stderr}")
            return False

    except Exception as e:
        print(f"動画変換例外: {e}")
        return False

@app.get("/stream-video/{file_id}")
async def stream_video(file_id: str):
    """検知済み動画をストリーミング"""
    try:
        # outputsディレクトリ内のファイルを検索
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            raise HTTPException(status_code=404, detail="出力ディレクトリが見つかりません")

        # ストリーミング用の動画ファイルを検索
        expected_streaming_filename = f"{file_id}_output_streaming.mp4"
        streaming_path = os.path.join(outputs_dir, expected_streaming_filename)

        print(f"検索対象ファイルID: {file_id}")
        print(f"期待されるストリーミングファイル名: {expected_streaming_filename}")

        # まずストリーミング用ファイルを確認
        if os.path.exists(streaming_path):
            output_filename = expected_streaming_filename
            output_path = streaming_path
            print(f"ストリーミング用ファイルが見つかりました: {output_filename}")
        else:
            # ストリーミング用ファイルが見つからない場合、元のファイルを検索
            expected_filename = f"{file_id}_output.mp4"
            output_path = os.path.join(outputs_dir, expected_filename)

            if os.path.exists(output_path):
                output_filename = expected_filename
                print(f"元のファイルが見つかりました: {output_filename}")
            else:
                # ファイルIDを含むファイルを検索
                matching_files = []
                for filename in os.listdir(outputs_dir):
                    if filename.endswith("_output.mp4") and file_id in filename:
                        matching_files.append(filename)

                if matching_files:
                    # ファイルIDを含むファイルが見つかった場合、最新のものを選択
                    matching_files.sort(key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)), reverse=True)
                    output_filename = matching_files[0]
                    output_path = os.path.join(outputs_dir, output_filename)
                    print(f"ファイルIDを含むファイルを選択: {output_filename}")
                else:
                    # それでも見つからない場合、最新のファイルを使用
                    mp4_files = [f for f in os.listdir(outputs_dir) if f.endswith("_output.mp4")]
                    print(f"利用可能な動画ファイル: {mp4_files}")
                    if not mp4_files:
                        raise HTTPException(status_code=404, detail="動画ファイルが見つかりません")
                    # 最新のファイルを選択（作成日時順）
                    mp4_files.sort(key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)), reverse=True)
                    output_filename = mp4_files[0]
                    output_path = os.path.join(outputs_dir, output_filename)
                    print(f"最新ファイルを選択: {output_filename}")

        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="動画ファイルが見つかりません")

        print(f"ストリーミングファイル: {output_path}")

        # ファイルを読み込んで返す
        def iterfile():
            with open(output_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Type": "video/mp4"
            }
        )
    except Exception as e:
        print(f"ストリーミングエラー: {e}")
        raise HTTPException(status_code=500, detail=f"動画のストリーミングに失敗しました: {str(e)}")

@app.get("/download-csv/{file_id}")
async def download_csv(file_id: str):
    """検知結果をCSV形式でダウンロード"""
    try:
        global current_detector

        if current_detector is None:
            raise HTTPException(status_code=400, detail="検知が開始されていません")

        # 検知結果を取得
        detection_results = current_detector.get_detection_results()

        # CSVデータを生成
        csv_content = generate_csv_content(detection_results)

        # CSVファイルとして返す
        headers = {
            "Content-Disposition": f"attachment; filename=detection_results_{file_id}.csv",
            "Content-Type": "text/csv; charset=utf-8"
        }

        return Response(content=csv_content, headers=headers)

    except Exception as e:
        print(f"CSVダウンロードエラー: {e}")
        raise HTTPException(status_code=500, detail="CSVの生成に失敗しました")

def generate_csv_content(detection_results: dict) -> str:
    """検知結果からCSVコンテンツを生成"""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # ヘッダー行
    writer.writerow(['クラス名', 'カウント数'])

    # データ行
    for class_id, class_name in detection_results.get('object_classes', {}).items():
        # カウント数は実際のカウント数
        counted_count = detection_results.get('total_counted', 0)
        writer.writerow([class_name, counted_count])

    # 合計行
    total_counted = detection_results.get('total_counted', 0)
    writer.writerow(['合計', total_counted])

    return output.getvalue()

@app.get("/download-video/{file_id}")
async def download_video(file_id: str):
    """検知済み動画をダウンロード"""
    try:
        # ファイルIDに基づいて動画ファイルを検索
        outputs_dir = "outputs"
        expected_filename = f"{file_id}_output.mp4"
        output_path = os.path.join(outputs_dir, expected_filename)

        print(f"ダウンロード検索対象ファイルID: {file_id}")
        print(f"期待されるファイル名: {expected_filename}")

        if not os.path.exists(output_path):
            # ファイルが見つからない場合、最新のファイルを使用
            mp4_files = [f for f in os.listdir(outputs_dir) if f.endswith("_output.mp4")]
            if not mp4_files:
                raise HTTPException(status_code=404, detail="動画ファイルが見つかりません")
            # 最新のファイルを選択（作成日時順）
            mp4_files.sort(key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)), reverse=True)
            output_filename = mp4_files[0]
            output_path = os.path.join(outputs_dir, output_filename)
            print(f"最新ファイルをダウンロード: {output_filename}")
        else:
            print(f"ファイルが見つかりました: {expected_filename}")

        # ファイルを読み込んで返す
        def iterfile():
            with open(output_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=detected_{file_id}.mp4"}
        )
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        raise HTTPException(status_code=500, detail=f"動画のダウンロードに失敗しました: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
