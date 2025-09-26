import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import argparse
import json
import os
import math
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Vehicle:
    """自転車情報を格納するデータクラス"""
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    center: Tuple[float, float]  # (center_x, center_y)
    last_seen: int  # 最後に見られたフレーム番号
    is_counted: bool = False
    track_history: List[Tuple[float, float]] = None  # 中心点の履歴

    def __post_init__(self):
        if self.track_history is None:
            self.track_history = [self.center]

class ConfigurableBicycleDetector:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.model = YOLO(self.config['model']['path'])
        self.bicycle_count = 0
        self.tracked_vehicles = defaultdict(list)
        self.counting_line_y = None
        self.counting_line_angle = None
        self.counting_zone = None
        self.counted_vehicles = set()

        # ハンガリアンアルゴリズム用の変数
        self.vehicles: Dict[int, Vehicle] = {}  # 現在追跡中の自転車
        self.next_vehicle_id = 1  # 次の自転車ID
        self.frame_count = 0  # フレームカウンター
        self.max_disappeared = self.config['tracking'].get('max_disappeared_frames', 30)  # 最大消失フレーム数
        self.max_distance = self.config['tracking'].get('max_distance', 100)  # 最大マッチング距離

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            print(f"警告: 設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
            return self.get_default_config()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"設定ファイル '{config_path}' を読み込みました。")
            return config
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            print("デフォルト設定を使用します。")
            return self.get_default_config()

    def get_default_config(self):
        return {
            "model": {
                "path": "yolov8n.pt",
                "confidence_threshold": 0.5
            },
            "detection": {
                "vehicle_classes": {
                    "1": "bicycle"    # 自転車
                },
                "tracking_history_frames": 10 # 追跡履歴を保持するフレーム数
            },
            "tracking": {
                "max_disappeared_frames": 30,  # 自転車が消失してから削除するまでのフレーム数
                "max_distance": 100,  # ハンガリアンアルゴリズムでの最大マッチング距離
                "min_confidence": 0.5  # 追跡に使用する最小信頼度
            },
            "counting": {
                "line_ratio": 0.6,   # カウントラインの画面高さに対する割合
                "line_angle": 0.0,   # カウントラインの角度（度数、0は水平）
                "zone_ratio": 0.3,   # カウントゾーンの画面幅に対する割合
                "direction": "both" # カウント方向 ("upward", "downward", "both")
            },
            "display": {
                "show_video": True,
                "save_screenshots": True,
                "progress_interval": 30, # 進捗表示のフレーム間隔
                "colors": {
                    "counted_vehicle": [0, 255, 0],    # カウント済み自転車のバウンディングボックス色 (BGR)
                    "uncounted_vehicle": [0, 0, 255],  # 未カウント自転車のバウンディングボックス色 (BGR)
                    "counting_line": [255, 0, 0],      # カウントラインの色 (BGR)
                    "counting_zone": [255, 255, 0],    # カウントゾーンの色 (BGR)
                    "vehicle_count_text": [0, 255, 255] # 自転車数テキストの色 (BGR)
                }
            },
            "output": {
                "save_video": False,
                "video_codec": "mp4v", # 出力動画のコーデック
                "screenshot_format": "jpg" # スクリーンショットのフォーマット
            }
        }

    def set_counting_line(self, frame_height, frame_width):
        line_ratio = self.config['counting']['line_ratio']
        line_angle = self.config['counting'].get('line_angle', 0.0)

        base_y = int(frame_height * line_ratio)
        self.counting_line_y = base_y
        self.counting_line_angle = line_angle

    def set_counting_zone(self, frame_width, frame_height):
        zone_ratio = self.config['counting']['zone_ratio']
        zone_width = int(frame_width * zone_ratio)
        zone_x = (frame_width - zone_width) // 2
        self.counting_zone = {
            'x1': zone_x,
            'x2': zone_x + zone_width,
            'y1': 0,
            'y2': frame_height
        }

    def is_vehicle(self, class_id):
        vehicle_classes = self.config['detection']['vehicle_classes']
        return str(class_id) in vehicle_classes

    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """2つの中心点間のユークリッド距離を計算"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def update_vehicle_tracking(self, detections: List[Tuple[List[float], int, float]]) -> None:
        """
        ハンガリアンアルゴリズムを使用して自転車追跡を更新

        Args:
            detections: [(bbox, class_id, confidence), ...] のリスト
        """
        # 現在のフレームで検出された自転車の中心点を計算
        current_centers = []
        current_detections = []

        for bbox, class_id, confidence in detections:
            if not self.is_vehicle(class_id):
                continue

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            center = (center_x, center_y)

            current_centers.append(center)
            current_detections.append((bbox, class_id, confidence, center))

        # 既存の自転車がない場合、すべて新しい自転車として追加
        if not self.vehicles:
            for bbox, class_id, confidence, center in current_detections:
                vehicle = Vehicle(
                    id=self.next_vehicle_id,
                    bbox=bbox,
                    class_id=class_id,
                    confidence=confidence,
                    center=center,
                    last_seen=self.frame_count
                )
                self.vehicles[self.next_vehicle_id] = vehicle
                self.next_vehicle_id += 1
            return

        # 既存の自転車の中心点を取得
        existing_centers = []
        existing_ids = []
        for vehicle_id, vehicle in self.vehicles.items():
            existing_centers.append(vehicle.center)
            existing_ids.append(vehicle_id)

        # 距離行列を作成
        if current_centers and existing_centers:
            distance_matrix = np.zeros((len(current_centers), len(existing_centers)))

            for i, current_center in enumerate(current_centers):
                for j, existing_center in enumerate(existing_centers):
                    distance = self.calculate_distance(current_center, existing_center)
                    # 最大距離を超える場合は大きな値に設定
                    if distance > self.max_distance:
                        distance = 999999
                    distance_matrix[i, j] = distance

            # ハンガリアンアルゴリズムで最適なマッチングを見つける
            row_indices, col_indices = linear_sum_assignment(distance_matrix)

            # マッチングされた自転車を更新
            matched_current = set()
            matched_existing = set()

            for i, j in zip(row_indices, col_indices):
                if distance_matrix[i, j] <= self.max_distance:
                    # 既存の自転車を更新
                    vehicle_id = existing_ids[j]
                    bbox, class_id, confidence, center = current_detections[i]

                    vehicle = self.vehicles[vehicle_id]
                    vehicle.bbox = bbox
                    vehicle.class_id = class_id
                    vehicle.confidence = confidence
                    vehicle.center = center
                    vehicle.last_seen = self.frame_count

                    # 追跡履歴を更新
                    vehicle.track_history.append(center)
                    max_history = self.config['detection']['tracking_history_frames']
                    if len(vehicle.track_history) > max_history:
                        vehicle.track_history = vehicle.track_history[-max_history:]

                    matched_current.add(i)
                    matched_existing.add(j)

            # マッチングされなかった既存の自転車を削除（消失フレーム数が上限を超えた場合）
            vehicles_to_remove = []
            for j, vehicle_id in enumerate(existing_ids):
                if j not in matched_existing:
                    vehicle = self.vehicles[vehicle_id]
                    if self.frame_count - vehicle.last_seen > self.max_disappeared:
                        vehicles_to_remove.append(vehicle_id)

            for vehicle_id in vehicles_to_remove:
                del self.vehicles[vehicle_id]

            # マッチングされなかった新しい検出を新しい自転車として追加
            for i, (bbox, class_id, confidence, center) in enumerate(current_detections):
                if i not in matched_current:
                    vehicle = Vehicle(
                        id=self.next_vehicle_id,
                        bbox=bbox,
                        class_id=class_id,
                        confidence=confidence,
                        center=center,
                        last_seen=self.frame_count
                    )
                    self.vehicles[self.next_vehicle_id] = vehicle
                    self.next_vehicle_id += 1
        else:
            # 検出がない場合、既存の自転車の消失フレーム数を増やす
            vehicles_to_remove = []
            for vehicle_id, vehicle in self.vehicles.items():
                if self.frame_count - vehicle.last_seen > self.max_disappeared:
                    vehicles_to_remove.append(vehicle_id)

            for vehicle_id in vehicles_to_remove:
                del self.vehicles[vehicle_id]

    def is_in_counting_zone(self, bbox):
        if self.counting_zone is None:
            return True

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (self.counting_zone['x1'] <= center_x <= self.counting_zone['x2'] and
                self.counting_zone['y1'] <= center_y <= self.counting_zone['y2'])

    def get_line_y_at_x(self, x, frame_width, frame_height):
        if self.counting_line_angle == 0:
            return self.counting_line_y

        angle_rad = math.radians(self.counting_line_angle)
        center_of_line_x = frame_width / 2
        relative_x = x - center_of_line_x
        y_offset = relative_x * math.tan(angle_rad)

        return self.counting_line_y + y_offset

    def has_crossed_line(self, vehicle: Vehicle, frame_width: int, frame_height: int) -> bool:
        """
        自転車がカウントラインを横断したかチェック（角度対応）

        Args:
            vehicle: 自転車オブジェクト
            frame_width: フレームの幅
            frame_height: フレームの高さ

        Returns:
            bool: ラインを横断した場合True
        """
        if self.counting_line_y is None:
            return False

        # 自転車の追跡履歴を取得
        track_history = vehicle.track_history
        if len(track_history) < 2:  # 少なくとも2つの履歴（現在と前回）が必要
            return False

        # 前回の位置と現在の位置を比較
        prev_center = track_history[-2]
        current_center = track_history[-1]

        # 現在位置と前回位置でのカウントラインのY座標を計算
        current_line_y = self.get_line_y_at_x(current_center[0], frame_width, frame_height)
        prev_line_y = self.get_line_y_at_x(prev_center[0], frame_width, frame_height)

        direction = self.config['counting']['direction']

        if direction == "upward":
            return (prev_center[1] > prev_line_y and
                    current_center[1] <= current_line_y)
        elif direction == "downward":
            return (prev_center[1] < prev_line_y and
                    current_center[1] >= current_line_y)
        else:
            return ((prev_center[1] > prev_line_y and current_center[1] <= current_line_y) or
                    (prev_center[1] < prev_line_y and current_center[1] >= current_line_y))


    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        self.frame_count += 1

        # YOLOで検出
        results = self.model(frame, verbose=False)

        # 検出結果を収集
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    # 信頼度閾値チェック
                    confidence_threshold = self.config['model']['confidence_threshold']
                    if not self.is_vehicle(class_id) or confidence < confidence_threshold:
                        continue

                    if not self.is_in_counting_zone(bbox):
                        continue

                    detections.append((bbox, class_id, confidence))

        # ハンガリアンアルゴリズムで自転車追跡を更新
        self.update_vehicle_tracking(detections)

        # 自転車のカウントと描画
        for vehicle_id, vehicle in self.vehicles.items():
            # ライン横断チェック
            if (not vehicle.is_counted and
                self.has_crossed_line(vehicle, frame_width, frame_height)):
                self.bicycle_count += 1
                vehicle.is_counted = True
                self.counted_vehicles.add(vehicle_id)

            # バウンディングボックスを描画
            colors = self.config['display']['colors']
            color = colors['counted_vehicle'] if vehicle.is_counted else colors['uncounted_vehicle']
            cv2.rectangle(frame,
                        (int(vehicle.bbox[0]), int(vehicle.bbox[1])),
                        (int(vehicle.bbox[2]), int(vehicle.bbox[3])),
                        color, 2)

            # ラベルを描画（クラス名と追跡ID、信頼度）
            vehicle_classes = self.config['detection']['vehicle_classes']
            vehicle_name = vehicle_classes.get(str(vehicle.class_id), f"class_{vehicle.class_id}")
            label = f"{vehicle_name} ID:{vehicle.id} Conf:{vehicle.confidence:.2f}"
            cv2.putText(frame, label,
                      (int(vehicle.bbox[0]), int(vehicle.bbox[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # カウントラインを描画（角度対応）
        if self.counting_line_y is not None:
            line_color = tuple(self.config['display']['colors']['counting_line'])

            if self.counting_line_angle == 0:
                # 水平線の場合
                cv2.line(frame, (0, self.counting_line_y),
                        (frame_width, self.counting_line_y), line_color, 2)
            else:
                # 角度付き線の場合
                # 画面の左右端でのY座標を計算
                left_y = int(self.get_line_y_at_x(0, frame_width, frame_height))
                right_y = int(self.get_line_y_at_x(frame_width, frame_width, frame_height))

                cv2.line(frame, (0, left_y), (frame_width, right_y), line_color, 2)

        # カウントゾーンを描画
        if self.counting_zone is not None:
            zone_color = tuple(self.config['display']['colors']['counting_zone'])
            cv2.rectangle(frame,
                        (self.counting_zone['x1'], self.counting_zone['y1']),
                        (self.counting_zone['x2'], self.counting_zone['y2']),
                        zone_color, 2)

        # 自転車数を表示
        text_color = tuple(self.config['display']['colors']['vehicle_count_text'])
        cv2.putText(frame, f"Bicycles: {self.bicycle_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        return frame, self.bicycle_count

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
            return

        # 動画の情報を取得
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # カウントラインとゾーンを設定
        self.set_counting_line(frame_height, frame_width)
        self.set_counting_zone(frame_width, frame_height)

        # 出力動画の設定
        out = None
        if output_path or self.config['output']['save_video']:
            if not output_path:
                # 出力パスが指定されていない場合、入力動画のファイル名から自動生成
                base_name = os.path.basename(video_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_path = f"{name_without_ext}_output.mp4" # .mp4は固定

            codec = self.config['output']['video_codec']
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"出力動画を保存します: {output_path}")


        show_video = self.config['display']['show_video']
        save_screenshots = self.config['display']['save_screenshots']
        progress_interval = self.config['display']['progress_interval']

        print("動画処理を開始します...")
        if show_video:
            print("'q'キーで終了、's'キーでスクリーンショット保存")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを処理
            processed_frame, bicycle_count = self.process_frame(frame)

            # 出力動画に書き込み
            if out:
                out.write(processed_frame)

            # 動画を表示
            if show_video:
                cv2.imshow('Bicycle Detection', processed_frame)

                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_screenshots:
                    screenshot_format = self.config['output']['screenshot_format']
                    screenshot_path = f"screenshot_{frame_count}.{screenshot_format}"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"スクリーンショットを保存: {screenshot_path}")

            frame_count += 1

            # 進捗表示
            if frame_count % progress_interval == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time
                print(f"処理済みフレーム: {frame_count}, 自転車数: {bicycle_count}, FPS: {fps_processed:.1f}")

        # リソースを解放
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n処理完了!")
        print(f"総フレーム数: {frame_count}")
        print(f"検知された自転車数: {self.bicycle_count}")
        print(f"処理時間: {time.time() - start_time:.2f}秒")

def main():
    parser = argparse.ArgumentParser(description='設定ファイルを使用した自転車検知システム')
    parser.add_argument('video_path', help='入力動画ファイルのパス')
    parser.add_argument('--config', '-c', default='config.json', help='設定ファイルのパス')
    parser.add_argument('--output', '-o', help='出力動画ファイルのパス')

    args = parser.parse_args()

    # 自転車検知器を作成
    detector = ConfigurableBicycleDetector(args.config)

    # 動画を処理
    detector.process_video(args.video_path, args.output)

if __name__ == "__main__":
    main()