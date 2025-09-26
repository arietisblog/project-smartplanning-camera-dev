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
class DetectedObject:
    """検知されたオブジェクト情報を格納するデータクラス"""
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

class ConfigurableObjectDetector:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.model = YOLO(self.config['model']['path'])
        self.object_count = 0
        self.tracked_objects = defaultdict(list)
        self.counting_line_y = None
        self.counting_line_angle = None
        self.counting_zone = None
        self.counted_objects = set()

        # ハンガリアンアルゴリズム用の変数
        self.objects: Dict[int, DetectedObject] = {}  # 現在追跡中のオブジェクト
        self.next_object_id = 1  # 次のオブジェクトID
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
            "input": {
                "video_path": None,
                "output_directory": "outputs"
            },
            "model": {
                "path": "yolov8n.pt",
                "confidence_threshold": 0.5
            },
            "detection": {
                "object_classes": {
                    "1": "bicycle"    # 検知対象のオブジェクトクラス
                },
                "tracking_history_frames": 10 # 追跡履歴を保持するフレーム数
            },
            "tracking": {
                "max_disappeared_frames": 30,  # オブジェクトが消失してから削除するまでのフレーム数
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
                    "counted_object": [0, 255, 0],    # カウント済みオブジェクトのバウンディングボックス色 (BGR)
                    "uncounted_object": [0, 0, 255],  # 未カウントオブジェクトのバウンディングボックス色 (BGR)
                    "counting_line": [255, 0, 0],      # カウントラインの色 (BGR)
                    "counting_zone": [255, 255, 0],    # カウントゾーンの色 (BGR)
                    "object_count_text": [0, 255, 255] # オブジェクト数テキストの色 (BGR)
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

    def is_target_object(self, class_id):
        object_classes = self.config['detection']['object_classes']
        return str(class_id) in object_classes

    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """2つの中心点間のユークリッド距離を計算"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def update_object_tracking(self, detections: List[Tuple[List[float], int, float]]) -> None:
        """
        ハンガリアンアルゴリズムを使用してオブジェクト追跡を更新

        Args:
            detections: [(bbox, class_id, confidence), ...] のリスト
        """
        # 現在のフレームで検出されたオブジェクトの中心点を計算
        current_centers = []
        current_detections = []

        for bbox, class_id, confidence in detections:
            if not self.is_target_object(class_id):
                continue

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            center = (center_x, center_y)

            current_centers.append(center)
            current_detections.append((bbox, class_id, confidence, center))

        # 既存のオブジェクトがない場合、すべて新しいオブジェクトとして追加
        if not self.objects:
            for bbox, class_id, confidence, center in current_detections:
                obj = DetectedObject(
                    id=self.next_object_id,
                    bbox=bbox,
                    class_id=class_id,
                    confidence=confidence,
                    center=center,
                    last_seen=self.frame_count
                )
                self.objects[self.next_object_id] = obj
                self.next_object_id += 1
            return

        # 既存のオブジェクトの中心点を取得
        existing_centers = []
        existing_ids = []
        for object_id, obj in self.objects.items():
            existing_centers.append(obj.center)
            existing_ids.append(object_id)

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

            # マッチングされたオブジェクトを更新
            matched_current = set()
            matched_existing = set()

            for i, j in zip(row_indices, col_indices):
                if distance_matrix[i, j] <= self.max_distance:
                    # 既存のオブジェクトを更新
                    object_id = existing_ids[j]
                    bbox, class_id, confidence, center = current_detections[i]

                    obj = self.objects[object_id]
                    obj.bbox = bbox
                    obj.class_id = class_id
                    obj.confidence = confidence
                    obj.center = center
                    obj.last_seen = self.frame_count

                    # 追跡履歴を更新
                    obj.track_history.append(center)
                    max_history = self.config['detection']['tracking_history_frames']
                    if len(obj.track_history) > max_history:
                        obj.track_history = obj.track_history[-max_history:]

                    matched_current.add(i)
                    matched_existing.add(j)

            # マッチングされなかった既存のオブジェクトを削除（消失フレーム数が上限を超えた場合）
            objects_to_remove = []
            for j, object_id in enumerate(existing_ids):
                if j not in matched_existing:
                    obj = self.objects[object_id]
                    if self.frame_count - obj.last_seen > self.max_disappeared:
                        objects_to_remove.append(object_id)

            for object_id in objects_to_remove:
                del self.objects[object_id]

            # マッチングされなかった新しい検出を新しいオブジェクトとして追加
            for i, (bbox, class_id, confidence, center) in enumerate(current_detections):
                if i not in matched_current:
                    obj = DetectedObject(
                        id=self.next_object_id,
                        bbox=bbox,
                        class_id=class_id,
                        confidence=confidence,
                        center=center,
                        last_seen=self.frame_count
                    )
                    self.objects[self.next_object_id] = obj
                    self.next_object_id += 1
        else:
            # 検出がない場合、既存のオブジェクトの消失フレーム数を増やす
            objects_to_remove = []
            for object_id, obj in self.objects.items():
                if self.frame_count - obj.last_seen > self.max_disappeared:
                    objects_to_remove.append(object_id)

            for object_id in objects_to_remove:
                del self.objects[object_id]

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

    def has_crossed_line(self, obj: DetectedObject, frame_width: int, frame_height: int) -> bool:
        """
        オブジェクトがカウントラインを横断したかチェック（角度対応）

        Args:
            obj: 検知されたオブジェクト
            frame_width: フレームの幅
            frame_height: フレームの高さ

        Returns:
            bool: ラインを横断した場合True
        """
        if self.counting_line_y is None:
            return False

        # オブジェクトの追跡履歴を取得
        track_history = obj.track_history
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
                    if not self.is_target_object(class_id) or confidence < confidence_threshold:
                        continue

                    if not self.is_in_counting_zone(bbox):
                        continue

                    detections.append((bbox, class_id, confidence))

        # ハンガリアンアルゴリズムでオブジェクト追跡を更新
        self.update_object_tracking(detections)

        # オブジェクトのカウントと描画
        for object_id, obj in self.objects.items():
            # ライン横断チェック
            if (not obj.is_counted and
                self.has_crossed_line(obj, frame_width, frame_height)):
                self.object_count += 1
                obj.is_counted = True
                self.counted_objects.add(object_id)

            # バウンディングボックスを描画
            colors = self.config['display']['colors']
            color = colors['counted_object'] if obj.is_counted else colors['uncounted_object']
            cv2.rectangle(frame,
                        (int(obj.bbox[0]), int(obj.bbox[1])),
                        (int(obj.bbox[2]), int(obj.bbox[3])),
                        color, 2)

            # ラベルを描画（クラス名と追跡ID、信頼度）
            object_classes = self.config['detection']['object_classes']
            object_name = object_classes.get(str(obj.class_id), f"class_{obj.class_id}")
            label = f"{object_name} ID:{obj.id} Conf:{obj.confidence:.2f}"
            cv2.putText(frame, label,
                      (int(obj.bbox[0]), int(obj.bbox[1] - 10)),
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

        # オブジェクト数を表示
        text_color = tuple(self.config['display']['colors']['object_count_text'])
        # 検知対象クラス名を取得して表示テキストを生成
        object_classes = self.config['detection']['object_classes']
        class_names = list(object_classes.values())
        if len(class_names) == 1:
            display_text = f"{class_names[0].title()}s: {self.object_count}"
        else:
            display_text = f"Objects: {self.object_count}"
        cv2.putText(frame, display_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        return frame, self.object_count

    def process_video(self, video_path=None, output_path=None):
        # 設定ファイルから動画パスを取得（コマンドライン引数が優先）
        if video_path is None:
            video_path = self.config['input']['video_path']
            if video_path is None:
                raise ValueError("動画パスが指定されていません。コマンドライン引数または設定ファイルで指定してください。")

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
                # 出力パスが指定されていない場合、設定ファイルの出力ディレクトリを使用
                output_dir = self.config['input']['output_directory']
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 入力動画のファイル名から自動生成
                base_name = os.path.basename(video_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}_output.mp4")

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
            processed_frame, object_count = self.process_frame(frame)

            # 出力動画に書き込み
            if out:
                out.write(processed_frame)

            # 動画を表示
            if show_video:
                # 検知対象クラス名を取得してウィンドウタイトルを生成
                object_classes = self.config['detection']['object_classes']
                class_names = list(object_classes.values())
                if len(class_names) == 1:
                    window_title = f'{class_names[0].title()} Detection'
                else:
                    window_title = 'Object Detection'
                cv2.imshow(window_title, processed_frame)

                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_screenshots:
                    screenshot_format = self.config['output']['screenshot_format']
                    output_dir = self.config['input']['output_directory']
                    screenshot_path = os.path.join(output_dir, f"screenshot_{frame_count}.{screenshot_format}")
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"スクリーンショットを保存: {screenshot_path}")

            frame_count += 1

            # 進捗表示
            if frame_count % progress_interval == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time
                print(f"処理済みフレーム: {frame_count}, オブジェクト数: {object_count}, FPS: {fps_processed:.1f}")

        # リソースを解放
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n処理完了!")
        print(f"総フレーム数: {frame_count}")
        print(f"検知されたオブジェクト数: {self.object_count}")
        print(f"処理時間: {time.time() - start_time:.2f}秒")

def main():
    parser = argparse.ArgumentParser(description='設定ファイルを使用したオブジェクト検知システム')
    parser.add_argument('video_path', nargs='?', help='入力動画ファイルのパス（設定ファイルでも指定可能）')
    parser.add_argument('--config', '-c', default='config.json', help='設定ファイルのパス')
    parser.add_argument('--output', '-o', help='出力動画ファイルのパス')

    args = parser.parse_args()

    # オブジェクト検知器を作成
    detector = ConfigurableObjectDetector(args.config)

    # 動画を処理（コマンドライン引数が優先、なければ設定ファイルから取得）
    detector.process_video(args.video_path, args.output)

if __name__ == "__main__":
    main()