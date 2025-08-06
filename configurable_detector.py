import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import argparse
import json
import os
import math

class ConfigurableVehicleDetector:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.model = YOLO(self.config['model']['path'])
        self.vehicle_count = 0
        self.tracked_vehicles = defaultdict(list)
        self.counting_line_y = None
        self.counting_line_angle = None
        self.counting_zone = None
        self.counted_vehicles = set()

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
                    "2": "car",      # 車
                    "3": "motorcycle", # バイク
                    "5": "bus",      # バス
                    "7": "truck"     # トラック
                },
                "tracking_history_frames": 10 # 追跡履歴を保持するフレーム数
            },
            "counting": {
                "line_ratio": 0.6,   # カウントラインの画面高さに対する割合
                "line_angle": 0.0,   # カウントラインの角度（度数、0は水平）
                "zone_ratio": 0.3,   # カウントゾーンの画面幅に対する割合
                "direction": "upward" # カウント方向 ("upward", "downward", "both")
            },
            "display": {
                "show_video": True,
                "save_screenshots": True,
                "progress_interval": 30, # 進捗表示のフレーム間隔
                "colors": {
                    "counted_vehicle": [0, 255, 0],    # カウント済み車両のバウンディングボックス色 (BGR)
                    "uncounted_vehicle": [0, 0, 255],  # 未カウント車両のバウンディングボックス色 (BGR)
                    "counting_line": [255, 0, 0],      # カウントラインの色 (BGR)
                    "counting_zone": [255, 255, 0],    # カウントゾーンの色 (BGR)
                    "vehicle_count_text": [0, 255, 255] # 車両数テキストの色 (BGR)
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

    def has_crossed_line(self, vehicle_id, current_bbox, frame_width, frame_height):
        """
        車両がカウントラインを横断したかチェック（角度対応）

        Args:
            vehicle_id (str): 車両のID (track_idを含む)
            current_bbox (list): 現在のバウンディングボックス
            frame_width (int): フレームの幅
            frame_height (int): フレームの高さ

        Returns:
            bool: ラインを横断した場合True
        """
        if self.counting_line_y is None:
            return False

        # 車両の履歴を取得
        history = self.tracked_vehicles[vehicle_id]
        if len(history) < 2: # 少なくとも2つの履歴（現在と前回）が必要
            return False

        # 前回の位置と現在の位置を比較
        prev_bbox = history[-2]
        current_center_x = (current_bbox[0] + current_bbox[2]) / 2
        current_center_y = (current_bbox[1] + current_bbox[3]) / 2
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2

        # 現在位置と前回位置でのカウントラインのY座標を計算
        current_line_y = self.get_line_y_at_x(current_center_x, frame_width, frame_height)
        prev_line_y = self.get_line_y_at_x(prev_center_x, frame_width, frame_height)

        direction = self.config['counting']['direction']

        if direction == "upward":
            return (prev_center_y > prev_line_y and
                    current_center_y <= current_line_y)
        elif direction == "downward":
            return (prev_center_y < prev_line_y and
                    current_center_y >= current_line_y)
        else:
            return ((prev_center_y > prev_line_y and current_center_y <= current_line_y) or
                    (prev_center_y < prev_line_y and current_center_y >= current_line_y))


    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]

        results = self.model.track(frame, persist=True, verbose=False)

        for result in results:
            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

            if boxes is not None:
                for i, box in enumerate(boxes):
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    # 追跡IDを取得。存在しない場合はNone（通常は発生しないはずだが念のため）
                    track_id = track_ids[i] if track_ids else None

                    # 信頼度閾値チェック
                    confidence_threshold = self.config['model']['confidence_threshold']
                    if not self.is_vehicle(class_id) or confidence < confidence_threshold:
                        continue

                    if track_id is None:
                        continue

                    if not self.is_in_counting_zone(bbox):
                        continue

                    # 車両IDとしてtrack_idとclass_idを組み合わせる
                    # これにより、同じ車両は一貫したIDを持ち続ける
                    vehicle_id = f"{class_id}_{track_id}"

                    # 車両の履歴を更新
                    self.tracked_vehicles[vehicle_id].append(bbox)

                    # 履歴が長すぎる場合は削除
                    max_history = self.config['detection']['tracking_history_frames']
                    if len(self.tracked_vehicles[vehicle_id]) > max_history:
                        self.tracked_vehicles[vehicle_id] = self.tracked_vehicles[vehicle_id][-max_history:]

                    # ライン横断チェック
                    if (vehicle_id not in self.counted_vehicles and
                        self.has_crossed_line(vehicle_id, bbox, frame_width, frame_height)):
                        self.vehicle_count += 1
                        self.counted_vehicles.add(vehicle_id)

                    # バウンディングボックスを描画
                    colors = self.config['display']['colors']
                    color = colors['counted_vehicle'] if vehicle_id in self.counted_vehicles else colors['uncounted_vehicle']
                    cv2.rectangle(frame,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                color, 2)

                    # ラベルを描画（クラス名と追跡ID、信頼度）
                    vehicle_classes = self.config['detection']['vehicle_classes']
                    vehicle_name = vehicle_classes.get(str(class_id), f"class_{class_id}")
                    label = f"{vehicle_name} ID:{track_id} Conf:{confidence:.2f}" # IDを表示に追加
                    cv2.putText(frame, label,
                              (int(bbox[0]), int(bbox[1] - 10)),
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

        # 車両数を表示
        text_color = tuple(self.config['display']['colors']['vehicle_count_text'])
        cv2.putText(frame, f"Vehicles: {self.vehicle_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        return frame, self.vehicle_count

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
            processed_frame, vehicle_count = self.process_frame(frame)

            # 出力動画に書き込み
            if out:
                out.write(processed_frame)

            # 動画を表示
            if show_video:
                cv2.imshow('Vehicle Detection', processed_frame)

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
                print(f"処理済みフレーム: {frame_count}, 車両数: {vehicle_count}, FPS: {fps_processed:.1f}")

        # リソースを解放
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n処理完了!")
        print(f"総フレーム数: {frame_count}")
        print(f"検知された車両数: {self.vehicle_count}")
        print(f"処理時間: {time.time() - start_time:.2f}秒")

def main():
    parser = argparse.ArgumentParser(description='設定ファイルを使用した車両検知システム')
    parser.add_argument('video_path', help='入力動画ファイルのパス')
    parser.add_argument('--config', '-c', default='config.json', help='設定ファイルのパス')
    parser.add_argument('--output', '-o', help='出力動画ファイルのパス')

    args = parser.parse_args()

    # 車両検知器を作成
    detector = ConfigurableVehicleDetector(args.config)

    # 動画を処理
    detector.process_video(args.video_path, args.output)

if __name__ == "__main__":
    main()
