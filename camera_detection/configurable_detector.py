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
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import shutil

@dataclass
class RepresentativeImage:
    """代表画像の情報を格納するデータクラス"""
    object_id: int
    image_path: str
    timestamp: float
    frame_number: int
    bbox: List[float]
    confidence: float
    last_updated: float

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
    representative_images: List[RepresentativeImage] = field(default_factory=list)  # 代表画像リスト
    candidate_images: List[RepresentativeImage] = field(default_factory=list)  # 候補画像リスト（大量収集用）
    is_finalized: bool = False  # 最終選択済みかどうか

    def __post_init__(self):
        if self.track_history is None:
            self.track_history = [self.center]

class RepresentativeImageManager:
    """代表画像の管理を行うクラス"""

    def __init__(self, config: dict):
        self.config = config
        base_directory = config.get('representative_images', {}).get('save_directory', 'representative_images')

        # 日時分秒を追加したディレクトリ名を生成
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_directory = f"{base_directory}_{timestamp}"

        self.max_images_per_object = config.get('representative_images', {}).get('max_images_per_object', 5)
        self.image_quality_threshold = config.get('representative_images', {}).get('image_quality_threshold', 0.7)
        self.image_format = config.get('representative_images', {}).get('image_format', 'jpg')

        # 保存ディレクトリを作成
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"代表画像保存ディレクトリを作成しました: {self.save_directory}")

    def save_candidate_image(self, obj: DetectedObject, frame: np.ndarray, frame_number: int) -> bool:
        """
        候補画像を大量収集（最終選択前の収集フェーズ）

        Args:
            obj: 検知されたオブジェクト
            frame: 現在のフレーム
            frame_number: フレーム番号

        Returns:
            bool: 保存に成功した場合True
        """
        if not obj.is_counted or obj.is_finalized:
            return False

        # 基本的な品質チェック（極端に低いもののみ除外）
        if obj.confidence < 0.2:  # より緩い条件で大量収集
            return False

        # バウンディングボックスの基本的なサイズチェック
        x1, y1, x2, y2 = obj.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        # 極端に小さいもののみ除外
        if bbox_area < 50:  # より緩い条件
            return False

        # バウンディングボックスから画像を切り出し
        x1, y1, x2, y2 = map(int, obj.bbox)

        # 境界チェック
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        # 画像を切り出し
        cropped_image = frame[y1:y2, x1:x2]

        if cropped_image.size == 0:
            return False

        # ファイル名を生成（候補用）
        timestamp = time.time()
        filename = f"obj_{obj.id}_candidate_{len(obj.candidate_images)}_{timestamp:.0f}.{self.image_format}"
        image_path = os.path.join(self.save_directory, filename)

        # 画像を保存
        try:
            cv2.imwrite(image_path, cropped_image)

            # RepresentativeImageオブジェクトを作成
            rep_image = RepresentativeImage(
                object_id=obj.id,
                image_path=image_path,
                timestamp=timestamp,
                frame_number=frame_number,
                bbox=obj.bbox.copy(),
                confidence=obj.confidence,
                last_updated=timestamp
            )

            # 候補画像リストに追加（制限なしで大量収集）
            obj.candidate_images.append(rep_image)

            return True

        except Exception as e:
            print(f"候補画像保存エラー: {e}")
            return False

    def _optimize_representative_images(self, obj: DetectedObject):
        """代表画像を最適化（オブジェクトごとの相対評価）"""
        if len(obj.representative_images) <= self.max_images_per_object:
            return

        # このオブジェクトの信頼度とサイズの範囲を取得
        confidences = [img.confidence for img in obj.representative_images]
        bbox_areas = []
        for img in obj.representative_images:
            x1, y1, x2, y2 = img.bbox
            bbox_areas.append((x2 - x1) * (y2 - y1))

        # 正規化用の範囲を計算
        min_conf = min(confidences)
        max_conf = max(confidences)
        min_area = min(bbox_areas)
        max_area = max(bbox_areas)

        # 各画像の相対スコアを計算
        def calculate_relative_score(img):
            # バウンディングボックスサイズ
            x1, y1, x2, y2 = img.bbox
            bbox_area = (x2 - x1) * (y2 - y1)

            # 相対信頼度スコア（このオブジェクト内での相対位置）
            if max_conf > min_conf:
                relative_confidence = (img.confidence - min_conf) / (max_conf - min_conf)
            else:
                relative_confidence = 1.0  # すべて同じ信頼度の場合

            # 相対サイズスコア（このオブジェクト内での相対位置）
            if max_area > min_area:
                relative_size = (bbox_area - min_area) / (max_area - min_area)
            else:
                relative_size = 1.0  # すべて同じサイズの場合

            # 時間分散スコア（他の画像との時間差の平均）
            time_diffs = []
            for other_img in obj.representative_images:
                if other_img != img:
                    time_diff = abs(img.timestamp - other_img.timestamp)
                    time_diffs.append(time_diff)

            # 時間分散スコア（時間差の平均を正規化）
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            time_diversity_score = min(avg_time_diff / 10.0, 1.0)  # 10秒で正規化

            # 総合スコア（重み付き）
            total_score = (
                relative_confidence * 0.4 +      # 相対信頼度 40%
                relative_size * 0.3 +             # 相対サイズ 30%
                time_diversity_score * 0.3        # 時間分散 30%
            )

            return total_score

        # スコアでソート
        scored_images = [(img, calculate_relative_score(img)) for img in obj.representative_images]
        scored_images.sort(key=lambda x: x[1], reverse=True)

        # 上位の画像を保持
        keep_images = [img for img, score in scored_images[:self.max_images_per_object]]

        # 削除する画像を特定
        images_to_remove = [img for img in obj.representative_images if img not in keep_images]

        # 不要な画像を削除
        for img in images_to_remove:
            try:
                if os.path.exists(img.image_path):
                    os.remove(img.image_path)
                obj.representative_images.remove(img)
            except Exception as e:
                print(f"画像削除エラー: {e}")

    def _remove_oldest_image(self, obj: DetectedObject):
        """最も古い代表画像を削除（後方互換性のため残す）"""
        if not obj.representative_images:
            return

        # 最も古い画像を特定（信頼度が低い順）
        oldest_image = min(obj.representative_images, key=lambda x: (x.confidence, x.timestamp))

        try:
            # ファイルを削除
            if os.path.exists(oldest_image.image_path):
                os.remove(oldest_image.image_path)
            # リストから削除
            obj.representative_images.remove(oldest_image)
        except Exception as e:
            print(f"画像削除エラー: {e}")

    def cleanup_old_images(self, objects: Dict[int, DetectedObject], current_time: float, cleanup_interval: float = 30.0):
        """一定期間カウントされていないオブジェクトの画像を削除"""
        objects_to_remove = []

        for object_id, obj in objects.items():
            # 一定期間カウントされていないオブジェクトの画像を削除
            if current_time - obj.last_seen > cleanup_interval:
                self._remove_all_images(obj)
                objects_to_remove.append(object_id)

        # オブジェクトを削除
        for object_id in objects_to_remove:
            del objects[object_id]

    def _remove_all_images(self, obj: DetectedObject):
        """オブジェクトのすべての代表画像を削除"""
        for rep_image in obj.representative_images:
            try:
                if os.path.exists(rep_image.image_path):
                    os.remove(rep_image.image_path)
            except Exception as e:
                print(f"画像削除エラー: {e}")

        obj.representative_images.clear()

    def finalize_representative_images(self, obj: DetectedObject):
        """
        候補画像から最適な5枚を最終選択

        Args:
            obj: 検知されたオブジェクト
        """
        if obj.is_finalized or len(obj.candidate_images) == 0:
            return

        # 候補画像が5枚以下の場合はそのまま採用
        if len(obj.candidate_images) <= self.max_images_per_object:
            obj.representative_images = obj.candidate_images.copy()
            obj.is_finalized = True
            return

        # 候補画像のスコアを計算
        def calculate_final_score(img):
            # バウンディングボックスサイズ
            x1, y1, x2, y2 = img.bbox
            bbox_area = (x2 - x1) * (y2 - y1)

            # このオブジェクトの信頼度とサイズの範囲を取得
            confidences = [candidate.confidence for candidate in obj.candidate_images]
            bbox_areas = []
            for candidate in obj.candidate_images:
                x1, y1, x2, y2 = candidate.bbox
                bbox_areas.append((x2 - x1) * (y2 - y1))

            min_conf = min(confidences)
            max_conf = max(confidences)
            min_area = min(bbox_areas)
            max_area = max(bbox_areas)

            # 相対信頼度スコア
            if max_conf > min_conf:
                relative_confidence = (img.confidence - min_conf) / (max_conf - min_conf)
            else:
                relative_confidence = 1.0

            # 相対サイズスコア
            if max_area > min_area:
                relative_size = (bbox_area - min_area) / (max_area - min_area)
            else:
                relative_size = 1.0

            # 時間分散スコア（他の候補画像との時間差の平均）
            time_diffs = []
            for other_img in obj.candidate_images:
                if other_img != img:
                    time_diff = abs(img.timestamp - other_img.timestamp)
                    time_diffs.append(time_diff)

            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            time_diversity_score = min(avg_time_diff / 10.0, 1.0)

            # 総合スコア
            total_score = (
                relative_confidence * 0.4 +      # 相対信頼度 40%
                relative_size * 0.3 +             # 相対サイズ 30%
                time_diversity_score * 0.3        # 時間分散 30%
            )

            return total_score

        # スコアでソートして上位5枚を選択
        scored_images = [(img, calculate_final_score(img)) for img in obj.candidate_images]
        scored_images.sort(key=lambda x: x[1], reverse=True)

        # 上位5枚を代表画像として採用
        selected_images = [img for img, score in scored_images[:self.max_images_per_object]]
        obj.representative_images = selected_images

        # 不要な候補画像を削除
        for img in obj.candidate_images:
            if img not in selected_images:
                try:
                    if os.path.exists(img.image_path):
                        os.remove(img.image_path)
                except Exception as e:
                    print(f"候補画像削除エラー: {e}")

        # 候補画像リストをクリア
        obj.candidate_images.clear()
        obj.is_finalized = True

        print(f"オブジェクト {obj.id}: {len(selected_images)}枚の代表画像を最終選択しました")

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

        # 代表画像管理
        self.image_manager = RepresentativeImageManager(self.config)
        self.last_cleanup_time = time.time()

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
                    "counting_line": [0, 165, 255],      # カウントラインの色 (BGR, オレンジ)
                    "counting_zone": [60, 180, 75],      # カウントゾーンの色 (BGR, 黄緑)
                    "object_count_text": [255, 255, 255] # オブジェクト数テキストの色 (BGR, 白)
                }
            },
            "output": {
                "save_video": False,
                "video_codec": "mp4v", # 出力動画のコーデック
                "screenshot_format": "jpg" # スクリーンショットのフォーマット
            },
            "representative_images": {
                "enabled": True,
                "max_images_per_object": 5,
                "save_interval_frames": 10,
                "cleanup_interval_seconds": 30,
                "max_storage_mb": 100,
                "image_format": "jpg",
                "save_directory": "representative_images"
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

        # 元のフレームを保存（描画前の状態）
        original_frame = frame.copy()

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

        # 定期的なクリーンアップ（30秒ごと）
        current_time = time.time()
        if current_time - self.last_cleanup_time > 30:
            cleanup_interval = self.config.get('representative_images', {}).get('cleanup_interval_seconds', 30)
            self.image_manager.cleanup_old_images(self.objects, current_time, cleanup_interval)
            self.last_cleanup_time = current_time

        # 消失したオブジェクトの最終選択処理
        for object_id, obj in list(self.objects.items()):
            if (obj.is_counted and
                not obj.is_finalized and
                self.frame_count - obj.last_seen > 5):  # 5フレーム消失したら最終選択
                print(f"オブジェクト {obj.id} の最終選択を実行します（候補画像数: {len(obj.candidate_images)}）")
                self.image_manager.finalize_representative_images(obj)

        # オブジェクトのカウントと描画
        for object_id, obj in self.objects.items():
            # ライン横断チェック
            if (not obj.is_counted and
                self.has_crossed_line(obj, frame_width, frame_height)):
                self.object_count += 1
                obj.is_counted = True
                self.counted_objects.add(object_id)

                # 代表画像を保存（カウントされた瞬間）
                if self.config.get('representative_images', {}).get('enabled', True):
                    self.image_manager.save_candidate_image(obj, original_frame, self.frame_count)

            # カウント済みオブジェクトの候補画像収集（大量収集フェーズ）
            if (obj.is_counted and
                not obj.is_finalized and
                self.config.get('representative_images', {}).get('enabled', True)):
                # 候補画像を積極的に収集
                should_save = False

                # 1. 定期的な保存（フレーム間隔）
                save_interval = self.config.get('representative_images', {}).get('save_interval_frames', 5)
                if self.frame_count % save_interval == 0:
                    should_save = True

                # 2. 信頼度が高い場合の追加保存
                if obj.confidence > 0.6:  # 中程度の信頼度でも保存
                    should_save = True

                # 3. バウンディングボックスが大きい場合の追加保存
                x1, y1, x2, y2 = obj.bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area > 2000:  # 中程度のサイズでも保存
                    should_save = True

                # 4. 最後の保存から一定時間経過した場合
                if obj.candidate_images:
                    last_save_time = max(img.timestamp for img in obj.candidate_images)
                    if time.time() - last_save_time > 1.0:  # 1秒経過したら保存
                        should_save = True
                else:
                    # 初回は必ず保存
                    should_save = True

                if should_save:
                    self.image_manager.save_candidate_image(obj, original_frame, self.frame_count)

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
            display_text = f"Counted {class_names[0].title()}s: {self.object_count}"
        else:
            display_text = f"Objects: {self.object_count}"

        # 文字サイズとフォントを設定
        font_scale = 1.0  # 文字サイズを大きくする
        font_thickness = 2  # 文字の太さを増やす

        # テキストのサイズを取得
        (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # テキストの位置を計算（上端から十分な距離を確保）
        padding = 10
        text_x = 12
        text_y = text_height + padding + 4  # 上端から十分な距離を確保

        # 背景の白い四角形を描画
        cv2.rectangle(frame,
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (255, 255, 255), -1)  # 白い背景

        # テキストを描画
        cv2.putText(frame, display_text,
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

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

        # 残りのオブジェクトの最終選択処理
        for object_id, obj in self.objects.items():
            if obj.is_counted and not obj.is_finalized:
                print(f"動画終了時: オブジェクト {obj.id} の最終選択を実行します（候補画像数: {len(obj.candidate_images)}）")
                self.image_manager.finalize_representative_images(obj)

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