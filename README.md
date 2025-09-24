# カメラ検知システム

YOLOv8を使用した人物検知・カウントシステムです。設定ファイルを通じて柔軟にカスタマイズ可能で、人物の検知、追跡、カウント機能を提供します。

## 機能

- **人物検知**: YOLOv8モデルを使用した高精度な人物検知
- **人物追跡**: フレーム間での人物追跡機能
- **カウント機能**: 設定可能なカウントラインとゾーンによる人物カウント
- **可視化**: リアルタイムでの検知結果表示
- **設定可能**: JSON設定ファイルによる柔軟なカスタマイズ

## 対応クラス

- 人 (person)

## インストール

### 前提条件

- Python 3.8以上
- YOLOv8モデルファイル (`yolov8n.pt`)

### 依存関係のインストール

#### pipを使用する場合

```bash
pip install -r requirements.txt
```

または

```bash
pip install opencv-python==4.8.1.78 ultralytics==8.0.196 numpy==1.24.3 matplotlib==3.7.2 Pillow==10.0.0
```

#### uvを使用する場合（推奨）

```bash
# pyproject.tomlから依存関係をインストール
uv sync

# または、仮想環境を作成してインストール
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
uv sync
```

### YOLOv8モデルのダウンロード

初回実行時に自動的にダウンロードされますが、手動でダウンロードする場合：

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

## 設定ファイル (config.json)

設定ファイルは以下の構造で構成されています：

### モデル設定

```json
"model": {
    "path": "yolov8n.pt",           // YOLOv8モデルファイルのパス
    "confidence_threshold": 0.6     // 検知信頼度の閾値 (0.0-1.0)
}
```

### 検知設定

```json
"detection": {
    "vehicle_classes": {            // 検知対象のクラス
        "0": "person"               // 人
    },
    "tracking_history_frames": 10   // 追跡履歴を保持するフレーム数
}
```

### カウント設定

```json
"counting": {
    "line_angle": 0.0,              // カウントラインの角度（度数、0は水平）
    "line_ratio": 0.5,              // カウントラインの画面高さに対する割合 (0.0-1.0)
    "zone_ratio": 0.9,              // カウントゾーンの画面幅に対する割合 (0.0-1.0)
    "direction": "downward"         // カウント方向 ("upward", "downward", "both")
}
```

**カウント方向の説明:**
- `"upward"`: 下から上への移動をカウント
- `"downward"`: 上から下への移動をカウント
- `"both"`: 両方向の移動をカウント

### 表示設定

```json
"display": {
    "show_video": true,             // 動画表示の有効/無効
    "save_screenshots": true,       // スクリーンショット保存の有効/無効
    "progress_interval": 30,        // 進捗表示のフレーム間隔
        "colors": {                     // 表示色の設定 (BGR形式)
            "counted_vehicle": [0, 255, 0],      // カウント済み人物（緑）
            "uncounted_vehicle": [0, 0, 255],    // 未カウント人物（赤）
            "counting_line": [255, 0, 0],        // カウントライン（青）
            "counting_zone": [255, 255, 0],      // カウントゾーン（シアン）
            "vehicle_count_text": [128, 0, 0]    // 人物数テキスト（茶色）
        }
}
```

### 出力設定

```json
"output": {
    "save_video": false,            // 出力動画保存の有効/無効
    "video_codec": "mp4v",          // 出力動画のコーデック
    "screenshot_format": "jpg"      // スクリーンショットのフォーマット
}
```

## 実行方法

### 基本的な実行

```bash
python configurable_detector.py 動画ファイルパス
```

### 設定ファイルを指定して実行

```bash
python configurable_detector.py 動画ファイルパス --config カスタム設定.json
```

### 出力動画を指定して実行

```bash
python configurable_detector.py 動画ファイルパス --output 出力動画.mp4
```

### 例

```bash
# 基本的な実行
python configurable_detector.py inputs/1900-151662242_small.mp4

# カスタム設定で実行
python configurable_detector.py inputs/1900-151662242_small.mp4 --config my_config.json

# 出力動画を保存
python configurable_detector.py inputs/1900-151662242_small.mp4 --output result.mp4
```

## 操作方法

実行中に以下のキー操作が可能です：

- **`q`**: 処理を終了
- **`s`**: 現在のフレームをスクリーンショットとして保存（`save_screenshots: true`の場合）

## 出力

### コンソール出力

処理中に以下の情報が表示されます：

- 処理済みフレーム数
- 検知された人物数
- 処理速度（FPS）
- 最終的な統計情報

### ファイル出力

設定に応じて以下のファイルが生成されます：

- **出力動画**: `save_video: true`の場合
- **スクリーンショット**: `s`キーを押した場合、または`save_screenshots: true`の場合

## カスタマイズ例

### 水平カウントラインの設定

```json
"counting": {
    "line_angle": 0.0,
    "line_ratio": 0.6,
    "direction": "downward"
}
```

### 斜めカウントラインの設定

```json
"counting": {
    "line_angle": 15.0,
    "line_ratio": 0.5,
    "direction": "both"
}
```

### 高精度検知の設定

```json
"model": {
    "confidence_threshold": 0.8
}
```

### 両方向カウントの設定

```json
"counting": {
    "direction": "both"
}
```


