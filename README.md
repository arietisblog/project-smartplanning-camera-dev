# リアルタイムオブジェクト検知Webアプリケーション

YOLOv8を使用したリアルタイムオブジェクト検知・カウントシステムのWebアプリケーション版です。

## プロジェクト構成

```
project-smartplanning-camera-dev/
├── logic_base/           # 元の検知ロジック（Python）
├── backend/             # FastAPI バックエンド
├── frontend/            # NextJS フロントエンド
├── docker-compose.yml   # Docker Compose設定
├── start.sh            # 起動スクリプト
├── stop.sh             # 停止スクリプト
└── README.md
```

## 機能

- **リアルタイム動画検知**: WebSocketを使用したリアルタイム動画処理
- **動画アップロード**: ブラウザから動画ファイルをアップロード
- **設定可能**: 検知対象クラス、信頼度閾値、カウントライン設定
- **リアルタイム表示**: 検知結果をリアルタイムでブラウザに表示
- **カウント機能**: オブジェクトの通過を自動カウント

## セットアップ

### 🐳 Docker での一発起動（推奨）

#### 開発モード（ホットリロード対応）
```bash
# 開発アプリケーションを起動
./start.sh dev

# 停止
./stop.sh dev
```

#### 本番モード（最適化済み）
```bash
# 本番アプリケーションを起動
./start.sh prod

# 停止
./stop.sh prod
```

#### 手動起動
```bash
# 環境変数ファイルを作成
cp env.example .env

# 開発モード
docker-compose -f docker-compose.dev.yml up -d

# 本番モード
docker-compose -f docker-compose.prod.yml up -d

# 停止
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.prod.yml down
```

### 🔧 手動セットアップ

#### 1. バックエンドのセットアップ

```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### 2. フロントエンドのセットアップ

```bash
cd frontend
npm install
npm run dev
```

## 使用方法

1. **動画アップロード**: ブラウザで動画ファイルをアップロード
2. **設定調整**: 検知対象クラス、信頼度閾値、カウントライン設定を調整
3. **検知開始**: 「検知開始」ボタンをクリック
4. **リアルタイム表示**: 検知結果がリアルタイムで表示されます

## アクセスURL

- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

## 便利なコマンド

### 開発モード
```bash
# ログを確認
docker-compose -f docker-compose.dev.yml logs -f

# 特定のサービスのログを確認
docker-compose -f docker-compose.dev.yml logs -f backend
docker-compose -f docker-compose.dev.yml logs -f frontend

# サービスを再起動
docker-compose -f docker-compose.dev.yml restart

# 完全にクリーンアップ
docker-compose -f docker-compose.dev.yml down -v --rmi all

# イメージを再ビルド
docker-compose -f docker-compose.dev.yml build --no-cache
```

### 本番モード
```bash
# ログを確認
docker-compose -f docker-compose.prod.yml logs -f

# 特定のサービスのログを確認
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f frontend

# サービスを再起動
docker-compose -f docker-compose.prod.yml restart

# 完全にクリーンアップ
docker-compose -f docker-compose.prod.yml down -v --rmi all

# イメージを再ビルド
docker-compose -f docker-compose.prod.yml build --no-cache
```

## 技術スタック

### バックエンド
- **FastAPI**: Web API フレームワーク
- **WebSocket**: リアルタイム通信
- **OpenCV**: 動画処理
- **YOLOv8**: オブジェクト検知
- **Python**: メイン言語

### フロントエンド
- **Next.js 16**: React フレームワーク
- **TypeScript**: 型安全なJavaScript
- **Tailwind CSS**: スタイリング
- **shadcn/ui**: UI コンポーネント
- **WebSocket**: リアルタイム通信

## 対応クラス

YOLOv8で検知可能な80種類のオブジェクトに対応：

- **乗り物**: person, bicycle, car, motorcycle, bus, truck など
- **動物**: cat, dog, horse, sheep, cow など
- **その他**: 家具、家電、食べ物など

詳細は `logic_base/YOLOv8_Classes.md` を参照してください。

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。
