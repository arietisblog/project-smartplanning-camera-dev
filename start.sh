#!/bin/bash

# デフォルトは開発モード
MODE=${1:-dev}

echo "🚀 リアルタイムオブジェクト検知Webアプリケーションを起動中... (モード: $MODE)"

# Dockerがインストールされているかチェック
if ! command -v docker &> /dev/null; then
    echo "❌ Dockerがインストールされていません。"
    echo "Dockerをインストールしてから再実行してください。"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Composeがインストールされていません。"
    echo "Docker Composeをインストールしてから再実行してください。"
    exit 1
fi

# 必要なディレクトリを作成
echo "📁 必要なディレクトリを作成中..."
mkdir -p backend/temp_videos
mkdir -p backend/outputs

# 環境変数ファイルをコピー（存在しない場合）
if [ ! -f .env ]; then
    echo "📝 環境変数ファイルを作成中..."
    cp env.example .env
fi

# Dockerイメージをビルドして起動
echo "🔨 Dockerイメージをビルド中..."

if [ "$MODE" = "prod" ]; then
    echo "📦 本番モードでビルド中..."
    docker-compose -f docker-compose.prod.yml build
    echo "🚀 本番アプリケーションを起動中..."
    docker-compose -f docker-compose.prod.yml up -d
else
    echo "🛠️ 開発モードでビルド中..."
    docker-compose -f docker-compose.dev.yml build
    echo "🚀 開発アプリケーションを起動中..."
    docker-compose -f docker-compose.dev.yml up -d
fi

echo "⏳ サービスが起動するまで待機中..."
sleep 10

# サービスが起動しているかチェック
echo "🔍 サービス状態を確認中..."
if [ "$MODE" = "prod" ]; then
    docker-compose -f docker-compose.prod.yml ps
else
    docker-compose -f docker-compose.dev.yml ps
fi

echo ""
echo "✅ アプリケーションが起動しました！"
echo ""
echo "🌐 アクセスURL:"
echo "   フロントエンド: http://localhost:3000"
echo "   バックエンドAPI: http://localhost:8000"
echo "   API ドキュメント: http://localhost:8000/docs"
echo ""
echo "📋 便利なコマンド:"
echo "   ログを確認: docker-compose logs -f"
echo "   停止: docker-compose down"
echo "   再起動: docker-compose restart"
echo ""
echo "🎉 ブラウザで http://localhost:3000 にアクセスしてアプリケーションをお楽しみください！"
