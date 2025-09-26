#!/bin/bash

# デフォルトは開発モード
MODE=${1:-dev}

echo "🛑 リアルタイムオブジェクト検知Webアプリケーションを停止中... (モード: $MODE)"

# Docker Composeでサービスを停止
if [ "$MODE" = "prod" ]; then
    docker-compose -f docker-compose.prod.yml down
else
    docker-compose -f docker-compose.dev.yml down
fi

echo "✅ アプリケーションが停止しました！"
echo ""
echo "📋 その他のコマンド:"
echo "   完全にクリーンアップ: docker-compose down -v --rmi all"
echo "   ログを確認: docker-compose logs"
echo "   再起動: ./start.sh"
