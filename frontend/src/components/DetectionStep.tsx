import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Play, Square, ArrowLeft, Video, Download, CheckCircle, RotateCw, AlertCircle, FileText } from 'lucide-react'
import { useState } from 'react'

interface DetectionStepProps {
  currentFrame: string | null
  objectCount: number
  classCounts: { [key: string]: number }
  availableClasses: Record<string, string>
  isProcessing: boolean
  frameCount: number
  fps: number
  progress: number
  isCompleted: boolean
  uploadedFileId: string | null
  onStopDetection: () => void
  onBack: () => void
  onDownloadVideo: () => void
  onDownloadCSV: () => void
}

export default function DetectionStep({
  currentFrame,
  objectCount,
  classCounts,
  availableClasses,
  isProcessing,
  frameCount,
  fps,
  progress,
  isCompleted,
  uploadedFileId,
  onStopDetection,
  onBack,
  onDownloadVideo,
  onDownloadCSV
}: DetectionStepProps) {
  const [videoError, setVideoError] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 制御パネル */}
      <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
        <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
          <CardTitle className="flex items-center gap-3 text-white">
            {isCompleted ? (
              <CheckCircle className="h-5 w-5 text-green-400" />
            ) : isProcessing ? (
              <RotateCw className="h-5 w-5 text-blue-400 animate-spin" />
            ) : (
              <Play className="h-5 w-5 text-purple-400" />
            )}
            <span className="text-lg font-semibold">
              ステップ3: 検知実行
            </span>
          </CardTitle>
          <CardDescription className="text-gray-300">
            {isCompleted ? (
              "検知処理が完了しました"
            ) : isProcessing ? (
              "リアルタイムでオブジェクト検知を実行中です"
            ) : (
              "検知処理を開始してください"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-4">
          {/* 制御ボタン */}
          <div className="flex gap-3 pt-2">
            {isCompleted ? (
              <>
                <Button
                  onClick={onBack}
                  variant="outline"
                  size="sm"
                  className="border-slate-500 bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white"
                >
                  <ArrowLeft className="h-3 w-3 mr-1" />
                  戻る
                </Button>
                <div className="flex gap-2">
                  <Button
                    onClick={async () => {
                      setIsDownloading(true)
                      try {
                        await onDownloadVideo()
                      } finally {
                        setIsDownloading(false)
                      }
                    }}
                    disabled={isDownloading}
                    className="flex-1 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-semibold py-2 rounded-lg shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isDownloading ? (
                      <>
                        <RotateCw className="h-4 w-4 mr-2 animate-spin" />
                        ダウンロード中...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        動画ダウンロード
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={onDownloadCSV}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold py-2 rounded-lg shadow-lg transition-all duration-300"
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    CSVダウンロード
                  </Button>
                </div>
              </>
            ) : isProcessing ? (
              <div className="flex-1 flex items-center justify-center py-2">
                <div className="flex items-center gap-2 text-blue-400">
                  <RotateCw className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-medium">検知処理中...</span>
                </div>
              </div>
            ) : (
              <Button
                onClick={onBack}
                variant="outline"
                size="sm"
                className="flex-1 border-slate-500 bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white"
              >
                <ArrowLeft className="h-3 w-3 mr-1" />
                戻る
              </Button>
            )}
          </div>

          {/* 検知統計 */}
          <div className="space-y-3 p-3 bg-slate-800/30 rounded-lg border border-slate-700">
            {/* クラス別カウント */}
            {Object.keys(classCounts).length > 0 && (
              <div className="space-y-2">
                {Object.entries(classCounts).map(([classId, count]) => (
                  <div key={classId} className="flex justify-between items-center">
                    <span className="text-lg text-gray-300 font-bold">
                      {availableClasses[classId] || `クラス${classId}`}
                    </span>
                    <span className="text-xl text-purple-400 font-extrabold">{count}</span>
                  </div>
                ))}
                <div className="border-t border-slate-600 pt-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300 font-semibold">合計</span>
                    <span className="text-lg text-green-400 font-bold">{objectCount}</span>
                  </div>
                </div>
              </div>
            )}

            {/* クラス別カウントがない場合は従来の表示 */}
            {Object.keys(classCounts).length === 0 && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300">検知されたオブジェクト数</span>
                <span className="text-lg text-green-400 font-bold">{objectCount}</span>
              </div>
            )}

            {isProcessing && (
              <>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">処理フレーム数</span>
                  <span className="text-sm text-blue-400 font-semibold">{frameCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">FPS</span>
                  <span className="text-sm text-green-400 font-bold">{fps.toFixed(1)}</span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>進捗</span>
                    <span>{progress.toFixed(1)}%</span>
                  </div>
                  <Progress value={progress} className="w-full h-1" />
                </div>
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  処理中...
                </div>
              </>
            )}

            {isCompleted && (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">最終フレーム数</span>
                  <span className="text-sm text-blue-400 font-semibold">{frameCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">平均FPS</span>
                  <span className="text-sm text-green-400 font-bold">{fps.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <CheckCircle className="w-3 h-3" />
                  処理完了
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 動画表示パネル */}
      <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
        <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
          <CardTitle className="flex items-center gap-3 text-white">
            <Video className="h-5 w-5 text-purple-400" />
            <span className="text-lg font-semibold">
              {isCompleted ? "検知完了結果" : "リアルタイム検知結果"}
            </span>
          </CardTitle>
          <CardDescription className="text-gray-300">
            {isCompleted ? (
              `検知完了 - 総オブジェクト数: ${objectCount}`
            ) : (
              `検知されたオブジェクト数: ${objectCount}`
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="p-4">
          <div className="aspect-video bg-slate-900 rounded-xl flex items-center justify-center border border-slate-700 overflow-hidden relative">
            {isCompleted && uploadedFileId ? (
              videoError ? (
                <div className="text-center text-gray-400">
                  <AlertCircle className="h-16 w-16 mx-auto mb-4 opacity-30 text-red-400" />
                  <p className="text-lg font-medium text-red-400">動画の再生に失敗しました</p>
                  <p className="text-sm opacity-75 mt-2">ダウンロードボタンから動画をダウンロードしてください</p>
                </div>
              ) : (
                <video
                  src={`http://localhost:8000/stream-video/${uploadedFileId}`}
                  controls
                  className="w-full h-full object-contain rounded-lg shadow-2xl"
                  autoPlay
                  muted
                  loop
                  onError={(e) => {
                    console.error('動画再生エラー:', e);
                    const target = e.target as HTMLVideoElement;
                    console.error('エラー詳細:', target.error);
                    setVideoError(true);
                  }}
                  onLoadStart={() => console.log('動画読み込み開始')}
                  onCanPlay={() => console.log('動画再生可能')}
                  onLoadedData={() => console.log('動画データ読み込み完了')}
                >
                  お使いのブラウザは動画の再生をサポートしていません。
                </video>
              )
            ) : currentFrame ? (
              <img
                src={`data:image/jpeg;base64,${currentFrame}`}
                alt="検知結果"
                className="w-full h-full object-contain rounded-lg shadow-2xl"
              />
            ) : isProcessing ? (
              <div className="text-center text-gray-400">
                <RotateCw className="h-16 w-16 mx-auto mb-4 opacity-30 animate-spin" />
                <p className="text-lg font-medium">検知処理中...</p>
                <p className="text-sm opacity-75 mt-2">リアルタイムで検知結果が表示されます</p>
              </div>
            ) : (
              <div className="text-center text-gray-400">
                <Video className="h-16 w-16 mx-auto mb-4 opacity-30" />
                <p className="text-lg font-medium">検知処理を開始してください</p>
                <p className="text-sm opacity-75 mt-2">リアルタイムで検知結果が表示されます</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
