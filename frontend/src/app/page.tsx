'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Upload, Play, Square, Settings, Video, BarChart3 } from 'lucide-react'

interface DetectionConfig {
  object_classes: Record<string, string>
  confidence_threshold: number
  line_ratio: number
  line_angle: number
  zone_ratio: number
  direction: string
}

interface WebSocketMessage {
  type: 'frame' | 'progress' | 'complete' | 'error'
  frame?: string
  object_count?: number
  frame_count?: number
  fps?: number
  total_frames?: number
  final_count?: number
  processing_time?: number
  message?: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [uploadedFileId, setUploadedFileId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentFrame, setCurrentFrame] = useState<string | null>(null)
  const [objectCount, setObjectCount] = useState(0)
  const [frameCount, setFrameCount] = useState(0)
  const [fps, setFps] = useState(0)
  const [progress, setProgress] = useState(0)
  const [availableClasses, setAvailableClasses] = useState<Record<string, string>>({})
  const [isClient, setIsClient] = useState(false)
  const [config, setConfig] = useState<DetectionConfig>({
    object_classes: { '1': 'bicycle' },
    confidence_threshold: 0.6,
    line_ratio: 0.5,
    line_angle: 0.0,
    zone_ratio: 0.9,
    direction: 'both'
  })

  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // 利用可能なクラスを取得
  useEffect(() => {
    setIsClient(true)
    const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
    fetch(`${apiUrl}/get-classes`)
      .then(res => res.json())
      .then(data => setAvailableClasses(data))
      .catch(console.error)
  }, [])

  // WebSocket接続
  useEffect(() => {
    if (isProcessing) {
      // ブラウザからアクセスする場合はlocalhostを使用
      const wsUrl = API_BASE_URL.replace('http://backend', 'ws://localhost').replace('http', 'ws')
      wsRef.current = new WebSocket(`${wsUrl}/ws`)

      wsRef.current.onmessage = (event) => {
        const message: WebSocketMessage = JSON.parse(event.data)

        switch (message.type) {
          case 'frame':
            if (message.frame) {
              setCurrentFrame(message.frame)
              setObjectCount(message.object_count || 0)
              setFrameCount(message.frame_count || 0)
              setFps(message.fps || 0)
            }
            break
          case 'progress':
            setObjectCount(message.object_count || 0)
            setFrameCount(message.frame_count || 0)
            setFps(message.fps || 0)
            break
          case 'complete':
            setIsProcessing(false)
            setProgress(100)
            alert(`処理完了！検知されたオブジェクト数: ${message.final_count}`)
            break
          case 'error':
            setIsProcessing(false)
            alert(`エラー: ${message.message}`)
            break
        }
      }

      wsRef.current.onclose = () => {
        setIsProcessing(false)
      }
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [isProcessing])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setVideoFile(file)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      const response = await fetch(`${apiUrl}/upload-video`, {
        method: 'POST',
        body: formData
      })

      const result = await response.json()
      setUploadedFileId(result.file_id)
    } catch (error) {
      console.error('アップロードエラー:', error)
      alert('動画のアップロードに失敗しました')
    }
  }

  const startDetection = async () => {
    if (!uploadedFileId) {
      alert('動画をアップロードしてください')
      return
    }

    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      const response = await fetch(`${apiUrl}/start-detection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          config,
          file_path: `temp_videos/${uploadedFileId}_${videoFile?.name}`
        })
      })

      if (response.ok) {
        setIsProcessing(true)
        setProgress(0)
        setObjectCount(0)
        setFrameCount(0)
      } else {
        alert('検知処理の開始に失敗しました')
      }
    } catch (error) {
      console.error('検知開始エラー:', error)
      alert('検知処理の開始に失敗しました')
    }
  }

  const stopDetection = async () => {
    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      await fetch(`${apiUrl}/stop-detection`, {
        method: 'POST'
      })
      setIsProcessing(false)
    } catch (error) {
      console.error('停止エラー:', error)
    }
  }

  const addObjectClass = () => {
    const newKey = Object.keys(availableClasses).find(key => !config.object_classes[key])
    if (newKey) {
      setConfig(prev => ({
        ...prev,
        object_classes: {
          ...prev.object_classes,
          [newKey]: availableClasses[newKey]
        }
      }))
    }
  }

  const removeObjectClass = (key: string) => {
    setConfig(prev => {
      const newClasses = { ...prev.object_classes }
      delete newClasses[key]
      return {
        ...prev,
        object_classes: newClasses
      }
    })
  }

  return (
    <div className="min-h-screen bg-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12 animate-slide-in">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            リアルタイムオブジェクト検知システム
          </h1>
          <p className="text-xl text-gray-400">
            AI搭載の高度な動画解析プラットフォーム
          </p>
          C        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 設定パネル */}
          <div className="space-y-6">
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
                <CardTitle className="flex items-center gap-3 text-white">
                  <Settings className="h-5 w-5 text-blue-400" />
                  <span className="text-lg font-semibold">
                    設定パネル
                  </span>
                </CardTitle>
                <CardDescription className="text-gray-300">
                  検知対象とパラメータを設定してください
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6 p-6">
                {/* 動画アップロード */}
                <div className="space-y-2">
                  <Label htmlFor="video-upload" className="text-white font-semibold flex items-center gap-2">
                    <Upload className="h-4 w-4 text-blue-400" />
                    動画ファイル
                  </Label>
                  <div className="relative">
                    <Input
                      id="video-upload"
                      type="file"
                      accept="video/*"
                      onChange={handleFileUpload}
                      className="mt-1 bg-slate-800 border-slate-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  {videoFile && (
                    <div className="flex items-center gap-2 p-3 bg-green-500/20 border border-green-500/30 rounded-lg">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <p className="text-sm text-green-300">
                        選択中: {videoFile.name}
                      </p>
                    </div>
                  )}
                </div>

                {/* 検知対象クラス */}
                <div className="space-y-3">
                  <Label className="text-white font-semibold">検知対象クラス</Label>
                  <div className="space-y-3">
                    {Object.entries(config.object_classes).map(([key, value]) => (
                      <div key={key} className="flex items-center gap-3 p-3 bg-white/5 border border-white/10 rounded-lg">
                        <Select
                          value={key}
                          onValueChange={(newKey) => {
                            const newClasses = { ...config.object_classes }
                            delete newClasses[key]
                            newClasses[newKey] = availableClasses[newKey]
                            setConfig(prev => ({ ...prev, object_classes: newClasses }))
                          }}
                        >
                          <SelectTrigger className="w-32 bg-slate-800 border-slate-600 text-white">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="bg-slate-800 border-slate-700">
                            {Object.entries(availableClasses).map(([k, v]) => (
                              <SelectItem key={k} value={k} className="text-white hover:bg-slate-700">
                                {v}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <span className="text-sm text-gray-300 flex-1 font-medium">
                          {availableClasses[key] || value}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => removeObjectClass(key)}
                          className="border-red-500/50 text-red-400 hover:bg-red-500/20"
                        >
                          削除
                        </Button>
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addObjectClass}
                      className="w-full border-blue-500/50 text-blue-400 hover:bg-blue-500/20"
                    >
                      + クラスを追加
                    </Button>
                  </div>
                </div>

                {/* 信頼度閾値 */}
                <div className="space-y-3">
                  <Label htmlFor="confidence" className="text-white font-semibold">
                    信頼度閾値: <span className="text-blue-400 font-bold">{config.confidence_threshold}</span>
                  </Label>
                  <div className="relative">
                    <Input
                      id="confidence"
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={config.confidence_threshold}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        confidence_threshold: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>0.1</span>
                      <span>1.0</span>
                    </div>
                  </div>
                </div>

                {/* カウントライン設定 */}
                <div className="space-y-3">
                  <Label htmlFor="line-ratio" className="text-white font-semibold">
                    カウントライン位置: <span className="text-purple-400 font-bold">{config.line_ratio}</span>
                  </Label>
                  <div className="relative">
                    <Input
                      id="line-ratio"
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={config.line_ratio}
                      onChange={(e) => setConfig(prev => ({
                        ...prev,
                        line_ratio: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>0.1</span>
                      <span>0.9</span>
                    </div>
                  </div>
                </div>

                {/* カウント方向 */}
                <div className="space-y-3">
                  <Label htmlFor="direction" className="text-white font-semibold">カウント方向</Label>
                  <Select
                    value={config.direction}
                    onValueChange={(value) => setConfig(prev => ({ ...prev, direction: value }))}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-600 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="upward" className="text-white hover:bg-slate-700">上向き</SelectItem>
                      <SelectItem value="downward" className="text-white hover:bg-slate-700">下向き</SelectItem>
                      <SelectItem value="both" className="text-white hover:bg-slate-700">両方向</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* 制御ボタン */}
                <div className="flex gap-3 pt-4">
                  <Button
                    onClick={startDetection}
                    disabled={!uploadedFileId || isProcessing}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 rounded-lg shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play className="h-5 w-5 mr-2" />
                    {isProcessing ? '処理中...' : '検知開始'}
                  </Button>
                  <Button
                    onClick={stopDetection}
                    disabled={!isProcessing}
                    className="flex-1 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-semibold py-3 rounded-lg shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Square className="h-5 w-5 mr-2" />
                    停止
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 動画表示パネル */}
          <div className="space-y-6">
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
              <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
                <CardTitle className="flex items-center gap-3 text-white">
                  <Video className="h-5 w-5 text-purple-400" />
                  <span className="text-lg font-semibold">
                    リアルタイム検知結果
                  </span>
                </CardTitle>
                <CardDescription className="text-gray-300">
                  検知されたオブジェクト数: <span className="text-green-400 font-bold">{objectCount}</span>
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <div className="aspect-video bg-slate-900 rounded-xl flex items-center justify-center border border-slate-700 overflow-hidden">
                  {currentFrame ? (
                    <img
                      src={`data:image/jpeg;base64,${currentFrame}`}
                      alt="検知結果"
                      className="w-full h-full object-contain rounded-lg shadow-2xl"
                    />
                  ) : (
                    <div className="text-center text-gray-400">
                      <Video className="h-16 w-16 mx-auto mb-4 opacity-30" />
                      <p className="text-lg font-medium">動画をアップロードして検知を開始してください</p>
                      <p className="text-sm opacity-75 mt-2">検知開始後にリアルタイムで結果が表示されます</p>
                    </div>
                  )}
                </div>

                {isProcessing && (
                  <div className="mt-6 space-y-4 p-4 bg-slate-800/30 rounded-lg border border-slate-700">
                    <div className="flex justify-between text-sm text-white">
                      <span className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        処理中...
                      </span>
                      <span className="text-blue-400 font-semibold">{frameCount} フレーム</span>
                    </div>
                    <Progress value={progress} className="w-full h-2" />
                    <div className="text-sm text-gray-300 flex justify-between">
                      <span>FPS: <span className="text-green-400 font-bold">{fps.toFixed(1)}</span></span>
                      <span>進捗: <span className="text-purple-400 font-bold">{progress.toFixed(1)}%</span></span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}