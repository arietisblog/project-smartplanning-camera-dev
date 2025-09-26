'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Upload, Play, Square, Settings } from 'lucide-react'

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
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">リアルタイムオブジェクト検知システム</h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 設定パネル */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  設定
                </CardTitle>
                <CardDescription>
                  検知対象とパラメータを設定してください
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* 動画アップロード */}
                <div>
                  <Label htmlFor="video-upload">動画ファイル</Label>
                  <Input
                    id="video-upload"
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    className="mt-1"
                  />
                  {videoFile && (
                    <p className="text-sm text-gray-600 mt-1">
                      選択中: {videoFile.name}
                    </p>
                  )}
                </div>

                {/* 検知対象クラス */}
                <div>
                  <Label>検知対象クラス</Label>
                  <div className="space-y-2 mt-2">
                    {Object.entries(config.object_classes).map(([key, value]) => (
                      <div key={key} className="flex items-center gap-2">
                        <Select
                          value={key}
                          onValueChange={(newKey) => {
                            const newClasses = { ...config.object_classes }
                            delete newClasses[key]
                            newClasses[newKey] = availableClasses[newKey]
                            setConfig(prev => ({ ...prev, object_classes: newClasses }))
                          }}
                        >
                          <SelectTrigger className="w-32">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(availableClasses).map(([k, v]) => (
                              <SelectItem key={k} value={k}>
                                {v}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <span className="text-sm text-gray-600 flex-1">
                          {availableClasses[key] || value}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => removeObjectClass(key)}
                        >
                          削除
                        </Button>
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addObjectClass}
                      className="w-full"
                    >
                      クラスを追加
                    </Button>
                  </div>
                </div>

                {/* 信頼度閾値 */}
                <div>
                  <Label htmlFor="confidence">信頼度閾値: {config.confidence_threshold}</Label>
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
                    className="mt-1"
                  />
                </div>

                {/* カウントライン設定 */}
                <div>
                  <Label htmlFor="line-ratio">カウントライン位置: {config.line_ratio}</Label>
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
                    className="mt-1"
                  />
                </div>

                {/* カウント方向 */}
                <div>
                  <Label htmlFor="direction">カウント方向</Label>
                  <Select
                    value={config.direction}
                    onValueChange={(value) => setConfig(prev => ({ ...prev, direction: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="upward">上向き</SelectItem>
                      <SelectItem value="downward">下向き</SelectItem>
                      <SelectItem value="both">両方向</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* 制御ボタン */}
                <div className="flex gap-2">
                  <Button
                    onClick={startDetection}
                    disabled={!uploadedFileId || isProcessing}
                    className="flex-1"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    検知開始
                  </Button>
                  <Button
                    onClick={stopDetection}
                    disabled={!isProcessing}
                    variant="destructive"
                    className="flex-1"
                  >
                    <Square className="h-4 w-4 mr-2" />
                    停止
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 動画表示パネル */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>リアルタイム検知結果</CardTitle>
                <CardDescription>
                  検知されたオブジェクト数: {objectCount}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-black rounded-lg overflow-hidden">
                  {currentFrame ? (
                    <img
                      src={`data:image/jpeg;base64,${currentFrame}`}
                      alt="検知結果"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-white">
                      動画をアップロードして検知を開始してください
                    </div>
                  )}
                </div>

                {isProcessing && (
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>処理中...</span>
                      <span>{frameCount} フレーム</span>
                    </div>
                    <Progress value={progress} className="w-full" />
                    <div className="text-sm text-gray-600">
                      FPS: {fps.toFixed(1)}
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