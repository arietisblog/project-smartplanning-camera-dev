'use client'

import { useState, useRef, useEffect } from 'react'
import ProgressIndicator from '@/components/ProgressIndicator'
import UploadStep from '@/components/UploadStep'
import PreviewStep from '@/components/PreviewStep'
import DetectionStep from '@/components/DetectionStep'

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
  const [showPreview, setShowPreview] = useState(true)
  const [videoThumbnail, setVideoThumbnail] = useState<string | null>(null)
  const [isPreviewMode, setIsPreviewMode] = useState(false)
  const [currentStep, setCurrentStep] = useState<'upload' | 'preview' | 'detection'>('upload')
  const [isCompleted, setIsCompleted] = useState(false)
  const [config, setConfig] = useState<DetectionConfig>({
    object_classes: { '1': 'bicycle' },
    confidence_threshold: 0.6,
    line_ratio: 0.5,
    line_angle: 0.0,
    zone_ratio: 0.8,
    direction: 'right'
  })

  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    // 利用可能なクラスを取得
    const fetchClasses = async () => {
      try {
        const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
        const response = await fetch(`${apiUrl}/get-classes`)
        if (response.ok) {
          const classes = await response.json()
          setAvailableClasses(classes)
        }
      } catch (error) {
        console.error('クラス取得エラー:', error)
      }
    }

    fetchClasses()
  }, [])

  useEffect(() => {
    // 設定変更時にバックエンドに送信（検知中のみ）
    if (uploadedFileId && currentStep === 'detection' && isProcessing) {
      updateConfig()
    }
  }, [config, uploadedFileId, currentStep, isProcessing])

  const generateThumbnail = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video')
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')

      video.addEventListener('loadeddata', () => {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx?.drawImage(video, 0, 0)
        const thumbnail = canvas.toDataURL('image/jpeg', 0.8)
        resolve(thumbnail)
      })

      video.addEventListener('error', reject)
      video.src = URL.createObjectURL(file)
      video.currentTime = 1 // 1秒後にサムネイルを生成
    })
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setVideoFile(file)

    // サムネイルを生成
    try {
      const thumbnail = await generateThumbnail(file)
      setVideoThumbnail(thumbnail)
      setIsPreviewMode(true) // プレビューモードに切り替え
      setCurrentStep('preview') // プレビューステップに進む
    } catch (error) {
      console.error('サムネイル生成エラー:', error)
    }

    const formData = new FormData()
    formData.append('file', file)

    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      const response = await fetch(`${apiUrl}/upload-video`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        setUploadedFileId(result.file_id)
      } else {
        alert('動画のアップロードに失敗しました')
      }
    } catch (error) {
      console.error('アップロードエラー:', error)
      alert('動画のアップロードに失敗しました')
    }
  }

  const updateConfig = async () => {
    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      await fetch(`${apiUrl}/update-config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          config: {
            confidence_threshold: config.confidence_threshold,
            object_classes: config.object_classes,
            line_ratio: config.line_ratio,
            line_angle: config.line_angle,
            zone_ratio: config.zone_ratio,
            direction: config.direction
          }
        })
      })
    } catch (error) {
      console.error('設定更新エラー:', error)
    }
  }

  const startDetection = async () => {
    if (!uploadedFileId || !videoFile) return

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
        setIsCompleted(false)
        setIsPreviewMode(false) // プレビューモードを終了
        setCurrentStep('detection') // 検知ステップに進む
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
      setCurrentStep('preview') // プレビューステップに戻る
    } catch (error) {
      console.error('停止エラー:', error)
    }
  }

  const downloadVideo = async () => {
    if (!uploadedFileId || !videoFile) return

    try {
      const apiUrl = API_BASE_URL.replace('http://backend', 'http://localhost')
      const response = await fetch(`${apiUrl}/download-video/${uploadedFileId}`)

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `detected_${videoFile.name}`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      } else {
        alert('動画のダウンロードに失敗しました')
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
    } catch (error) {
      console.error('ダウンロードエラー:', error)
      alert('動画のダウンロードに失敗しました')
      throw error
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
      return { ...prev, object_classes: newClasses }
    })
  }

  const drawPreviewOverlay = (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => {
    const canvasWidth = canvas.width
    const canvasHeight = canvas.height

    // 画像の実際の表示サイズとアスペクト比を計算
    const img = canvas.parentElement?.querySelector('img')
    if (!img) return

    const imgRect = img.getBoundingClientRect()
    const imgAspectRatio = img.naturalWidth / img.naturalHeight
    const canvasAspectRatio = canvasWidth / canvasHeight

    let drawWidth, drawHeight, offsetX, offsetY

    if (imgAspectRatio > canvasAspectRatio) {
      drawWidth = canvasWidth
      drawHeight = canvasWidth / imgAspectRatio
      offsetX = 0
      offsetY = (canvasHeight - drawHeight) / 2
    } else {
      drawWidth = canvasHeight * imgAspectRatio
      drawHeight = canvasHeight
      offsetX = (canvasWidth - drawWidth) / 2
      offsetY = 0
    }

    // カウントラインを描画
    const lineY = offsetY + (drawHeight * config.line_ratio)
    ctx.strokeStyle = '#8b5cf6' // violet-500
    ctx.lineWidth = 3
    ctx.setLineDash([10, 5])
    ctx.beginPath()
    ctx.moveTo(offsetX, lineY)
    ctx.lineTo(offsetX + drawWidth, lineY)
    ctx.stroke()

    // カウントエリアを描画
    const zoneWidth = drawWidth * config.zone_ratio
    const zoneHeight = drawHeight
    const zoneX = offsetX + (drawWidth - zoneWidth) / 2
    const zoneY = offsetY

    ctx.strokeStyle = '#a855f7' // purple-500
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.strokeRect(zoneX, zoneY, zoneWidth, zoneHeight)

    // ラベルを描画
    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(offsetX + 10, lineY - 25, 120, 20)
    ctx.fillStyle = '#8b5cf6'
    ctx.font = 'bold 14px Arial'
    ctx.fillText('カウントライン', offsetX + 15, lineY - 10)

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(zoneX + 10, zoneY + 10, 100, 20)
    ctx.fillStyle = '#a855f7'
    ctx.fillText('カウントエリア', zoneX + 15, zoneY + 25)
  }

  useEffect(() => {
    if (isProcessing) {
      const wsUrl = API_BASE_URL.replace('http://backend', 'ws://localhost').replace('http://', 'ws://')
      wsRef.current = new WebSocket(`${wsUrl}/ws`)

      wsRef.current.onmessage = (event) => {
        const message: WebSocketMessage = JSON.parse(event.data)

        switch (message.type) {
          case 'frame':
            if (message.frame) {
              setCurrentFrame(message.frame)
            }
            if (message.object_count !== undefined) {
              setObjectCount(message.object_count)
            }
            if (message.frame_count !== undefined) {
              setFrameCount(message.frame_count)
            }
            if (message.fps !== undefined) {
              setFps(message.fps)
            }
            break
          case 'progress':
            if (message.total_frames && message.frame_count) {
              setProgress((message.frame_count / message.total_frames) * 100)
            }
            break
          case 'complete':
            setIsProcessing(false)
            setIsCompleted(true)
            if (message.final_count !== undefined) {
              setObjectCount(message.final_count)
            }
            break
          case 'error':
            console.error('WebSocket error:', message.message)
            setIsProcessing(false)
            break
        }
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket connection closed')
      }

      return () => {
        if (wsRef.current) {
          wsRef.current.close()
        }
      }
    }
  }, [isProcessing])

  return (
    <div className="min-h-screen bg-slate-900 p-4 pt-12">
      <div className="max-w-full mx-auto">
        <div className="text-center mb-12 animate-slide-in">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            カメラ検知システムデモ
          </h1>
          <p className="text-xl text-gray-400 mb-8">

          </p>

          <ProgressIndicator currentStep={currentStep} />
        </div>

        <div className="space-y-6">
          {/* 設定系コンテンツ（上部） */}
          <div className="w-full">
            {currentStep === 'upload' && (
              <UploadStep
                videoFile={videoFile}
                onFileUpload={handleFileUpload}
              />
            )}

            {currentStep === 'preview' && (
              <PreviewStep
                config={config}
                setConfig={setConfig}
                availableClasses={availableClasses}
                addObjectClass={addObjectClass}
                removeObjectClass={removeObjectClass}
                videoThumbnail={videoThumbnail}
                showPreview={showPreview}
                setShowPreview={setShowPreview}
                onStartDetection={startDetection}
                onBack={() => setCurrentStep('upload')}
                uploadedFileId={uploadedFileId}
                isProcessing={isProcessing}
                drawPreviewOverlay={drawPreviewOverlay}
              />
            )}

            {currentStep === 'detection' && (
              <DetectionStep
                currentFrame={currentFrame}
                objectCount={objectCount}
                isProcessing={isProcessing}
                frameCount={frameCount}
                fps={fps}
                progress={progress}
                isCompleted={isCompleted}
                uploadedFileId={uploadedFileId}
                onStopDetection={stopDetection}
                onBack={() => setCurrentStep('preview')}
                onDownloadVideo={downloadVideo}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}