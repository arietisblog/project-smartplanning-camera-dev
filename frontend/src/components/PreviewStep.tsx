import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Settings, Play, ArrowLeft, Video } from 'lucide-react'

interface DetectionConfig {
    object_classes: Record<string, string>
    confidence_threshold: number
    line_ratio: number
    line_angle: number
    zone_ratio: number
    direction: string
}

interface PreviewStepProps {
    config: DetectionConfig
    setConfig: React.Dispatch<React.SetStateAction<DetectionConfig>>
    availableClasses: Record<string, string>
    addObjectClass: () => void
    removeObjectClass: (key: string) => void
    videoThumbnail: string | null
    showPreview: boolean
    setShowPreview: (show: boolean) => void
    onStartDetection: () => void
    onBack: () => void
    uploadedFileId: string | null
    isProcessing: boolean
    drawPreviewOverlay: (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => void
}

export default function PreviewStep({
    config,
    setConfig,
    availableClasses,
    addObjectClass,
    removeObjectClass,
    videoThumbnail,
    showPreview,
    setShowPreview,
    onStartDetection,
    onBack,
    uploadedFileId,
    isProcessing,
    drawPreviewOverlay
}: PreviewStepProps) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 設定パネル */}
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
                <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
                    <CardTitle className="flex items-center gap-3 text-white">
                        <Settings className="h-5 w-5 text-purple-400" />
                        <span className="text-lg font-semibold">
                            ステップ2: パラメータ設定
                        </span>
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                        カウントラインとエリアの位置を調整してください
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4 p-4 pt-0">
                    {/* 検知対象クラス */}
                    <div className="space-y-2">
                        <Label className="text-white font-semibold text-sm">検知対象クラス</Label>
                        <div className="space-y-2 w-full">
                            {Object.entries(config.object_classes).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center p-2 bg-white/5 border border-white/10 rounded-lg w-full">
                                    <Select
                                        value={key}
                                        onValueChange={(newKey) => {
                                            const newClasses = { ...config.object_classes }
                                            delete newClasses[key]
                                            newClasses[newKey] = availableClasses[newKey]
                                            setConfig(prev => ({ ...prev, object_classes: newClasses }))
                                        }}
                                    >
                                        <SelectTrigger className="w-32 bg-slate-800 border-slate-600 text-white text-sm">
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
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => removeObjectClass(key)}
                                        className="border-red-500/50 text-red-400 hover:bg-red-500/20 text-xs px-2 py-1"
                                    >
                                        削除
                                    </Button>
                                </div>
                            ))}
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={addObjectClass}
                                className="w-full bg-slate-700 border-slate-600 text-slate-200 hover:bg-slate-600 hover:text-white text-sm py-1"
                            >
                                + クラスを追加
                            </Button>
                        </div>
                    </div>

                    {/* 信頼度閾値 */}
                    <div className="space-y-2">
                        <Label className="text-white font-semibold text-sm">
                            信頼度閾値: {config.confidence_threshold}
                        </Label>
                        <Slider
                            value={[config.confidence_threshold]}
                            onValueChange={(value) => setConfig(prev => ({ ...prev, confidence_threshold: value[0] }))}
                            min={0.1}
                            max={1.0}
                            step={0.05}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-500">
                            検知の信頼度がこの値以上のオブジェクトのみをカウントします
                        </p>
                    </div>

                    {/* カウントライン設定 */}
                    <div className="space-y-2">
                        <Label className="text-white font-semibold text-sm">
                            カウントライン位置: {Math.round(config.line_ratio * 100)}%
                        </Label>
                        <Slider
                            value={[config.line_ratio]}
                            onValueChange={(value) => setConfig(prev => ({ ...prev, line_ratio: value[0] }))}
                            min={0.1}
                            max={0.9}
                            step={0.05}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-500">
                            動画の上から何%の位置にカウントラインを設定するか
                        </p>
                    </div>

                    {/* カウントエリア設定 */}
                    <div className="space-y-2">
                        <Label className="text-white font-semibold text-sm">
                            カウントエリア幅: {Math.round(config.zone_ratio * 100)}%
                        </Label>
                        <Slider
                            value={[config.zone_ratio]}
                            onValueChange={(value) => setConfig(prev => ({ ...prev, zone_ratio: value[0] }))}
                            min={0.1}
                            max={1.0}
                            step={0.05}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-500">
                            動画の幅の何%をカウントエリアとするか
                        </p>
                    </div>

                    {/* カウント方向 */}
                    <div className="space-y-2">
                        <Label className="text-white font-semibold text-sm">カウント方向</Label>
                        <Select
                            value={config.direction}
                            onValueChange={(value) => setConfig(prev => ({ ...prev, direction: value }))}
                        >
                            <SelectTrigger className="bg-slate-800 border-slate-600 text-white text-sm">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-slate-700">
                                <SelectItem value="right" className="text-white hover:bg-slate-700">
                                    上方向
                                </SelectItem>
                                <SelectItem value="left" className="text-white hover:bg-slate-700">
                                    下方向
                                </SelectItem>
                                <SelectItem value="both" className="text-white hover:bg-slate-700">
                                    両方向
                                </SelectItem>
                            </SelectContent>
                        </Select>
                        {/* <p className="text-xs text-gray-400">
                            カウントラインを横切る方向を指定します
                        </p> */}
                    </div>


                    {/* 制御ボタン */}
                    <div className="flex gap-3 pt-2">
                        <Button
                            onClick={onBack}
                            variant="outline"
                            size="sm"
                            className="border-slate-500 bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white"
                        >
                            <ArrowLeft className="h-3 w-3 mr-1" />
                            戻る
                        </Button>
                        <Button
                            onClick={onStartDetection}
                            disabled={!uploadedFileId || isProcessing}
                            size="sm"
                            className="flex-1 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-semibold py-2 rounded-lg shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Play className="h-4 w-4 mr-1" />
                            検知開始
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* 動画表示パネル */}
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
                <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700 py-3">
                    <CardTitle className="flex items-center gap-2 text-white">
                        <Video className="h-4 w-4 text-purple-400" />
                        <span className="text-base font-semibold">
                            プレビュー & パラメータ調整
                        </span>
                    </CardTitle>
                    <CardDescription className="text-gray-300 text-sm">
                        プレビューモード: パラメータを調整してから検証を開始してください
                    </CardDescription>
                </CardHeader>
                <CardContent className="p-4">
                    <div className="aspect-video bg-slate-900 rounded-xl flex items-center justify-center border border-slate-700 overflow-hidden relative">
                        {videoThumbnail ? (
                            <div className="relative w-full h-full">
                                <img
                                    src={videoThumbnail}
                                    alt="動画プレビュー"
                                    className="w-full h-full object-contain rounded-lg shadow-2xl"
                                />
                                {/* プレビューオーバーレイ */}
                                <canvas
                                    ref={(canvas) => {
                                        if (canvas && videoThumbnail && showPreview) {
                                            // 少し遅延させてDOMの更新を待つ
                                            setTimeout(() => {
                                                const ctx = canvas.getContext('2d')
                                                if (ctx) {
                                                    // 実際の表示サイズを取得
                                                    const rect = canvas.getBoundingClientRect()
                                                    canvas.width = rect.width
                                                    canvas.height = rect.height

                                                    drawPreviewOverlay(canvas, ctx)
                                                }
                                            }, 50)
                                        }
                                    }}
                                    className="absolute inset-0 w-full h-full pointer-events-none"
                                    style={{
                                        imageRendering: 'pixelated',
                                        zIndex: 10
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="text-center text-gray-400">
                                <Video className="h-16 w-16 mx-auto mb-4 opacity-30" />
                                <p className="text-lg font-medium">動画をアップロードしてください</p>
                                <p className="text-sm opacity-75 mt-2">アップロード後にプレビューでパラメータを調整できます</p>
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
