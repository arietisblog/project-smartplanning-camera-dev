import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Upload } from 'lucide-react'

interface UploadStepProps {
    videoFile: File | null
    onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void
}

export default function UploadStep({ videoFile, onFileUpload }: UploadStepProps) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 設定パネル */}
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
                <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
                    <CardTitle className="flex items-center gap-3 text-white">
                        <Upload className="h-5 w-5 text-purple-400" />
                        <span className="text-lg font-semibold">
                            ステップ1: 動画アップロード
                        </span>
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                        検知したい動画ファイルをアップロードしてください
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 p-6">
                    {/* 動画アップロード */}
                    <div className="space-y-2">
                        <Label htmlFor="video-upload" className="text-white font-semibold flex items-center gap-2">
                            <Upload className="h-4 w-4 text-blue-400" />
                            動画ファイル
                        </Label>
                        <div className="space-y-2">
                            <Input
                                id="video-upload"
                                type="file"
                                accept="video/*"
                                onChange={onFileUpload}
                                className="mt-1 bg-slate-800 border-slate-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500 file:bg-slate-700 file:text-white file:border-0 file:rounded-md file:px-3 file:py-1 file:mr-3 file:text-sm"
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
                </CardContent>
            </Card>

            {/* 動画表示パネル */}
            <Card className="bg-slate-800/50 border border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
                <CardHeader className="bg-slate-800 rounded-t-lg border-b border-slate-700">
                    <CardTitle className="flex items-center gap-3 text-white">
                        <Upload className="h-5 w-5 text-purple-400" />
                        <span className="text-lg font-semibold">
                            動画プレビュー
                        </span>
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                        動画をアップロードすると、ここにプレビューが表示されます
                    </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                    <div className="aspect-video bg-slate-900 rounded-xl flex items-center justify-center border border-slate-700">
                        <div className="text-center text-gray-400">
                            <Upload className="h-16 w-16 mx-auto mb-4 opacity-30" />
                            <p className="text-lg font-medium">動画をアップロードしてください</p>
                            <p className="text-sm opacity-75 mt-2">アップロード後にプレビューでパラメータを調整できます</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
