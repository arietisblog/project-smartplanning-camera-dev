interface ProgressIndicatorProps {
    currentStep: 'upload' | 'preview' | 'detection'
}

export default function ProgressIndicator({ currentStep }: ProgressIndicatorProps) {
    return (
        <div className="flex justify-center items-center space-x-4 mb-8">
            <div className={`flex items-center space-x-2 ${currentStep === 'upload' ? 'text-purple-400' : currentStep === 'preview' || currentStep === 'detection' ? 'text-green-400' : 'text-gray-500'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'upload' ? 'bg-purple-500' : currentStep === 'preview' || currentStep === 'detection' ? 'bg-green-500' : 'bg-gray-600'}`}>
                    <span className="text-white font-bold text-sm">1</span>
                </div>
                <span className="font-semibold">動画アップロード</span>
            </div>

            <div className={`w-12 h-0.5 ${currentStep === 'preview' || currentStep === 'detection' ? 'bg-green-500' : 'bg-gray-600'}`}></div>

            <div className={`flex items-center space-x-2 ${currentStep === 'preview' ? 'text-purple-400' : currentStep === 'detection' ? 'text-green-400' : 'text-gray-500'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'preview' ? 'bg-purple-500' : currentStep === 'detection' ? 'bg-green-500' : 'bg-gray-600'}`}>
                    <span className="text-white font-bold text-sm">2</span>
                </div>
                <span className="font-semibold">パラメータ設定</span>
            </div>

            <div className={`w-12 h-0.5 ${currentStep === 'detection' ? 'bg-green-500' : 'bg-gray-600'}`}></div>

            <div className={`flex items-center space-x-2 ${currentStep === 'detection' ? 'text-purple-400' : 'text-gray-500'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'detection' ? 'bg-purple-500' : 'bg-gray-600'}`}>
                    <span className="text-white font-bold text-sm">3</span>
                </div>
                <span className="font-semibold">検知実行</span>
            </div>
        </div>
    )
}
