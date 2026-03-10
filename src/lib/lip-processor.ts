export const LIP_LANDMARK_INDICES = [
  // outer lip
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
  37, 39, 40, 185,
  // inner lip
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14,
  87, 178, 88, 95,
]

export const MODEL_INPUT_WIDTH = 128
export const MODEL_INPUT_HEIGHT = 64
export const MODEL_FRAME_COUNT = 75 // 3 seconds at 25fps

export type LipROI = {
  imageData: ImageData
  landmarks: Array<{ x: number; y: number }>
}

let cropCanvas: HTMLCanvasElement | null = null

const getCropCanvas = (): HTMLCanvasElement => {
  if (!cropCanvas) {
    cropCanvas = document.createElement("canvas")
  }
  return cropCanvas
}

export const extractLipROI = (
  video: HTMLVideoElement,
  faceLandmarks: Array<{ x: number; y: number; z: number }>,
): LipROI | null => {
  const lipLandmarks = LIP_LANDMARK_INDICES.map((i) => faceLandmarks[i]).filter(
    Boolean,
  )
  if (lipLandmarks.length === 0) return null

  const xs = lipLandmarks.map((l) => l.x * video.videoWidth)
  const ys = lipLandmarks.map((l) => l.y * video.videoHeight)
  const padding = 0.3

  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const w = maxX - minX
  const h = maxY - minY

  const cropX = Math.max(0, minX - w * padding)
  const cropY = Math.max(0, minY - h * padding)
  const cropW = Math.min(video.videoWidth - cropX, w * (1 + 2 * padding))
  const cropH = Math.min(video.videoHeight - cropY, h * (1 + 2 * padding))

  const canvas = getCropCanvas()
  canvas.width = MODEL_INPUT_WIDTH
  canvas.height = MODEL_INPUT_HEIGHT
  const ctx = canvas.getContext("2d", { willReadFrequently: true })
  if (!ctx) return null

  ctx.drawImage(
    video,
    cropX,
    cropY,
    cropW,
    cropH,
    0,
    0,
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
  )
  const imageData = ctx.getImageData(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

  const normalizedLandmarks = lipLandmarks.map((l) => ({
    x: (l.x * video.videoWidth - cropX) / cropW,
    y: (l.y * video.videoHeight - cropY) / cropH,
  }))

  return { imageData, landmarks: normalizedLandmarks }
}

export const imageDataToGrayscale = (imageData: ImageData): Float32Array => {
  const { data, width, height } = imageData
  const gray = new Float32Array(width * height)
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  }
  return gray
}
