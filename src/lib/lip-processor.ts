export const MODEL_INPUT_WIDTH = 128
export const MODEL_INPUT_HEIGHT = 64
export const MODEL_FRAME_COUNT = 75

/**
 * MediaPipe indices approximating dlib 68 landmarks 17-67 (excluding jaw 0-16).
 * Source: community-verified mapping from StackOverflow #71293574.
 */
const DLIB_17_TO_67_MEDIAPIPE = [
  71, 63, 105, 66, 107,        // dlib 17-21: right eyebrow
  336, 296, 334, 293, 301,     // dlib 22-26: left eyebrow
  168, 197, 5, 4,              // dlib 27-30: nose bridge
  75, 97, 2, 326, 305,        // dlib 31-35: nose bottom
  33, 160, 158, 133, 153, 144, // dlib 36-41: right eye
  362, 385, 387, 263, 373, 380,// dlib 42-47: left eye
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, // dlib 48-59: outer lip
  78, 82, 13, 312, 308, 317, 14, 87, // dlib 60-67: inner lip
]

/**
 * 20 lip landmarks matching dlib 48-67 (outer lip 12 + inner lip 8).
 */
const LIP_LANDMARK_MEDIAPIPE = [
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
  78, 82, 13, 312, 308, 317, 14, 87,
]

/**
 * Full set of lip indices for the visualization overlay (bounding box).
 * Uses MediaPipe's own lip connection groups for accurate rendering.
 */
export const ALL_LIP_INDICES = [
  // lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner (deduplicated)
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
  146, 91, 181, 84, 17, 314, 405, 321, 375,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
  95, 88, 178, 87, 14, 317, 402, 318, 324,
]

const CANONICAL_256 = buildCanonicalPositions(256)

function buildCanonicalPositions(size: number): Array<[number, number]> {
  const padding = 0.25
  const rawX = [
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397,
    0.586856, 0.689483, 0.799124, 0.904991, 0.98004,
    0.490127, 0.490127, 0.490127, 0.490127,
    0.36688, 0.426036, 0.490127, 0.554217, 0.613373,
    0.121737, 0.187122, 0.265825, 0.334606, 0.260918, 0.182743,
    0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335,
    0.254149, 0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104,
    0.642159, 0.556721, 0.490127, 0.423532, 0.338094,
    0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689,
  ]
  const rawY = [
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906,
    0.0773906, 0.0344891, 0.0187482, 0.038915, 0.106454,
    0.203352, 0.307009, 0.409805, 0.515625,
    0.587326, 0.609345, 0.628106, 0.609345, 0.587326,
    0.216423, 0.178758, 0.179852, 0.231733, 0.245099, 0.244077,
    0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233,
    0.864805, 0.902192, 0.909281, 0.902192, 0.864805,
    0.784792, 0.778746, 0.785343, 0.778746, 0.784792,
    0.824182, 0.831803, 0.824182,
  ]
  return rawX.map((x, i) => [
    ((x + padding) / (2 * padding + 1)) * size,
    ((rawY[i] + padding) / (2 * padding + 1)) * size,
  ])
}

// Precompute the lip crop region from canonical positions
const INNER_LIP_CANONICAL = CANONICAL_256.slice(43) // dlib 60-67
const LIP_CENTER_X = INNER_LIP_CANONICAL.reduce((s, p) => s + p[0], 0) / INNER_LIP_CANONICAL.length
const LIP_CENTER_Y = INNER_LIP_CANONICAL.reduce((s, p) => s + p[1], 0) / INNER_LIP_CANONICAL.length
const CROP_W = 160
const CROP_H = 80
const CROP_X = Math.round(LIP_CENTER_X) - CROP_W / 2
const CROP_Y = Math.round(LIP_CENTER_Y) - CROP_H / 2

function solveAffine(
  src: Array<[number, number]>,
  dst: Array<[number, number]>,
): [number, number, number, number, number, number] {
  const n = src.length
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0
  let bx0 = 0, bx1 = 0, bx2 = 0
  let by0 = 0, by1 = 0, by2 = 0

  for (let i = 0; i < n; i++) {
    const [x, y] = src[i]
    const [u, v] = dst[i]
    sxx += x * x; sxy += x * y; sx += x
    syy += y * y; sy += y
    bx0 += x * u; bx1 += y * u; bx2 += u
    by0 += x * v; by1 += y * v; by2 += v
  }

  const solve3x3 = (
    mat: number[][],
    rhs: number[],
  ): [number, number, number] => {
    const m = mat.map((r, i) => [...r, rhs[i]])
    for (let col = 0; col < 3; col++) {
      let maxRow = col
      for (let row = col + 1; row < 3; row++) {
        if (Math.abs(m[row][col]) > Math.abs(m[maxRow][col])) maxRow = row
      }
      [m[col], m[maxRow]] = [m[maxRow], m[col]]
      if (Math.abs(m[col][col]) < 1e-12) return [0, 0, 0]
      for (let row = col + 1; row < 3; row++) {
        const f = m[row][col] / m[col][col]
        for (let j = col; j < 4; j++) m[row][j] -= f * m[col][j]
      }
    }
    const r: number[] = [0, 0, 0]
    for (let i = 2; i >= 0; i--) {
      let s = m[i][3]
      for (let j = i + 1; j < 3; j++) s -= m[i][j] * r[j]
      r[i] = s / m[i][i]
    }
    return r as [number, number, number]
  }

  const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]]
  const [m00, m01, m02] = solve3x3(A.map(r => [...r]), [bx0, bx1, bx2])
  const [m10, m11, m12] = solve3x3(A.map(r => [...r]), [by0, by1, by2])
  return [m00, m01, m02, m10, m11, m12]
}

export type LipROI = {
  imageData: ImageData
  landmarks: Array<{ x: number; y: number }>
}

// Persistent canvases — only allocated once, never resized
let alignCanvas: HTMLCanvasElement | null = null
let alignCtx: CanvasRenderingContext2D | null = null
let cropCanvas: HTMLCanvasElement | null = null
let cropCtx: CanvasRenderingContext2D | null = null

function initCanvases() {
  if (!alignCanvas) {
    alignCanvas = document.createElement("canvas")
    alignCanvas.width = 256
    alignCanvas.height = 256
    alignCtx = alignCanvas.getContext("2d", { willReadFrequently: true })
  }
  if (!cropCanvas) {
    cropCanvas = document.createElement("canvas")
    cropCanvas.width = MODEL_INPUT_WIDTH
    cropCanvas.height = MODEL_INPUT_HEIGHT
    cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true })
  }
}

export const extractLipROI = (
  video: HTMLVideoElement,
  faceLandmarks: Array<{ x: number; y: number; z: number }>,
): LipROI | null => {
  const vw = video.videoWidth
  const vh = video.videoHeight
  if (vw === 0 || vh === 0) return null

  initCanvases()
  if (!alignCtx || !cropCtx || !alignCanvas) return null

  const srcPoints: Array<[number, number]> = []
  for (const mpIdx of DLIB_17_TO_67_MEDIAPIPE) {
    const lm = faceLandmarks[mpIdx]
    if (!lm) return null
    srcPoints.push([lm.x * vw, lm.y * vh])
  }

  const [m00, m01, m02, m10, m11, m12] = solveAffine(srcPoints, CANONICAL_256)

  // Warp video to 256×256 aligned face
  alignCtx.clearRect(0, 0, 256, 256)
  alignCtx.setTransform(m00, m10, m01, m11, m02, m12)
  alignCtx.drawImage(video, 0, 0)
  alignCtx.setTransform(1, 0, 0, 1, 0, 0)

  // Crop lip region from aligned face → 128×64
  cropCtx.clearRect(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
  cropCtx.drawImage(
    alignCanvas,
    CROP_X, CROP_Y, CROP_W, CROP_H,
    0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
  )
  const imageData = cropCtx.getImageData(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

  const lipLandmarks = LIP_LANDMARK_MEDIAPIPE.map((mpIdx) => {
    const lm = faceLandmarks[mpIdx]
    return lm ? { x: lm.x, y: lm.y } : { x: 0, y: 0 }
  })

  return { imageData, landmarks: lipLandmarks }
}

export const imageDataToBGR = (imageData: ImageData): Float32Array => {
  const { data, width, height } = imageData
  const pixelCount = width * height
  const bgr = new Float32Array(3 * pixelCount)

  for (let i = 0; i < pixelCount; i++) {
    bgr[i] = data[i * 4 + 2] / 255
    bgr[pixelCount + i] = data[i * 4 + 1] / 255
    bgr[2 * pixelCount + i] = data[i * 4] / 255
  }
  return bgr
}
