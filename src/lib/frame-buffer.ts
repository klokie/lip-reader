import { MODEL_FRAME_COUNT, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT } from "./lip-processor"

const CHANNELS = 3
const PIXELS = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT
const FRAME_SIZE = CHANNELS * PIXELS
const COORD_POINTS = 20
const COORD_DIMS = 2
const COORD_FRAME_SIZE = COORD_POINTS * COORD_DIMS

export type FrameBuffer = {
  count: number
  isFull: () => boolean
  push: (rgb: Float32Array, landmarks: Array<{ x: number; y: number }>) => void
  getVideoTensor: () => Float32Array
  getCoordsTensor: () => Float32Array
  reset: () => void
}

/**
 * Circular buffer that accumulates MODEL_FRAME_COUNT frames of video + landmarks.
 * When full, getVideoTensor/getCoordsTensor return data shaped for the model.
 */
export const createFrameBuffer = (): FrameBuffer => {
  const videoData = new Float32Array(MODEL_FRAME_COUNT * FRAME_SIZE)
  const coordsData = new Float32Array(MODEL_FRAME_COUNT * COORD_FRAME_SIZE)
  let writeIndex = 0
  let count = 0

  return {
    get count() {
      return count
    },

    isFull: () => count >= MODEL_FRAME_COUNT,

    push: (rgb: Float32Array, landmarks: Array<{ x: number; y: number }>) => {
      const frameOffset = writeIndex * FRAME_SIZE
      videoData.set(rgb, frameOffset)

      const coordOffset = writeIndex * COORD_FRAME_SIZE
      for (let i = 0; i < Math.min(landmarks.length, COORD_POINTS); i++) {
        coordsData[coordOffset + i * COORD_DIMS] = landmarks[i].x
        coordsData[coordOffset + i * COORD_DIMS + 1] = landmarks[i].y
      }

      writeIndex = (writeIndex + 1) % MODEL_FRAME_COUNT
      if (count < MODEL_FRAME_COUNT) count++
    },

    /**
     * Returns [1, 3, 75, 64, 128] — batch × channels × frames × H × W
     *
     * The internal buffer stores frames as [frame, channel, pixel].
     * We need to interleave them into [channel, frame, pixel] ordering
     * for the Conv3d input.
     */
    getVideoTensor: () => {
      const tensor = new Float32Array(MODEL_FRAME_COUNT * FRAME_SIZE)

      for (let f = 0; f < MODEL_FRAME_COUNT; f++) {
        const srcFrame = ((writeIndex - MODEL_FRAME_COUNT + f) % MODEL_FRAME_COUNT + MODEL_FRAME_COUNT) % MODEL_FRAME_COUNT
        for (let c = 0; c < CHANNELS; c++) {
          const srcOffset = srcFrame * FRAME_SIZE + c * PIXELS
          const dstOffset = c * MODEL_FRAME_COUNT * PIXELS + f * PIXELS
          tensor.set(videoData.subarray(srcOffset, srcOffset + PIXELS), dstOffset)
        }
      }

      return tensor
    },

    /**
     * Returns [1, 75, 20, 2] — batch × frames × landmarks × xy
     */
    getCoordsTensor: () => {
      const tensor = new Float32Array(MODEL_FRAME_COUNT * COORD_FRAME_SIZE)

      for (let f = 0; f < MODEL_FRAME_COUNT; f++) {
        const srcFrame = ((writeIndex - MODEL_FRAME_COUNT + f) % MODEL_FRAME_COUNT + MODEL_FRAME_COUNT) % MODEL_FRAME_COUNT
        const srcOffset = srcFrame * COORD_FRAME_SIZE
        const dstOffset = f * COORD_FRAME_SIZE
        tensor.set(coordsData.subarray(srcOffset, srcOffset + COORD_FRAME_SIZE), dstOffset)
      }

      return tensor
    },

    reset: () => {
      videoData.fill(0)
      coordsData.fill(0)
      writeIndex = 0
      count = 0
    },
  }
}
