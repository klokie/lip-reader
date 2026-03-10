export type ModelState = "not-loaded" | "loading" | "ready" | "error"

export type InferenceResult = {
  text: string
  confidence: number
}

export type ModelInference = {
  state: ModelState
  load: (modelUrl: string) => Promise<void>
  infer: (
    frames: Float32Array,
    landmarks: Float32Array,
  ) => Promise<InferenceResult>
  dispose: () => void
}

/**
 * Stub implementation — replace with ONNX Runtime Web inference
 * once LipCoordNet has been converted to ONNX format.
 *
 * Expected model input:
 *   video:     Float32[1, 1, 75, 64, 128]  (grayscale frames)
 *   landmarks: Float32[1, 75, 40, 2]       (lip landmark coords)
 *
 * Expected output:
 *   probabilities: Float32[1, 75, 28]  (CTC output — 27 chars + blank)
 */
export const createModelInference = (): ModelInference => ({
  state: "not-loaded" as ModelState,

  load: async (_modelUrl: string) => {
    // const ort = await import("onnxruntime-web")
    // const session = await ort.InferenceSession.create(modelUrl)
  },

  infer: async (
    _frames: Float32Array,
    _landmarks: Float32Array,
  ): Promise<InferenceResult> => {
    return { text: "", confidence: 0 }
  },

  dispose: () => {},
})
