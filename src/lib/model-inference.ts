import { MODEL_FRAME_COUNT, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT } from "./lip-processor"
import { ctcGreedyDecode } from "./ctc-decoder"

export type ModelState = "not-loaded" | "loading" | "ready" | "error"

export type InferenceResult = {
  text: string
  confidence: number
  latencyMs: number
}

const NUM_CLASSES = 28
const CHANNELS = 3

export type ModelInference = {
  state: ModelState
  load: () => Promise<void>
  infer: (
    videoTensor: Float32Array,
    coordsTensor: Float32Array,
  ) => Promise<InferenceResult>
  dispose: () => void
}

export const createModelInference = (): ModelInference => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let session: any = null
  let ort: typeof import("onnxruntime-web") | null = null
  const state: { current: ModelState } = { current: "not-loaded" }

  return {
    get state() {
      return state.current
    },

    load: async () => {
      if (state.current === "ready" || state.current === "loading") return
      state.current = "loading"

      ort = await import("onnxruntime-web")
      ort.env.wasm.numThreads = 1
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/"

      session = await ort.InferenceSession.create("/models/lipcoordnet_int8.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      })
      state.current = "ready"
    },

    infer: async (
      videoTensor: Float32Array,
      coordsTensor: Float32Array,
    ): Promise<InferenceResult> => {
      if (!session || !ort) {
        throw new Error("Model not loaded")
      }

      const start = performance.now()

      const videoInput = new ort.Tensor(
        "float32",
        videoTensor,
        [1, CHANNELS, MODEL_FRAME_COUNT, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
      )

      const coordsInput = new ort.Tensor(
        "float32",
        coordsTensor,
        [1, MODEL_FRAME_COUNT, 20, 2],
      )

      const results = await session.run({
        video: videoInput,
        coords: coordsInput,
      })

      const latencyMs = performance.now() - start
      const output = results.output.data as Float32Array

      const text = ctcGreedyDecode(output, MODEL_FRAME_COUNT, NUM_CLASSES)

      let totalMax = 0
      for (let t = 0; t < MODEL_FRAME_COUNT; t++) {
        const offset = t * NUM_CLASSES
        let maxVal = output[offset]
        for (let c = 1; c < NUM_CLASSES; c++) {
          if (output[offset + c] > maxVal) maxVal = output[offset + c]
        }
        totalMax += maxVal
      }
      const confidence = totalMax / MODEL_FRAME_COUNT

      return { text, confidence, latencyMs }
    },

    dispose: () => {
      session?.release()
      session = null
      ort = null
      state.current = "not-loaded"
    },
  }
}
