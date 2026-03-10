"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createModelInference, type ModelState, type InferenceResult } from "@/lib/model-inference"
import { createFrameBuffer } from "@/lib/frame-buffer"
import { extractLipROI, imageDataToBGR, MODEL_FRAME_COUNT } from "@/lib/lip-processor"

export type LipReaderState = {
  modelState: ModelState
  modelError: string | null
  result: InferenceResult | null
  isInferring: boolean
  frameCount: number
}

export const useLipReader = (
  videoRef: React.RefObject<HTMLVideoElement | null>,
  faceLandmarks: React.MutableRefObject<Array<{ x: number; y: number; z: number }> | null>,
  isActive: boolean,
  faceDetected: boolean,
) => {
  const inferenceRef = useRef(createModelInference())
  const bufferRef = useRef(createFrameBuffer())
  const inferringRef = useRef(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const frameCountRef = useRef(0)

  const [modelState, setModelState] = useState<ModelState>("not-loaded")
  const [modelError, setModelError] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [isInferring, setIsInferring] = useState(false)
  const [frameCount, setFrameCount] = useState(0)

  const loadModel = useCallback(async () => {
    const inference = inferenceRef.current
    if (inference.state === "ready" || inference.state === "loading") return

    setModelState("loading")
    setModelError(null)
    try {
      await inference.load()
      setModelState("ready")
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load model"
      setModelError(msg)
      setModelState("error")
      throw err
    }
  }, [])

  useEffect(() => {
    if (isActive && inferenceRef.current.state === "not-loaded") {
      loadModel().catch(() => {})
    }
  }, [isActive, loadModel])

  useEffect(() => {
    if (!isActive || !faceDetected || modelState !== "ready") {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      return
    }

    intervalRef.current = setInterval(() => {
      const video = videoRef.current
      const landmarks = faceLandmarks.current
      if (!video || !landmarks || video.readyState < 2) return
      if (inferringRef.current) return

      const roi = extractLipROI(video, landmarks)
      if (!roi) return

      const bgr = imageDataToBGR(roi.imageData)
      bufferRef.current.push(bgr, roi.landmarks)
      frameCountRef.current = bufferRef.current.count

      // Only update React state every 5 frames to reduce re-renders
      if (frameCountRef.current % 5 === 0 || bufferRef.current.isFull()) {
        setFrameCount(frameCountRef.current)
      }

      if (bufferRef.current.isFull()) {
        inferringRef.current = true
        setIsInferring(true)

        const videoTensor = bufferRef.current.getVideoTensor()
        const coordsTensor = bufferRef.current.getCoordsTensor()

        // Reset buffer immediately so fresh frames start accumulating
        bufferRef.current.reset()
        frameCountRef.current = 0
        setFrameCount(0)

        inferenceRef.current
          .infer(videoTensor, coordsTensor)
          .then((r) => {
            setResult(r)
          })
          .catch((err) => {
            console.warn("Inference failed:", err)
          })
          .finally(() => {
            inferringRef.current = false
            setIsInferring(false)
          })
      }
    }, 40) // ~25fps

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [isActive, faceDetected, modelState, videoRef, faceLandmarks])

  useEffect(() => {
    if (!faceDetected) {
      bufferRef.current.reset()
      frameCountRef.current = 0
      setFrameCount(0)
    }
  }, [faceDetected])

  return {
    modelState,
    modelError,
    result,
    isInferring,
    frameCount,
  }
}
