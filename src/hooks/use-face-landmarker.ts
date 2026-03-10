"use client"

import { useEffect, useRef, useState, type RefObject } from "react"
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision"
import { ALL_LIP_INDICES } from "@/lib/lip-processor"

export type LandmarkerState = "loading" | "ready" | "error"

const mediaPipeNoise = /XNNPACK|TensorFlow Lite|Feedback manager|acceleration to xnnpack/i

const patchConsoleForMediaPipe = () => {
  const originalError = console.error
  const originalWarn = console.warn
  const filter =
    (original: (...args: unknown[]) => void) =>
    (...args: unknown[]) => {
      if (mediaPipeNoise.test(String(args[0] ?? ""))) return
      original.apply(console, args)
    }
  console.error = filter(originalError)
  console.warn = filter(originalWarn)
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null

const initLandmarker = (): Promise<FaceLandmarker> => {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      patchConsoleForMediaPipe()
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
      )
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: false,
        outputFaceBlendshapes: false,
      })
    })()
  }
  return landmarkerPromise
}

export const useFaceLandmarker = (
  videoRef: RefObject<HTMLVideoElement | null>,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  isActive: boolean,
) => {
  const landmarkerRef = useRef<FaceLandmarker | null>(null)
  const rafRef = useRef<number>(0)
  const faceLandmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const [state, setState] = useState<LandmarkerState>("loading")
  const [error, setError] = useState<string | null>(null)
  const [faceDetected, setFaceDetected] = useState(false)

  useEffect(() => {
    let active = true

    initLandmarker()
      .then((landmarker) => {
        if (active) {
          landmarkerRef.current = landmarker
          setState("ready")
        }
      })
      .catch((err) => {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to load face landmarker",
          )
          setState("error")
        }
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!isActive || state !== "ready") {
      setFaceDetected(false)
      return
    }

    let lastVideoTime = -1
    let lastDetected = false
    let drawingUtils: DrawingUtils | null = null

    const detect = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      const landmarker = landmarkerRef.current

      if (!video || !canvas || !landmarker || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(detect)
        return
      }

      if (video.currentTime === lastVideoTime) {
        rafRef.current = requestAnimationFrame(detect)
        return
      }
      lastVideoTime = video.currentTime

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        drawingUtils = null
      }

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      if (!drawingUtils) {
        drawingUtils = new DrawingUtils(ctx)
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      try {
        const result = landmarker.detectForVideo(video, Date.now())

        const detected = (result.faceLandmarks?.length ?? 0) > 0

        if (detected !== lastDetected) {
          setFaceDetected(detected)
          lastDetected = detected
        }

        faceLandmarksRef.current = result.faceLandmarks?.[0] ?? null

        if (result.faceLandmarks) {
          for (const face of result.faceLandmarks) {
            drawingUtils.drawConnectors(
              face,
              FaceLandmarker.FACE_LANDMARKS_TESSELATION,
              { color: "rgba(255,255,255,0.04)", lineWidth: 0.5 },
            )

            drawingUtils.drawConnectors(
              face,
              FaceLandmarker.FACE_LANDMARKS_LIPS,
              { color: "#00ffd5", lineWidth: 2 },
            )

            drawLipBoundingBox(ctx, face, canvas.width, canvas.height)
          }
        }
      } catch (err) {
        console.warn("detectForVideo failed:", err)
      }

      rafRef.current = requestAnimationFrame(detect)
    }

    rafRef.current = requestAnimationFrame(detect)
    return () => cancelAnimationFrame(rafRef.current)
  }, [isActive, state, videoRef, canvasRef])

  return { state, error, faceDetected, faceLandmarksRef }
}

const drawLipBoundingBox = (
  ctx: CanvasRenderingContext2D,
  face: NormalizedLandmark[],
  width: number,
  height: number,
) => {
  const lipLandmarks = ALL_LIP_INDICES.map((i) => face[i]).filter(Boolean)
  if (lipLandmarks.length === 0) return

  const xs = lipLandmarks.map((l) => l.x)
  const ys = lipLandmarks.map((l) => l.y)
  const padding = 0.03

  const x = Math.max(0, Math.min(...xs) - padding) * width
  const y = Math.max(0, Math.min(...ys) - padding) * height
  const w = (Math.max(...xs) - Math.min(...xs) + 2 * padding) * width
  const h = (Math.max(...ys) - Math.min(...ys) + 2 * padding) * height

  ctx.strokeStyle = "rgba(0, 255, 213, 0.25)"
  ctx.lineWidth = 1
  ctx.setLineDash([4, 4])
  ctx.strokeRect(x, y, w, h)
  ctx.setLineDash([])
}
