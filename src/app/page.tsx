"use client"

import { useRef } from "react"
import { useCamera } from "@/hooks/use-camera"
import { useFaceLandmarker } from "@/hooks/use-face-landmarker"
import { useLipReader } from "@/hooks/use-lip-reader"
import { CameraView } from "@/components/camera-view"
import { StatusBar } from "@/components/status-bar"
import { Controls } from "@/components/controls"
import { ResultsDisplay } from "@/components/results-display"
import { MODEL_FRAME_COUNT } from "@/lib/lip-processor"

export default function Home() {
  const {
    videoRef,
    state: cameraState,
    error: cameraError,
    start,
    stop,
  } = useCamera()

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const {
    state: landmarkerState,
    error: landmarkerError,
    faceDetected,
    faceLandmarksRef,
  } = useFaceLandmarker(videoRef, canvasRef, cameraState === "active")

  const {
    modelState,
    modelError,
    result,
    isInferring,
    frameCount,
  } = useLipReader(videoRef, faceLandmarksRef, cameraState === "active", faceDetected)

  return (
    <main className="flex h-dvh flex-col items-center gap-3 px-3 py-3 sm:px-4 sm:py-4">
      <header className="flex w-full items-baseline justify-between">
        <h1 className="text-lg font-semibold tracking-tight">
          Lip Reader
        </h1>
        <p className="text-xs text-muted">
          runs entirely in your browser
        </p>
      </header>

      <CameraView
        videoRef={videoRef}
        canvasRef={canvasRef}
        isActive={cameraState === "active"}
        faceDetected={faceDetected}
      />

      <StatusBar
        camera={cameraState}
        landmarker={landmarkerState}
        model={modelState}
        faceDetected={faceDetected}
      />

      {(cameraError || landmarkerError || modelError) && (
        <p className="max-w-md text-center text-sm text-red-400">
          {cameraError || landmarkerError || modelError}
        </p>
      )}

      <Controls
        onStart={start}
        onStop={stop}
        isActive={cameraState === "active"}
        isLoading={cameraState === "requesting"}
      />

      <ResultsDisplay
        text={result?.text ?? ""}
        isInferring={isInferring}
        latencyMs={result?.latencyMs}
        frameCount={frameCount}
        totalFrames={MODEL_FRAME_COUNT}
      />
    </main>
  )
}
