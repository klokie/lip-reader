"use client"

import { useRef } from "react"
import { useCamera } from "@/hooks/use-camera"
import { useFaceLandmarker } from "@/hooks/use-face-landmarker"
import { CameraView } from "@/components/camera-view"
import { StatusBar } from "@/components/status-bar"
import { Controls } from "@/components/controls"
import { ResultsDisplay } from "@/components/results-display"

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
  } = useFaceLandmarker(videoRef, canvasRef, cameraState === "active")

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
        model="not-loaded"
        faceDetected={faceDetected}
      />

      {(cameraError || landmarkerError) && (
        <p className="max-w-md text-center text-sm text-red-400">
          {cameraError || landmarkerError}
        </p>
      )}

      <Controls
        onStart={start}
        onStop={stop}
        isActive={cameraState === "active"}
        isLoading={cameraState === "requesting"}
      />

      <ResultsDisplay text="" />
    </main>
  )
}
