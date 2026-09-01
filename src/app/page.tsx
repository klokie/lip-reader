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
        <a href="https://lip-reader.klokie.com" className="text-lg font-semibold tracking-tight hover:text-accent transition-colors">
          Lip Reader
        </a>
        <div className="flex items-center gap-3 text-xs text-muted">
          <span className="hidden sm:inline">runs entirely in your browser</span>
          <a
            href="https://github.com/klokie/lip-reader"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
            title="View source on GitHub"
          >
            <span className="sr-only">View source on GitHub</span>
            <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>
        </div>
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

      <p className="max-w-lg text-center text-xs leading-relaxed text-muted/70">
        Camera feeds MediaPipe for face detection, lip landmarks are extracted
        and aligned, then LipCoordNet (ONNX, 25MB) runs inference via WebAssembly.
        All processing happens on-device &mdash; no data leaves your browser.
      </p>

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
