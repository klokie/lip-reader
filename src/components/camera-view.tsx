"use client"

import type { RefObject } from "react"

type CameraViewProps = {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null>
  isActive: boolean
  faceDetected: boolean
}

export const CameraView = ({
  videoRef,
  canvasRef,
  isActive,
  faceDetected,
}: CameraViewProps) => (
  <div
    className={`relative min-h-0 w-full flex-1 overflow-hidden rounded-2xl border-2 transition-all duration-700 ${
      faceDetected
        ? "border-accent/30 detecting"
        : isActive
          ? "border-border"
          : "border-border/50"
    }`}
  >
    <video
      ref={videoRef}
      className="mirror h-full w-full bg-black object-cover"
      playsInline
      muted
    />
    <canvas
      ref={canvasRef}
      className="mirror absolute inset-0 h-full w-full"
    />
    {!isActive && (
      <div className="absolute inset-0 flex items-center justify-center bg-surface">
        <div className="text-center">
          <div className="mb-2 text-4xl opacity-30">&#9673;</div>
          <p className="text-sm text-muted">Start camera to begin</p>
        </div>
      </div>
    )}
  </div>
)
