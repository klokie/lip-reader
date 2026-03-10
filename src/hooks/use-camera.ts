"use client"

import { useCallback, useRef, useState } from "react"

export type CameraState = "idle" | "requesting" | "active" | "error"

export const useCamera = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [state, setState] = useState<CameraState>("idle")
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(async () => {
    setState("requesting")
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setState("active")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera access denied")
      setState("error")
    }
  }, [])

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setState("idle")
    setError(null)
  }, [])

  return { videoRef, state, error, start, stop }
}
