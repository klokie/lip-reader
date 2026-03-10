import type { CameraState } from "@/hooks/use-camera"
import type { LandmarkerState } from "@/hooks/use-face-landmarker"
import type { ModelState } from "@/lib/model-inference"

type StatusDotState = "active" | "loading" | "inactive" | "error"

const dotColor: Record<StatusDotState, string> = {
  active: "bg-accent",
  loading: "bg-yellow-400 animate-pulse",
  inactive: "bg-muted/50",
  error: "bg-red-400",
}

const StatusDot = ({ state }: { state: StatusDotState }) => (
  <span className={`inline-block h-1.5 w-1.5 rounded-full ${dotColor[state]}`} />
)

const cameraToState = (s: CameraState): StatusDotState => {
  if (s === "active") return "active"
  if (s === "requesting") return "loading"
  if (s === "error") return "error"
  return "inactive"
}

const landmarkerToState = (s: LandmarkerState): StatusDotState => {
  if (s === "ready") return "active"
  if (s === "loading") return "loading"
  return "error"
}

const modelToState = (s: ModelState): StatusDotState => {
  if (s === "ready") return "active"
  if (s === "loading") return "loading"
  if (s === "error") return "error"
  return "inactive"
}

type StatusBarProps = {
  camera: CameraState
  landmarker: LandmarkerState
  model: ModelState
  faceDetected: boolean
}

export const StatusBar = ({
  camera,
  landmarker,
  model,
  faceDetected,
}: StatusBarProps) => (
  <div className="flex items-center gap-4 font-mono text-xs text-muted">
    <span className="flex items-center gap-1.5">
      <StatusDot state={cameraToState(camera)} /> Camera
    </span>
    <span className="flex items-center gap-1.5">
      <StatusDot state={landmarkerToState(landmarker)} /> MediaPipe
    </span>
    <span className="flex items-center gap-1.5">
      <StatusDot state={faceDetected ? "active" : "inactive"} /> Face
    </span>
    <span className="flex items-center gap-1.5">
      <StatusDot state={modelToState(model)} /> Model
    </span>
  </div>
)
