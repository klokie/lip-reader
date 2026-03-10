"use client"

type ControlsProps = {
  onStart: () => void
  onStop: () => void
  isActive: boolean
  isLoading: boolean
}

export const Controls = ({
  onStart,
  onStop,
  isActive,
  isLoading,
}: ControlsProps) => (
  <div className="flex gap-3">
    {!isActive ? (
      <button
        onClick={onStart}
        disabled={isLoading}
        className="cursor-pointer rounded-full bg-accent px-6 py-2.5 text-sm font-medium text-black transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? "Requesting access\u2026" : "Start Camera"}
      </button>
    ) : (
      <button
        onClick={onStop}
        className="cursor-pointer rounded-full border border-border bg-surface px-6 py-2.5 text-sm font-medium text-foreground transition-all hover:border-foreground/20"
      >
        Stop
      </button>
    )}
  </div>
)
