type ResultsDisplayProps = {
  text: string
  isInferring: boolean
  latencyMs?: number
  frameCount: number
  totalFrames: number
}

export const ResultsDisplay = ({
  text,
  isInferring,
  latencyMs,
  frameCount,
  totalFrames,
}: ResultsDisplayProps) => {
  const hasText = text.length > 0
  const progress = Math.min(frameCount / totalFrames, 1)

  return (
    <div className="w-full max-w-2xl space-y-2">
      <div className="rounded-xl border border-border bg-surface px-5 py-4">
        {hasText ? (
          <p className="font-mono text-lg tracking-wide text-accent">{text}</p>
        ) : (
          <p className="font-mono text-sm text-muted">
            {isInferring
              ? "Reading lips\u2026"
              : frameCount > 0
                ? `Buffering frames\u2026 ${frameCount}/${totalFrames}`
                : "Waiting for face detection\u2026"}
          </p>
        )}
      </div>

      <div className="flex items-center justify-between px-1 font-mono text-xs text-muted">
        <div className="flex items-center gap-3">
          <div className="h-1 w-24 overflow-hidden rounded-full bg-border">
            <div
              className="h-full bg-accent/50 transition-all duration-200"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          <span>{frameCount}/{totalFrames}</span>
        </div>
        {latencyMs != null && latencyMs > 0 && (
          <span>{Math.round(latencyMs)}ms</span>
        )}
      </div>

      {hasText && (
        <p className="px-1 text-xs text-muted/70">
          Trained on GRID corpus &mdash; recognizes: &lt;command&gt; &lt;color&gt; &lt;prep&gt; &lt;letter&gt; &lt;digit&gt; &lt;adverb&gt;.{" "}
          Try &ldquo;set blue by A four please&rdquo;
        </p>
      )}
    </div>
  )
}
