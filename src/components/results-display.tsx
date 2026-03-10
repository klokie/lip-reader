type ResultsDisplayProps = {
  text: string
}

export const ResultsDisplay = ({ text }: ResultsDisplayProps) => {
  if (!text) return null

  return (
    <div className="w-full max-w-2xl rounded-xl border border-border bg-surface px-5 py-4">
      <p className="font-mono text-lg tracking-wide text-accent">{text}</p>
    </div>
  )
}
