/**
 * CTC greedy decoder matching the training vocabulary exactly.
 *
 * Index 0 = CTC blank token
 * Index 1-27 = [space, A, B, C, ..., Z]
 */

const LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
const BLANK = 0

/**
 * Greedy CTC decode: argmax per timestep, collapse repeats, strip blanks.
 * Input: Float32Array of shape [frames, numClasses] (flattened from [1, 75, 28]).
 */
export const ctcGreedyDecode = (
  logits: Float32Array,
  numFrames: number,
  numClasses: number,
): string => {
  let prev = -1
  const chars: string[] = []

  for (let t = 0; t < numFrames; t++) {
    const offset = t * numClasses
    let maxIdx = 0
    let maxVal = logits[offset]

    for (let c = 1; c < numClasses; c++) {
      if (logits[offset + c] > maxVal) {
        maxVal = logits[offset + c]
        maxIdx = c
      }
    }

    if (maxIdx !== prev && maxIdx !== BLANK) {
      if (maxIdx >= 1 && maxIdx <= LETTERS.length) {
        chars.push(LETTERS[maxIdx - 1])
      }
    }
    prev = maxIdx
  }

  return chars
    .join("")
    .replace(/\s+/g, " ")
    .trim()
}
