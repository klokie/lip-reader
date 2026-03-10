# Lip Reader

Real-time lip reading in the browser using machine learning. No server required — all inference runs client-side via WebAssembly.

![License](https://img.shields.io/badge/license-MIT-blue)

## How It Works

```
Camera → MediaPipe Face Landmarker → Lip ROI + Landmarks → LipCoordNet (ONNX) → CTC Decode → Text
```

1. **Camera** captures video via `getUserMedia`
2. **MediaPipe Face Landmarker** detects 468 face landmarks in real-time, extracts lip region
3. **LipCoordNet** (ONNX model via WebAssembly) reads lip movements from cropped frames + landmark coordinates
4. **CTC decoder** converts model output probabilities to text

Currently implemented: steps 1-2 (camera + face/lip detection). Model inference is stubbed pending ONNX conversion of [LipCoordNet](https://huggingface.co/SilentSpeak/LipCoordNet).

## Demo

Start the camera and the app detects your face in real-time, highlighting lip landmarks with a cyan overlay and dashed bounding box around the lip region.

## Stack

- **Framework:** Next.js 16 + TypeScript + Tailwind v4
- **Face Detection:** [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js) (468 landmarks, 30-60fps)
- **ML Runtime:** [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM/WebGPU)
- **Target Model:** [SilentSpeak/LipCoordNet](https://huggingface.co/SilentSpeak/LipCoordNet) (1.7% WER, 0.6% CER)
- **Deploy:** Cloudflare Pages (static export)

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and grant camera access.

## Build & Deploy

Static export for Cloudflare Pages:

```bash
npm run build
```

Output directory: `out/`

Cloudflare Pages config:

- **Build command:** `npm run build`
- **Output directory:** `out`

The `public/_headers` file configures `Cross-Origin-Embedder-Policy` and `Cross-Origin-Opener-Policy` for WASM threading support.

## Project Structure

```
src/
├── app/
│   ├── globals.css              # Dark theme, animations
│   ├── layout.tsx               # Metadata + fonts
│   └── page.tsx                 # Main composition
├── components/
│   ├── camera-view.tsx          # Video + canvas overlay
│   ├── controls.tsx             # Start/stop
│   ├── status-bar.tsx           # System status indicators
│   └── results-display.tsx      # Recognized text output
├── hooks/
│   ├── use-camera.ts            # getUserMedia management
│   └── use-face-landmarker.ts   # MediaPipe init + detection loop
└── lib/
    ├── lip-processor.ts         # Lip ROI extraction + grayscale
    └── model-inference.ts       # ONNX inference stub
```

## Roadmap

- [x] Camera access + live video feed
- [x] Real-time face landmark detection (MediaPipe)
- [x] Lip region highlighting + bounding box
- [x] Status indicators (Camera, MediaPipe, Face, Model)
- [x] Cloudflare Pages static export config
- [ ] Convert LipCoordNet from PyTorch to ONNX
- [ ] Quantize model (int8) for browser delivery
- [ ] Wire ONNX Runtime Web inference
- [ ] CTC decoder for output → text
- [ ] Frame buffering pipeline (75 frames @ 25fps)
- [ ] Real-time end-to-end lip reading

## License

MIT
